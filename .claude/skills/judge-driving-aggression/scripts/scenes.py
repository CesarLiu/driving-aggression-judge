"""Load a scene file in either format the screens accept, as a ``responsibility.scene.Scene``.

* ScenarioNet / MetaDrive pickles (CAT's ``raw_scenes_500/*.pkl``): read as they are.
* catk's cached WOMD (``catk/data/womd_cache/<split>/<scenario_id>.pkl``, made by catk's
  src/data_preprocess.py): the tracks are complete, but the map is SMART's token form, so it is
  rebuilt here into the MetaDrive layout the CAT code reads:

  - ``map_save.traj_pos`` [n, 3, 2] holds every polyline cut into ~5 m pieces of 3 points, in
    order; pieces that join end to start with the same point and polygon type are put back
    together. Lane ids are lost, so lanes get new ids.
  - point type: lanes 0 freeway, 1 surface street, 2 a lane with a stop sign, 3 bike lane; road
    edges 3-5, road lines 6-8, crosswalks / speed bumps / driveways 9 (one class in catk).
  - catk keeps no lane connections. Exits are rebuilt where one lane starts within 1 m of where
    another ends, running within 45 degrees of it; left/right neighbours where a lane runs within
    20 degrees of another 2.5 to 5 m to its side.
  - stop signs: a lane of type 2 gets a stop sign at its first point.
  - the cache holds PyTorch tensors, so reading it needs ``torch`` installed (ScenarioNet scenes
    do not); without it ``load_scene`` raises ``NeedsTorch``.
  - traffic lights: catk stores only the state at the current step (1.0 s). It is held for the
    whole clip, so a light that changes later is not seen; screen verdicts that rest on a
    signal are less certain in this format.
"""

import pickle
from pathlib import Path

import numpy as np

AGENT_TYPES = {0: "VEHICLE", 1: "PEDESTRIAN", 2: "CYCLIST"}
LANE_TYPES = {0: "LANE_FREEWAY", 1: "LANE_SURFACE_STREET", 2: "LANE_SURFACE_STREET", 3: "LANE_BIKE_LANE"}
EDGE_TYPES = {3: "ROAD_EDGE_BOUNDARY", 4: "ROAD_EDGE_BOUNDARY", 5: "ROAD_EDGE_MEDIAN"}
LINE_TYPES = {6: "ROAD_LINE_BROKEN_SINGLE_WHITE", 7: "ROAD_LINE_SOLID_SINGLE_WHITE", 8: "ROAD_LINE_SOLID_DOUBLE_YELLOW"}
LIGHTS = {1: "LANE_STATE_UNKNOWN", 2: "LANE_STATE_STOP", 3: "LANE_STATE_GO", 4: "LANE_STATE_CAUTION"}


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def is_catk(raw):
    return isinstance(raw, dict) and "map_save" in raw and "agent" in raw


def _polylines(pos, ptype, pl_type, light):
    """Join consecutive 3-point pieces back into polylines: [(points [P, 2], type, pl_type, light)]."""
    out, cur = [], None
    for k in range(len(pos)):
        key = (int(ptype[k]), int(pl_type[k]), int(light[k]))
        if cur is not None and key == cur[1] and np.linalg.norm(pos[k, 0] - cur[0][-1]) < 0.1:
            cur[0].extend([pos[k, 1], pos[k, 2]])
        else:
            if cur is not None:
                out.append((np.array(cur[0]), *cur[1]))
            cur = ([pos[k, 0], pos[k, 1], pos[k, 2]], key)
    if cur is not None:
        out.append((np.array(cur[0]), *cur[1]))
    return out


def _heading(p, at_end):
    d = p[-1] - p[-2] if at_end else p[1] - p[0]
    return np.arctan2(d[1], d[0])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def catk_to_description(raw, scenario_id=None):
    agent = raw["agent"]
    ids = [str(int(i)) for i in _np(agent["id"])]
    pos, head, vel = _np(agent["position"]), _np(agent["heading"]), _np(agent["velocity"])
    valid, shape, kind = _np(agent["valid_mask"]).astype(bool), _np(agent["shape"]), _np(agent["type"])
    role = _np(agent["role"]).astype(bool)
    T = pos.shape[1]
    tracks = {}
    for i, tid in enumerate(ids):
        tracks[tid] = {"type": AGENT_TYPES.get(int(kind[i]), "OTHER"),
                       "state": {"position": pos[i].astype(np.float32), "heading": head[i].astype(np.float32),
                                 "velocity": vel[i].astype(np.float32),
                                 "length": np.full(T, shape[i, 0], np.float32), "width": np.full(T, shape[i, 1], np.float32),
                                 "height": np.full(T, shape[i, 2], np.float32), "valid": valid[i]},
                       "metadata": {"object_id": tid, "type": AGENT_TYPES.get(int(kind[i]), "OTHER")}}
    sdc = [ids[i] for i in np.flatnonzero(role[:, 0])]
    ooi = [ids[i] for i in np.flatnonzero(role[:, 1])]

    ms, pt = raw["map_save"], raw["pt_token"]
    polys = _polylines(_np(ms["traj_pos"]), _np(pt["type"]), _np(pt["pl_type"]), _np(pt["light_type"]))
    features, lights, lanes = {}, {}, []
    for n, (p, ptype, pl_type, light) in enumerate(polys):
        fid = str(n)
        xyz = np.concatenate([p, np.zeros((len(p), 1))], axis=1)
        if pl_type == 0 and len(p) >= 2:
            features[fid] = {"type": LANE_TYPES.get(ptype, "LANE_SURFACE_STREET"), "polyline": xyz,
                             "exit_lanes": [], "entry_lanes": [], "left_neighbor": [], "right_neighbor": []}
            lanes.append(fid)
            if ptype == 2:
                features["stop_" + fid] = {"type": "STOP_SIGN", "position": xyz[0], "lane": [fid]}
            if light in LIGHTS:
                lights[fid] = {"type": "TRAFFIC_LIGHT", "lane": fid, "stop_point": xyz[0],
                               "state": {"object_state": [LIGHTS[light]] * T}}
        elif pl_type == 1:
            features[fid] = {"type": EDGE_TYPES.get(ptype, "ROAD_EDGE_BOUNDARY"), "polyline": xyz}
        elif pl_type == 2:
            features[fid] = {"type": LINE_TYPES.get(ptype, "ROAD_LINE_BROKEN_SINGLE_WHITE"), "polyline": xyz}
        elif pl_type == 3:
            features[fid] = {"type": "CROSSWALK", "polygon": xyz}

    # lane connections, rebuilt from geometry
    pts = {f: features[f]["polyline"][:, :2] for f in lanes}
    starts = np.array([pts[f][0] for f in lanes]).reshape(-1, 2)
    start_h = np.array([_heading(pts[f], False) for f in lanes])
    for a in lanes:
        end, h = pts[a][-1], _heading(pts[a], True)
        near = np.flatnonzero((np.linalg.norm(starts - end, axis=1) < 1.0) & (np.abs(_wrap(start_h - h)) < np.radians(45)))
        for k in near:
            b = lanes[k]
            if b != a:
                features[a]["exit_lanes"].append(b)
                features[b]["entry_lanes"].append(a)
    for a in lanes:
        p = pts[a]
        mid = p[len(p) // 2]
        h = _heading(p[max(0, len(p) // 2 - 1):len(p) // 2 + 2], True) if len(p) >= 3 else _heading(p, True)
        for b in lanes:
            if b == a:
                continue
            q = pts[b]
            d = np.linalg.norm(q - mid, axis=1)
            k = int(np.argmin(d))
            if not 2.5 <= d[k] <= 5.0:
                continue
            kk = min(k + 1, len(q) - 1)
            hb = np.arctan2(*(q[kk] - q[max(kk - 1, 0)])[::-1])
            if abs(_wrap(hb - h)) > np.radians(20):
                continue
            side = np.cos(h) * (q[k, 1] - mid[1]) - np.sin(h) * (q[k, 0] - mid[0])
            features[a]["left_neighbor" if side > 0 else "right_neighbor"].append({"feature_id": b})

    sid = str(raw.get("scenario_id", scenario_id or ""))
    return {"id": sid, "tracks": tracks, "map_features": features, "dynamic_map_states": lights,
            "metadata": {"scenario_id": sid, "sdc_id": sdc[0] if sdc else ids[0], "objects_of_interest": ooi,
                         "source": "catk_cache"}}


class NeedsTorch(RuntimeError):
    """A scene file holds PyTorch tensors (catk's cache) and torch is not installed."""


def load_scene(path, Scene):
    """A Scene from either format; ``Scene`` is responsibility.scene.Scene (passed in so this module
    does not import the scene code)."""
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except ImportError as exc:  # ModuleNotFoundError included
        if "torch" not in str(exc):
            raise
        raise NeedsTorch("%s holds PyTorch tensors (catk's womd_cache): reading it needs torch "
                         "(pip install torch); ScenarioNet scenes do not" % Path(path).name) from exc
    if is_catk(raw):
        return Scene.from_description(catk_to_description(raw, Path(path).stem)), "catk_cache"
    return Scene.from_description(raw), "scenarionet"


PAIR_MODES = ("auto", "ooi", "sdc-nearest")


def pick_pair(scene, mode="auto"):
    """(first, second, how) for the pair to judge, or None.

    ``auto``: the SDC and its object-of-interest partner when the SDC is one of the objects of
    interest (all CAT scenes); else the two objects of interest (WOMD marks pairs without the SDC
    too); else the SDC and the vehicle that comes closest to it. ``ooi`` uses the first two rules
    only, ``sdc-nearest`` the last only. In the screens' output "sdc_*" columns are the first agent
    of the pair and "ooi_*" the second; ``pair`` says how they were chosen.
    """
    sdc, ooi = scene.sdc, list(scene.objects_of_interest)
    if mode in ("auto", "ooi"):
        if sdc in ooi and len(ooi) >= 2:
            return sdc, [i for i in ooi if i != sdc][0], "sdc_with_ooi"
        if len(ooi) >= 2:
            return ooi[0], ooi[1], "ooi_pair"
        if mode == "ooi":
            return None
    best, near = None, np.inf
    for i in range(scene.n_agents):
        if i == sdc or scene.types[i] != "VEHICLE":
            continue
        both = scene.valid[i] & scene.valid[sdc]
        if both.sum() < 10:
            continue
        d = np.linalg.norm(scene.position[i, both, :2] - scene.position[sdc, both, :2], axis=1).min()
        if d < near:
            best, near = i, d
    return None if best is None else (sdc, best, "sdc_nearest")
