"""Screen WOMD-style scenes for vehicles that failed pedestrians and cyclists.

For each vehicle screened (the SDC by default) and every pedestrian and cyclist that comes within
15 m of it, it looks for two events and writes one row per vehicle and VRU with either:

* **crossing**: the VRU's body entered the vehicle's swept path (centreline +- half the vehicle's
  width + half the VRU's + 0.3 m) while crossing it. The profile's ``vru`` block
  (``jurisdiction.vru_duty``) decides what the vehicle owed it from where (marked crosswalk, junction,
  mid-block, off the road), the VRU's state when the vehicle reached the conflict (on the
  crosswalk or carriageway; waiting at its edge), the vehicle's light and whether it had turned.
  A ``violation`` is a vehicle that owed the VRU the way and went through first (or with the VRU in
  its path); it is an ``aggressive_candidate`` if the VRU paid for it: it waited or slowed while
  the vehicle passed, the vehicle passed within ``--close-pet`` s (1 s) of it, or within 1.5 m of it. A
  cyclist crossing like a vehicle (not a turner across a cyclist going straight) is judged by the
  vehicle rules (screen_crossing.screen_pair), reported in ``row_*`` columns. Without a duty, a
  vehicle that drove through within 1 s or 0.5 m of a VRU already on the carriageway at 3 m/s or
  more is an ``endangered`` candidate (due care).
* **passing**: the vehicle overtook a VRU moving its way. The smallest lateral gap while alongside
  (box to box) is compared with the profile's distance (``jurisdiction.passing_distance``: CVC
  21760(c) 3 ft for a cyclist, StVO 5(4) 1.5 m in town and 2 m outside; a 1 m physical fallback
  where the law gives none). Overtaking slower than 1.5 m/s relative is not flagged (CVC 21760(d)).

Rows are candidates, not labels; review them as SKILL.md says. Input as for the other screens.

    python screen_vru.py --scenes <folder-or-file> --profile us-california --out vru.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jurisdiction  # noqa: E402
from lights import infer_lights  # noqa: E402
from scenes import NeedsTorch, load_scene  # noqa: E402
from screen_crossing import DT, scene_files, screen_pair, smooth, stopped_before  # noqa: E402
from responsibility import lanes as lanes_mod  # noqa: E402
from responsibility import right_of_way as R  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

VRU_TYPES = ("PEDESTRIAN", "CYCLIST")
NEAR = 15.0  # m: a VRU never this close to the vehicle is not screened
MARGIN = 0.3  # m added to the swept path's half width
CW_TOL = 1.5  # m: a conflict point this close to a crosswalk polygon is on that crosswalk
WAIT_TOL = 2.5  # m: a VRU this close outside the crosswalk is at its edge
CARRIAGEWAY = 2.0  # m: a VRU this close to a lane centreline is on the carriageway
OFF_ROAD = 3.0  # m: a conflict point further than this from every lane is off the road
JUNCTION_TOL = 6.0  # m: a conflict point this close to where a junction lane starts or ends is at the junction
CROSSING = (math.radians(45.0), math.radians(135.0))  # VRU heading against the path: crossing it
ALONG = math.radians(30.0)  # and running along it
URBAN_KMH = 50.0  # a lane speed limit above this counts as outside town (StVO 5(4))
REACTION, BRAKE = 1.0, 4.0  # s, m/s²: a vehicle closer than its stopping distance cannot stop
MIN_REL = 1.5  # m/s: overtaking slower than this is not flagged
CLOSE_M = 1.5  # m box to box: passing a crossing VRU closer than this cost it (StVO 5(4)'s distance in town)
SAME_EVENT_M = 3.0  # m: rows of the same vehicle and VRU ids this close are one event cut twice
MIN_SEEN = 20  # steps (2 s): a VRU seen for less is never a candidate (flag short_vru_track)
COLS = ["scene", "scenario_id", "format", "jurisdiction", "vehicle_id", "vehicle_is_sdc", "vru_id", "vru_type",
        "event", "where", "vru_state", "vehicle_control", "vehicle_signal", "vehicle_turn", "turning",
        "cyclist_direction", "duty", "rule", "went_first", "gap_s", "min_gap_m", "vehicle_speed",
        "vru_waited", "vru_slowed", "violation", "affected", "affected_by", "aggressive_candidate",
        "candidate_kind", "vehicle_red_entry", "row_case", "row_holder", "row_violation", "passing_gap_m",
        "passing_required_m", "passing_statutory", "passing_rel_speed", "t_vehicle", "conflict_x",
        "conflict_y", "confidence", "flags"]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def inside(poly, q, tol=0.0):
    """Whether point q lies inside polygon [M, 2], or within ``tol`` metres of its edges."""
    x, y = q
    hit, near = False, np.inf
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
        seg = np.array([x2 - x1, y2 - y1])
        t = np.clip(np.dot(q - [x1, y1], seg) / max(np.dot(seg, seg), 1e-12), 0.0, 1.0)
        near = min(near, float(np.linalg.norm(q - ([x1, y1] + t * seg))))
    return hit or near <= tol


class MapInfo:
    """What the screen reads from the map: crosswalk polygons, lane points, junction lane ends,
    and the speed limit of each lane point."""

    def __init__(self, scene, graph):
        f = scene.map_features
        self.crosswalks = [np.asarray(x["polygon"], float)[:, :2] for x in f.values()
                           if x.get("type") == "CROSSWALK" and len(x.get("polygon", [])) >= 3]
        lanes = [k for k in graph.points]
        self.lane_pts = np.concatenate([graph.points[k] for k in lanes]) if lanes else np.zeros((0, 2))
        limits = []
        for k in lanes:
            kmh = float((f.get(k) or f.get(str(k)) or {}).get("speed_limit_kmh", 0) or 0)
            limits.append(np.full(len(graph.points[k]), kmh))
        self.lane_kmh = np.concatenate(limits) if limits else np.zeros(0)
        lights = R._lights(scene)
        stops, _ = R._stop_signs(scene)
        branching = {e for k, ex in graph.exits.items() if len(ex) >= 2 for e in ex}
        entries = {}
        for ex in graph.exits.values():
            for e in ex:
                entries[e] = entries.get(e, 0) + 1
        merging = {k for k, n in entries.items() if n >= 2}
        junction = [k for k in lanes if k in lights or k in stops or k in branching or k in merging]
        self.junction_ends = (np.array([p for k in junction for p in (graph.points[k][0], graph.points[k][-1])])
                              if junction else np.zeros((0, 2)))

    def lane_dist(self, q):
        return float(np.linalg.norm(self.lane_pts - q, axis=1).min()) if len(self.lane_pts) else np.inf

    def speed_limit(self, q):
        if not len(self.lane_pts):
            return 0.0
        return float(self.lane_kmh[int(np.argmin(np.linalg.norm(self.lane_pts - q, axis=1)))])

    def crosswalk_at(self, q):
        for poly in self.crosswalks:
            if inside(poly, q, CW_TOL):
                return poly
        return None

    def where(self, q):
        if self.crosswalk_at(q) is not None:
            return "crosswalk"
        if self.lane_dist(q) > OFF_ROAD:
            return "off_road"
        if len(self.junction_ends) and np.linalg.norm(self.junction_ends - q, axis=1).min() <= JUNCTION_TOL:
            return "intersection"
        return "road"


def motion_heading(pos, k, steps=5):
    """Direction of travel at index k from the displacement over +-``steps`` samples; None when
    it moved less than 0.3 m (standing)."""
    d = pos[min(k + steps, len(pos) - 1)] - pos[max(k - steps, 0)]
    return None if np.linalg.norm(d) < 0.3 else math.atan2(d[1], d[0])


def first_time(idx, values, threshold):
    """Step (as time in s) at which ``values`` first reach ``threshold``; NaN if never."""
    hit = np.flatnonzero(values >= threshold)
    return float(idx[hit[0]]) * DT if hit.size else float("nan")


def vru_state(info, poly, q):
    if poly is not None:
        return "on" if inside(poly, q, 0.3) else "waiting" if inside(poly, q, WAIT_TOL) else ""
    return "on" if info.lane_dist(q) <= CARRIAGEWAY else ""


def box_gap(scene, v, u, steps, L, W, wu):
    """Per common step, the gap (m) between the vehicle's box and a VRU disc of diameter ``wu``,
    with the VRU's position in the vehicle's frame (longitudinal, lateral)."""
    d = scene.position[u, steps, :2] - scene.position[v, steps, :2]
    h = scene.heading[v, steps]
    lon = d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h)
    lat = -d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)
    gap = np.hypot(np.maximum(np.abs(lon) - L / 2, 0), np.maximum(np.abs(lat) - W / 2, 0)) - wu / 2
    return gap, lon, lat


def vru_cost(t_u, speed_u, du, t_in, t_out):
    """Whether the VRU waited (stood within 6 m of the conflict point while the vehicle passed, then
    went on) or slowed (lost 0.5 m/s, held over 1 s, in the 3 s before the vehicle cleared)."""
    w = (t_u >= t_in - 3.0) & (t_u <= t_out)
    after = (t_u > t_out) & (t_u <= t_out + 4.0)
    waited = bool(w.any() and ((speed_u < 0.3) & (du < 6.0) & w).any() and after.any() and speed_u[after].max() >= 0.7)
    before = (t_u >= t_in - 6.0) & (t_u < t_in - 3.0)
    slowed = False
    if before.any() and w.sum() >= 10:
        ref = float(np.median(speed_u[before]))
        held = np.convolve((speed_u[w] <= ref - 0.5).astype(float), np.ones(10), mode="valid")
        slowed = ref >= 0.7 and bool((held >= 10).any())
    return waited, slowed


def crossing_event(scene, v, u, info, profile, path, params, close_pet, tie_steps):
    """The row for a VRU that crossed the vehicle's swept path, or None."""
    iv = np.flatnonzero(scene.valid[v])
    iu = np.flatnonzero(scene.valid[u])
    L, W = scene.shape_at(v, int(iv[len(iv) // 2]))
    wu = float(np.clip(scene.width[u, iu].mean(), 0.4, 1.0))
    pv = scene.position[v, iv, :2].astype(float)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pv, axis=0), axis=1))])
    pu = scene.position[u, iu, :2].astype(float)
    proj = np.array([path.project(q, real=True) for q in pu])
    zone = proj[:, 1] <= W / 2 + wu / 2 + MARGIN
    if not zone.any():
        return None
    k_in = int(np.flatnonzero(zone)[0])  # the first time the VRU's body enters the path
    seg = k_in
    while seg + 1 < len(zone) and (zone[seg + 1] or (seg + 2 < len(zone) and zone[seg + 2])):
        seg += 1
    s_star = float(np.median(proj[k_in:seg + 1, 0]))
    if s_star <= L / 2 + MARGIN:
        return None  # the vehicle was already over the point when the clip began
    hv = path.heading_at(s_star)
    hu = motion_heading(pu, k_in)
    rel = None if hu is None else abs(_wrap(hu - hv))
    if rel is not None and not CROSSING[0] <= rel <= CROSSING[1]:
        return None  # moving along the path: a passing or following event, not a crossing
    point = path.point_at(s_star)
    t_u_in, t_u_out = float(iu[k_in]) * DT, float(iu[seg]) * DT
    t_v_in = first_time(iv, arc + L / 2, s_star - wu / 2 - MARGIN)
    t_v_out = first_time(iv, arc - L / 2, s_star + wu / 2 + MARGIN)
    A = R._approach(scene, v, path, s_star, params)
    turning = A.turn != "straight" and abs(_wrap(hv - A.heading)) >= math.radians(45.0)
    flags = []
    if t_v_in != t_v_in:
        went, gap = "vru", ""
    elif t_v_out == t_v_out and t_v_out <= t_u_in:
        went, gap = "vehicle", round(t_u_in - t_v_out, 1)
    elif t_u_out <= t_v_in:
        went, gap = "vru", round(t_v_in - t_u_out, 1)
    else:
        went, gap = "together", 0.0
    if t_v_in == t_v_in and t_v_out != t_v_out:
        flags.append("vehicle_never_cleared_in_clip")
    poly = info.crosswalk_at(point)
    where = info.where(point)
    if where == "intersection":
        flags.append("junction_from_lane_ends")
    t_ref = t_v_in if t_v_in == t_v_in else t_u_in  # the VRU's state when the vehicle reached the conflict
    ku = int(np.argmin(np.abs(iu * DT - t_ref)))
    state = vru_state(info, poly, pu[ku])
    if abs(iu[ku] * DT - t_ref) > 0.5:
        state = ""
        flags.append("vru_unseen_when_vehicle_arrived")
    # stepped out late: when the VRU first came onto the crosswalk / carriageway, the vehicle was
    # already within its stopping distance of the conflict
    late = False
    on = [k for k in range(len(iu)) if vru_state(info, poly, pu[k]) == "on"]
    if on and on[0] == 0:
        flags.append("vru_first_seen_in_road")
    if on and on[0] > 0 and went in ("vehicle", "together"):  # seen stepping on, not first seen there
        t_on = float(iu[on[0]]) * DT
        kv = int(np.argmin(np.abs(iv * DT - t_on)))
        sp = float(np.linalg.norm(scene.velocity[v, iv[kv]]))
        dist = s_star - (arc[kv] + L / 2)
        late = 0 < dist < sp * REACTION + sp ** 2 / (2 * BRAKE) and abs(iv[kv] * DT - t_on) <= 0.5
    cyc = "crossing"
    if scene.types[u] == "CYCLIST" and hu is not None:
        r = abs(_wrap(hu - A.heading))
        cyc = "same_direction" if r < math.radians(45) else "oncoming" if r > math.radians(135) else "crossing"
    duty, rule = jurisdiction.vru_duty(profile, scene.types[u], where, state, A.control, turning, cyc, late)
    kv_in = int(np.argmin(np.abs(iv * DT - (t_v_in if t_v_in == t_v_in else t_u_in))))
    v_speed = float(np.linalg.norm(scene.velocity[v, iv[kv_in]]))
    common = np.flatnonzero(scene.valid[v] & scene.valid[u])
    gaps, _, _ = box_gap(scene, v, u, common, L, W, wu)
    min_gap = float(gaps.min())
    speed_u = smooth(np.linalg.norm(scene.velocity[u, iu], axis=1))
    du = np.linalg.norm(pu - point, axis=1)
    waited = slowed = False
    if went in ("vehicle", "together") and t_v_out == t_v_out:
        waited, slowed = vru_cost(iu * DT, speed_u, du, t_v_in, t_v_out)
    row = {"event": "crossing", "where": where, "vru_state": state, "vehicle_control": A.control,
           "vehicle_signal": A.signal or "", "vehicle_turn": A.turn, "turning": int(turning),
           "cyclist_direction": cyc if scene.types[u] == "CYCLIST" else "", "duty": duty, "rule": rule,
           "went_first": went, "gap_s": gap, "min_gap_m": round(min_gap, 2), "vehicle_speed": round(v_speed, 1),
           "vru_waited": int(waited), "vru_slowed": int(slowed), "violation": 0, "affected": "", "affected_by": "",
           "aggressive_candidate": 0, "candidate_kind": "",
           "vehicle_red_entry": jurisdiction.red_entry(profile, A, stopped_before(scene, v, A.entered)),
           "t_vehicle": round(t_v_in, 1) if t_v_in == t_v_in else "",
           "conflict_x": round(float(point[0]), 1), "conflict_y": round(float(point[1]), 1)}
    if late:
        flags.append("stepped_out_late")
    if duty == "vehicle_rules":  # a cyclist crossing like a vehicle: the vehicle right-of-way screen
        r = screen_pair(R, lanes_mod, scene, v, u, profile, tie_steps)
        row.update(row_case=r["case"], row_holder={"sdc": "vehicle", "ooi": "cyclist"}.get(r["holder"], r["holder"]),
                   row_violation=r["sdc_violation"], affected=r["affected"], affected_by=r["affected_by"],
                   aggressive_candidate=r["sdc_aggressive_candidate"],
                   candidate_kind="right_of_way" if r["sdc_aggressive_candidate"] else "")
        row["violation"] = r["sdc_violation"]
    elif duty == "yield" and went in ("vehicle", "together"):
        why = [n for n, hit in (("vru_waited", waited), ("vru_slowed", slowed),
                                ("cut_close", gap != "" and gap <= close_pet and v_speed >= 2.0),
                                ("passed_within_1.5m", min_gap < CLOSE_M)) if hit]
        row.update(violation=1, affected=int(bool(why)), affected_by="+".join(why),
                   aggressive_candidate=int(bool(why)), candidate_kind="failed_to_yield" if why else "")
    elif duty == "yield":
        row["affected"] = 0
    elif state == "on" and not late and went in ("vehicle", "together") and v_speed >= 3.0 and (
            (gap != "" and gap < 1.0) or min_gap < 0.5):
        row.update(aggressive_candidate=1, candidate_kind="endangered", affected=1, affected_by="no_due_care")
    if A.control == R.UNKNOWN:
        flags.append("vehicle_light_unknown")
    if len(iu) < MIN_SEEN and row["aggressive_candidate"]:  # a VRU seen this briefly may be a false detection
        flags.append("short_vru_track")
        row["aggressive_candidate"], row["candidate_kind"] = 0, ""
    row["flags"] = ";".join(flags)
    if row["aggressive_candidate"] or row["violation"]:
        low = {"vru_unseen_when_vehicle_arrived", "vehicle_never_cleared_in_clip"} & set(flags)
        row["confidence"] = ("low" if low else "medium" if where == "intersection" or state == "waiting"
                             or rule == "cyclist_crossing" else "high")
    return row


def passing_event(scene, v, u, info, profile, in_town=False):
    """The row fields for the vehicle overtaking a VRU moving its way, or None."""
    common = np.flatnonzero(scene.valid[v] & scene.valid[u])
    if common.size < 10:
        return None
    L, W = scene.shape_at(v, int(common[len(common) // 2]))
    iu = np.flatnonzero(scene.valid[u])
    wu = float(np.clip(scene.width[u, iu].mean(), 0.4, 1.0))
    gap, lon, lat = box_gap(scene, v, u, common, L, W, wu)
    if not ((lon > L / 2).any() and (lon < -L / 2).any()):
        return None  # the VRU was not first ahead of the vehicle and then behind it
    first_ahead = int(np.flatnonzero(lon > L / 2)[0])
    behind = np.flatnonzero(lon < -L / 2)
    behind = behind[behind > first_ahead]
    if not behind.size:
        return None
    side = np.arange(first_ahead, int(behind[0]) + 1)
    side = side[np.abs(lon[side]) <= L / 2 + 0.5]  # alongside
    if not side.size or np.abs(lat[side]).min() > 5.0:
        return None
    hv = scene.heading[v, common[side]]
    pos_u = scene.position[u, iu, :2].astype(float)
    k = int(np.argmin(np.abs(iu - common[side[0]])))
    hu = motion_heading(pos_u, k)
    if hu is None or abs(_wrap(hu - float(hv[0]))) > ALONG:
        return None
    vel_v = scene.velocity[v, common[side]]
    vel_u = scene.velocity[u, common[side]]
    rel = float(np.median(np.linalg.norm(vel_v, axis=1) - np.linalg.norm(vel_u, axis=1)))
    j = side[int(np.argmin(gap[side]))]
    q = scene.position[v, common[j], :2].astype(float)
    kmh = info.speed_limit(q)
    required, statutory = jurisdiction.passing_distance(profile, scene.types[u], not in_town and kmh > URBAN_KMH)
    flags = [] if kmh > 0 or in_town else ["speed_limit_unknown_taken_as_urban"]
    if info.lane_dist(scene.position[u, common[j], :2].astype(float)) > CARRIAGEWAY:
        flags.append("vru_off_carriageway")  # on a sidewalk or verge: whether this is overtaking is for review
    if not statutory:
        flags.append("no_statutory_distance")
    g = float(gap[j])
    viol = g < required and rel >= MIN_REL and float(np.linalg.norm(scene.velocity[v, common[j]])) >= 3.0
    return {"event": "passing", "passing_gap_m": round(g, 2), "passing_required_m": required,
            "passing_statutory": int(statutory), "passing_rel_speed": round(rel, 1),
            "vehicle_speed": round(float(np.linalg.norm(scene.velocity[v, common[j]])), 1),
            "t_vehicle": round(float(common[j]) * DT, 1), "violation": int(viol), "affected": int(viol),
            "affected_by": "passed_too_close" if viol else "", "aggressive_candidate": int(viol),
            "candidate_kind": "close_pass" if viol else "", "min_gap_m": round(float(gap.min()), 2),
            "conflict_x": round(float(q[0]), 1), "conflict_y": round(float(q[1]), 1),
            "confidence": ("high" if statutory else "medium") if viol else "", "flags": ";".join(flags)}


def vehicles_of(scene, mode):
    if mode == "sdc":
        return [scene.sdc]
    if mode == "ooi":
        return [i for i in scene.objects_of_interest if scene.types[i] == "VEHICLE"]
    return [i for i in range(scene.n_agents) if scene.types[i] == "VEHICLE" and scene.valid[i].sum() >= 10]


def near_pairs(scene, mode):
    """(vehicle, VRU) pairs seen together for 0.5 s that come within NEAR metres."""
    vrus = [i for i in range(scene.n_agents) if scene.types[i] in VRU_TYPES]
    pairs = []
    for v in vehicles_of(scene, mode):
        for u in vrus:
            both = scene.valid[v] & scene.valid[u]
            if both.sum() >= 5 and np.linalg.norm(scene.position[v, both, :2] - scene.position[u, both, :2],
                                                  axis=1).min() <= NEAR:
                pairs.append((v, u))
    return pairs


def screen_scene(scene, profile, mode, close_pet, tie_steps, in_town=False):
    pairs = near_pairs(scene, mode)
    if not pairs:  # most scenes: no map work at all
        return []
    info = MapInfo(scene, lanes_mod.lane_graph(scene))
    params = R.RightOfWayParams()
    rows, paths = [], {}
    for v, u in pairs:
        if v not in paths:
            paths[v] = R._Path(scene, v, scene.n_steps - 1, params.extend)
        base = {"vehicle_id": scene.track_ids[v], "vehicle_is_sdc": int(v == scene.sdc),
                "vru_id": scene.track_ids[u], "vru_type": scene.types[u]}
        for event in (crossing_event(scene, v, u, info, profile, paths[v], params, close_pet, tie_steps),
                      passing_event(scene, v, u, info, profile, in_town)):
            if event is not None:
                rows.append({**base, **event})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="folder of scene pickles, or one file")
    ap.add_argument("--profile", default="us-california", help="profile id or path: %s" % ", ".join(jurisdiction.list_profiles()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--vehicles", choices=("sdc", "ooi", "all"), default="sdc",
                    help="which vehicles to screen: the SDC, the vehicles among the objects of interest, or all")
    ap.add_argument("--part", default=None,
                    help="K/N: only every N-th scene file starting at the K-th (0-based), to run N processes")
    ap.add_argument("--close-pet", type=float, default=1.0,
                    help="s: a VRU reaching the conflict this soon after the vehicle cleared it was cut close "
                         "(walkers are slow, so a longer gap is told by the distance, %.1f m)" % CLOSE_M)
    ap.add_argument("--tie-seconds", type=float, default=1.0, help="for cyclists judged by the vehicle rules")
    ap.add_argument("--in-town", action="store_true",
                    help="every scene is in town (passing distances): use it for WOMD, all US cities; without it "
                         "a lane speed limit above %.0f km/h counts as outside town" % URBAN_KMH)
    ap.add_argument("--infer-lights", action="store_true", help="fill in unknown light states (lights.py)")
    args = ap.parse_args()

    profile = jurisdiction.load_profile(args.profile)
    if "vru" not in profile:
        sys.exit("profile %s has no vru block (references/jurisdictions.md)" % profile["id"])
    if profile["status"].startswith("draft"):
        print("WARNING: profile %s is a draft: %s" % (profile["id"], profile["status"]), file=sys.stderr)
    tie_steps = int(round(args.tie_seconds / DT))
    files = scene_files(args.scenes)
    if args.part:
        k, n = (int(x) for x in args.part.split("/"))
        files = files[k::n]
    rows, skipped = [], 0
    for path in files:
        try:
            scene, fmt = load_scene(path, Scene)
            if args.infer_lights:
                scene, _ = infer_lights(scene)
            found = screen_scene(scene, profile, args.vehicles, args.close_pet, tie_steps, args.in_town)
        except NeedsTorch as exc:
            sys.exit(str(exc))
        except Exception as exc:  # one bad scene must not stop a batch
            print("skip %s: %s: %s" % (path.name, type(exc).__name__, exc), file=sys.stderr)
            skipped += 1
            continue
        for r in found:
            r.update(scene=path.stem, scenario_id=scene.scenario_id, format=fmt, jurisdiction=profile["id"])
        rows += found
    # the same recording cut at different times gives the same event at the same place
    groups = {}
    for r in rows:
        groups.setdefault((r["event"], r["vehicle_id"], r["vru_id"]), []).append(r)
    for r in rows:
        same = [x["scene"] for x in groups[(r["event"], r["vehicle_id"], r["vru_id"])] if x["scene"] != r["scene"]
                and math.hypot(x["conflict_x"] - r["conflict_x"], x["conflict_y"] - r["conflict_y"]) <= SAME_EVENT_M]
        if same:
            r["flags"] = (r["flags"] + ";" if r["flags"] else "") + "same_event_as=" + "/".join(same)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n = lambda kind: sum(r["candidate_kind"] == kind for r in rows)
    print("%d scenes (%d skipped); %d crossing and %d passing events; violations %d; candidates: failed to yield %d, "
          "endangered %d, close pass %d, right of way (cyclist) %d -> %s"
          % (len(files) - skipped, skipped, sum(r["event"] == "crossing" for r in rows),
             sum(r["event"] == "passing" for r in rows), sum(r["violation"] for r in rows), n("failed_to_yield"),
             n("endangered"), n("close_pass"), n("right_of_way"), args.out))


if __name__ == "__main__":
    main()
