"""Converts Waymo Open Motion Dataset scenario tfrecords into ScenarioNet / MetaDrive scene pickles,
the format of CAT's raw_scenes_500 that the screens and the judge read.

Pure Python plus NumPy: it reads the TFRecord framing directly and decodes the Scenario protobuf
from its wire format, with the field numbers of WOMD's scenario.proto and map.proto. No TensorFlow,
no waymo_open_dataset, no MetaDrive. The output keeps what MetaDrive's Waymo converter writes and
the screens use:
- tracks (position, heading, velocity, box, valid);
- the map features: lanes with entry and exit lanes, neighbours and boundaries; road lines; road
  edges; stop signs; crosswalks; speed bumps; driveways;
- the per-step light state of each controlled lane;
- the metadata.
It leaves out MetaDrive's lane widths and its object and number summaries.

``python convert_womd.py --check-against raw_scenes_500 --tfrecords cat_scenarios_v121.tfrecord``
compares the conversion field by field with scenes MetaDrive's converter made.

    python convert_womd.py --tfrecords E:/womd/validation_interactive_sdc --out E:/womd/scenes --workers 8
    python convert_womd.py ... --ids ids.txt      # only these scenario ids
"""

import argparse
import multiprocessing
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

TRACK_TYPES = {0: "UNSET", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}
LANE_TYPES = {0: "LANE_UNKNOWN", 1: "LANE_FREEWAY", 2: "LANE_SURFACE_STREET", 3: "LANE_BIKE_LANE"}
LINE_TYPES = {0: "UNKNOWN", 1: "ROAD_LINE_BROKEN_SINGLE_WHITE", 2: "ROAD_LINE_SOLID_SINGLE_WHITE",
              3: "ROAD_LINE_SOLID_DOUBLE_WHITE", 4: "ROAD_LINE_BROKEN_SINGLE_YELLOW", 5: "ROAD_LINE_BROKEN_DOUBLE_YELLOW",
              6: "ROAD_LINE_SOLID_SINGLE_YELLOW", 7: "ROAD_LINE_SOLID_DOUBLE_YELLOW", 8: "ROAD_LINE_PASSING_DOUBLE_YELLOW"}
EDGE_TYPES = {0: "ROAD_EDGE_UNKNOWN", 1: "ROAD_EDGE_BOUNDARY", 2: "ROAD_EDGE_MEDIAN"}
LIGHT_STATES = {0: "LANE_STATE_UNKNOWN", 1: "LANE_STATE_ARROW_STOP", 2: "LANE_STATE_ARROW_CAUTION",
                3: "LANE_STATE_ARROW_GO", 4: "LANE_STATE_STOP", 5: "LANE_STATE_CAUTION", 6: "LANE_STATE_GO",
                7: "LANE_STATE_FLASHING_STOP", 8: "LANE_STATE_FLASHING_CAUTION"}


# --- protobuf wire format ------------------------------------------------------------------------

def records(path):
    """The data of every record of a tfrecord file."""
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            (n,) = struct.unpack("<Q", head[:8])
            data = f.read(n)
            f.read(4)
            yield data


def _varint(b, i):
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        if c < 0x80:
            return x, i
        s += 7


def message(b):
    """{field number: [(wire type, raw value), ...]} of one message."""
    out, i = {}, 0
    while i < len(b):
        key, i = _varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError("unsupported wire type %d" % wt)
        out.setdefault(num, []).append((wt, v))
    return out


def _signed(x):
    return x - (1 << 64) if x >= 1 << 63 else x


def ints(m, num):
    """A repeated (packed or not) or single integer field."""
    out = []
    for wt, v in m.get(num, []):
        if wt == 2:
            j = 0
            while j < len(v):
                x, j = _varint(v, j)
                out.append(_signed(x))
        else:
            out.append(_signed(v))
    return out


def reals(m, num, double):
    fmt, size = ("<d", 8) if double else ("<f", 4)
    out = []
    for wt, v in m.get(num, []):
        if wt == 2:  # packed
            out += [x[0] for x in struct.iter_unpack(fmt, v)]
        else:
            out.append(struct.unpack(fmt, v)[0])
    return out


def one(values, default):
    return values[-1] if values else default


def subs(m, num):
    return [message(v) for _, v in m.get(num, [])]


def points(m, num):
    """Repeated MapPoint (x = 1, y = 2, z = 3) as float32 [N, 3]."""
    pts = [[one(reals(p, k, True), 0.0) for k in (1, 2, 3)] for p in subs(m, num)]
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


# --- Scenario -> scene description ----------------------------------------------------------------

def _track(t, n_steps):
    """Track: id = 1, object_type = 2, states = 3. ObjectState: center_x/y/z = 2/3/4 (double),
    length/width/height = 5/6/7, heading = 8, velocity_x/y = 9/10 (float), valid = 11."""
    states = subs(t, 3)
    arr = {k: np.zeros(n_steps, np.float32) for k in ("x", "y", "z", "length", "width", "height", "heading", "vx", "vy")}
    valid = np.zeros(n_steps, bool)
    for k, s in enumerate(states[:n_steps]):
        for name, num, double in (("x", 2, True), ("y", 3, True), ("z", 4, True), ("length", 5, False),
                                  ("width", 6, False), ("height", 7, False), ("heading", 8, False),
                                  ("vx", 9, False), ("vy", 10, False)):
            arr[name][k] = one(reals(s, num, double), 0.0)
        valid[k] = bool(one(ints(s, 11), 0))
    tid = str(one(ints(t, 1), -1))
    kind = TRACK_TYPES.get(one(ints(t, 2), 0), "UNSET")
    return tid, {"type": kind,
                 "state": {"position": np.stack([arr["x"], arr["y"], arr["z"]], 1), "length": arr["length"],
                           "width": arr["width"], "height": arr["height"], "heading": arr["heading"],
                           "velocity": np.stack([arr["vx"], arr["vy"]], 1), "valid": valid},
                 "metadata": {"track_length": n_steps, "type": kind, "object_id": tid, "dataset": "waymo"}}


def _boundaries(m, num):
    """BoundarySegment: lane_start_index = 1, lane_end_index = 2, boundary_feature_id = 3, boundary_type = 4."""
    return [{"lane_start_index": str(one(ints(b, 1), 0)), "lane_end_index": str(one(ints(b, 2), 0)),
             "boundary_type": LINE_TYPES.get(one(ints(b, 4), 0), "UNKNOWN"),
             "boundary_feature_id": str(one(ints(b, 3), 0))} for b in subs(m, num)]


def _neighbours(m, num):
    """LaneNeighbor: feature_id = 1, self_start/end_index = 2/3, neighbor_start/end_index = 4/5, boundaries = 6."""
    return [{"feature_id": str(one(ints(n, 1), 0)), "self_start_index": str(one(ints(n, 2), 0)),
             "self_end_index": str(one(ints(n, 3), 0)), "neighbor_start_index": str(one(ints(n, 4), 0)),
             "neighbor_end_index": str(one(ints(n, 5), 0)), "boundaries": _boundaries(n, 6)} for n in subs(m, num)]


def _feature(f):
    """MapFeature: id = 1; lane = 3, road_line = 4, road_edge = 5, stop_sign = 7, crosswalk = 8,
    speed_bump = 9, driveway = 10."""
    fid = str(one(ints(f, 1), -1))
    if 3 in f:  # LaneCenter: speed_limit_mph = 1, type = 2, interpolating = 3, polyline = 8, entry_lanes = 9,
        # exit_lanes = 10, left_neighbors = 11, right_neighbors = 12, left_boundaries = 13, right_boundaries = 14
        m = message(f[3][-1][1])
        mph = one(reals(m, 1, True), 0.0)
        return fid, {"speed_limit_mph": mph, "speed_limit_kmh": mph * 1.609344,
                     "type": LANE_TYPES.get(one(ints(m, 2), 0), "LANE_UNKNOWN"), "polyline": points(m, 8),
                     "interpolating": bool(one(ints(m, 3), 0)), "entry_lanes": ints(m, 9), "exit_lanes": ints(m, 10),
                     "left_boundaries": _boundaries(m, 13), "right_boundaries": _boundaries(m, 14),
                     "left_neighbor": _neighbours(m, 11), "right_neighbor": _neighbours(m, 12)}
    if 4 in f:  # RoadLine: type = 1, polyline = 2
        m = message(f[4][-1][1])
        return fid, {"type": LINE_TYPES.get(one(ints(m, 1), 0), "UNKNOWN").replace("UNKNOWN", "ROAD_LINE_UNKNOWN"),
                     "polyline": points(m, 2)}
    if 5 in f:  # RoadEdge: type = 1, polyline = 2
        m = message(f[5][-1][1])
        return fid, {"type": EDGE_TYPES.get(one(ints(m, 1), 0), "ROAD_EDGE_UNKNOWN"), "polyline": points(m, 2)}
    if 7 in f:  # StopSign: lane = 1, position = 2
        m = message(f[7][-1][1])
        pos = subs(m, 2)
        p = np.asarray([one(reals(pos[0], k, True), 0.0) for k in (1, 2, 3)] if pos else [0.0] * 3, np.float32)
        return fid, {"type": "STOP_SIGN", "lane": ints(m, 1), "position": p}
    for num, kind in ((8, "CROSSWALK"), (9, "SPEED_BUMP"), (10, "DRIVEWAY")):  # polygon = 1
        if num in f:
            return fid, {"type": kind, "polygon": points(message(f[num][-1][1]), 1)}
    return fid, None


def convert(data):
    """One Scenario record -> a ScenarioNet scene description.

    Scenario: timestamps_seconds = 1, tracks = 2, objects_of_interest = 4, scenario_id = 5,
    sdc_track_index = 6, dynamic_map_states = 7, map_features = 8, current_time_index = 10,
    tracks_to_predict = 11 (RequiredPrediction: track_index = 1, difficulty = 2).
    DynamicMapState: lane_states = 1 (TrafficSignalLaneState: lane = 1, state = 2, stop_point = 3)."""
    m = message(data)
    sid = one([v.decode() for _, v in m.get(5, [])], "")
    ts = reals(m, 1, True)
    n = len(ts)
    tracks = dict(_track(t, n) for t in subs(m, 2))
    ids = list(tracks)
    sdc_index = one(ints(m, 6), 0)
    features = {}
    for f in subs(m, 8):
        fid, feat = _feature(f)
        if feat is not None:
            features[fid] = feat
    lights = {}
    for k, step in enumerate(subs(m, 7)[:n]):
        for ls in subs(step, 1):
            lane = one(ints(ls, 1), -1)
            light = lights.setdefault(str(lane), {
                "type": "TRAFFIC_LIGHT", "state": {"object_state": [None] * n}, "lane": lane,
                "stop_point": None, "metadata": {"track_length": n, "type": "TRAFFIC_LIGHT", "object_id": str(lane),
                                                 "dataset": "waymo"}})
            light["state"]["object_state"][k] = LIGHT_STATES.get(one(ints(ls, 2), 0), "LANE_STATE_UNKNOWN")
            sp = subs(ls, 3)
            if sp and light["stop_point"] is None:
                light["stop_point"] = np.asarray([one(reals(sp[0], j, True), 0.0) for j in (1, 2, 3)], np.float32)
    for light in lights.values():
        if light["stop_point"] is None:
            light["stop_point"] = np.zeros(3, np.float32)
    predict = {}
    for r in subs(m, 11):
        i = one(ints(r, 1), -1)
        if 0 <= i < len(ids):
            predict[ids[i]] = {"track_index": i, "track_id": ids[i], "difficulty": one(ints(r, 2), 0),
                               "object_type": tracks[ids[i]]["type"]}
    meta = {"coordinate": "waymo", "ts": np.asarray(ts, np.float32) - np.float32(ts[0] if ts else 0.0),
            "metadrive_processed": False, "sdc_id": ids[sdc_index] if 0 <= sdc_index < len(ids) else "",
            "dataset": "waymo", "scenario_id": sid, "source_file": "", "track_length": n,
            "current_time_index": one(ints(m, 10), 0), "sdc_track_index": sdc_index,
            "objects_of_interest": [str(x) for x in ints(m, 4)], "tracks_to_predict": predict}
    return {"id": sid, "version": "convert_womd.py (MetaDrive v0.3.0.1 layout)", "length": n, "tracks": tracks,
            "dynamic_map_states": lights, "map_features": features, "metadata": meta}


# --- running and checking -------------------------------------------------------------------------

def _shard(job):
    path, out, wanted = job
    done = 0
    for data in records(path):
        sd = convert(data)
        if wanted is not None and sd["id"] not in wanted:
            continue
        sd["metadata"]["source_file"] = Path(path).name
        tmp = Path(out) / (sd["id"] + ".pkl.part")
        with open(tmp, "wb") as f:
            pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(Path(out) / (sd["id"] + ".pkl"))
        done += 1
    return Path(path).name, done


def check(tfrecords, reference_dir):
    """Field-by-field comparison with scenes MetaDrive's converter made (matched by scenario id)."""
    ref = {}
    for p in Path(reference_dir).glob("*.pkl"):
        with open(p, "rb") as f:
            sd = pickle.load(f)
        ref[sd["metadata"]["scenario_id"]] = (p.name, sd)
    from collections import Counter
    c, notes = Counter(), []
    for path in tfrecords:
        for data in records(path):
            ours = convert(data)
            if ours["id"] not in ref:
                continue
            name, theirs = ref[ours["id"]]
            c["scenes"] += 1
            bad = compare(ours, theirs)
            for k in bad:
                c[k] += 1
            if bad and len(notes) < 10:
                notes.append("%s %s: %s" % (name, ours["id"], ", ".join(sorted(bad)[:6])))
    return c, notes


def compare(ours, theirs):
    bad = set()
    if ours["metadata"]["sdc_id"] != theirs["metadata"]["sdc_id"]:
        bad.add("sdc_id")
    if [str(x) for x in ours["metadata"]["objects_of_interest"]] != [str(x) for x in theirs["metadata"]["objects_of_interest"]]:
        bad.add("objects_of_interest")
    if set(ours["tracks"]) != set(theirs["tracks"]):
        bad.add("track ids")
    for tid in set(ours["tracks"]) & set(theirs["tracks"]):
        a, b = ours["tracks"][tid], theirs["tracks"][tid]
        if a["type"] != b["type"]:
            bad.add("track type")
        if not np.array_equal(a["state"]["valid"], b["state"]["valid"]):
            bad.add("valid")
        v = b["state"]["valid"]
        for k in ("position", "heading", "velocity", "length", "width", "height"):
            if not np.allclose(a["state"][k][v], b["state"][k][v], atol=1e-3):
                bad.add("track " + k)
    fa, fb = ours["map_features"], theirs["map_features"]
    if set(fa) != set(fb):
        bad.add("feature ids (%+d)" % (len(fa) - len(fb)))
    for fid in set(fa) & set(fb):
        a, b = fa[fid], fb[fid]
        if a["type"] != b["type"]:
            bad.add("feature type %s/%s" % (a["type"], b["type"]))
            continue
        for k in ("polyline", "polygon", "position"):
            if k in b and (k not in a or a[k].shape != np.asarray(b[k]).shape
                           or not np.allclose(a[k], np.asarray(b[k]), atol=1e-2)):
                bad.add(b["type"].split("_")[0] + " " + k)
        for k in ("entry_lanes", "exit_lanes", "lane"):
            if k in b and [int(x) for x in a.get(k, [])] != [int(x) for x in b[k]]:
                bad.add(k)
        for k in ("left_neighbor", "right_neighbor", "left_boundaries", "right_boundaries"):
            if k in b and a.get(k) != b[k]:
                bad.add(k)
        for k in ("speed_limit_mph", "interpolating"):
            if k in b and a.get(k) != b[k]:
                bad.add(k)
    la, lb = ours["dynamic_map_states"], theirs["dynamic_map_states"]
    if set(la) != set(lb):
        bad.add("light lanes")
    for k in set(la) & set(lb):
        if la[k]["state"]["object_state"] != lb[k]["state"]["object_state"]:
            bad.add("light states")
        if not np.allclose(la[k]["stop_point"], lb[k]["stop_point"], atol=1e-2):
            bad.add("stop_point")
    return bad


def tfrecord_files(items):
    out = []
    for item in items:
        p = Path(item)
        out += sorted(x for x in p.iterdir() if x.is_file() and ".tfrecord" in x.name) if p.is_dir() else [p]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfrecords", nargs="+", required=True, help="tfrecord files or folders of them")
    ap.add_argument("--out", help="folder for <scenario_id>.pkl")
    ap.add_argument("--ids", help="only these scenario ids (one per line)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--check-against", help="folder of MetaDrive-converted scenes to compare with (no output)")
    args = ap.parse_args(argv)
    files = tfrecord_files(args.tfrecords)
    if args.check_against:
        c, notes = check(files, args.check_against)
        for k, v in sorted(c.items()):
            print("%s: %d" % (k, v))
        print("\n".join(notes))
        return 0
    if not args.out:
        sys.exit("--out is required")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    wanted = None
    if args.ids:
        wanted = frozenset(l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#"))
    jobs = [(str(f), args.out, wanted) for f in files]
    results = map(_shard, jobs) if args.workers <= 1 else multiprocessing.Pool(args.workers).imap_unordered(_shard, jobs)
    total = 0
    for k, (name, n) in enumerate(results, 1):
        total += n
        print("[%d/%d] %s: %d scenes (total %d)" % (k, len(jobs), name, n, total), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
