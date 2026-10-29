"""The duty x cost x attribution judge: an agent is aggressive toward another when it owed it a duty,
caused the conflict and asked of it more braking than logged drivers in the same situation do.
references/judge.md has the method and what it was measured to do; responsibility/aggression_judge.py
the decision. No motion model, only NumPy.

Three commands:

  calibrate  the natural references and thresholds of a scene folder (every logged join, following
             episode and crossing whose yielder went first; the --quantile, 0.9 by default, of
             each). Writes OUT/calibration.json and OUT/references.csv. Calibrate on the dataset you
             judge, or on a larger one from the same place: thresholds are not portable.
  run        judges each scene's pair both ways (scenes.pick_pair: the SDC and its partner by
             default) against a calibration. Writes one row per scene and direction.
  self-test  synthetic checks of attribution, cost and verdict (no scene files needed).

The crossing duty is right_of_way.py's table: California's (--profile us-california) or its mirror
image for left-hand traffic (--profile left-hand-draft). Other profiles are not supported here; use
screen_crossing.py for them. Cut-ins and tailgating need no profile.

    python judge.py calibrate --scenes raw_scenes_500 --out cal --workers 4
    python judge.py run --scenes raw_scenes_500 --calibration cal/calibration.json --out judge.csv
    python judge.py self-test
"""

import argparse
import csv
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from responsibility import attribution as attr  # noqa: E402
from responsibility.aggression_judge import KINDS, candidate_pairs, judge, same_direction_references  # noqa: E402
from responsibility.forced_yield import forced_yield, track_conflict  # noqa: E402
from responsibility.right_of_way import RightOfWayParams  # noqa: E402
from responsibility.scene import Scene  # noqa: E402
from scenes import PAIR_MODES, NeedsTorch, load_scene, pick_pair  # noqa: E402

BIG = 1e6  # stands for an infinite demand in quantiles
SIDES = {"us-california": "right", "left-hand-draft": "left"}  # jurisdiction profile -> right_of_way.py's table
FIELDS = (["scene", "pair", "direction", "agent", "other", "verdict", "kind", "reason"]
          + [f"{k}_{x}" for k in KINDS for x in ("demand", "quantile")]
          + ["crossing_case", "crossing_actor", "cut-in_headway_at_join", "cut-in_speed_difference", "cut-in_join",
             "tailgate_headway", "error"])


def scene_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.pkl"), key=lambda p: (int(p.stem) if p.stem.isdigit() else float("inf"), p.stem))


def params_for(profile: str) -> RightOfWayParams:
    if profile not in SIDES:
        sys.exit(f"judge.py supports the profiles {', '.join(SIDES)}; use screen_crossing.py for {profile}")
    return RightOfWayParams(traffic_side=SIDES[profile])


# --- calibrate ---------------------------------------------------------------------------------

def crossing_reference(scene: Scene, pairs, params: RightOfWayParams):
    rows = []
    for i, j in pairs:
        if track_conflict(scene, i, j) is None:
            continue
        for x, y in ((i, j), (j, i)):
            cross = attr.crossing(scene, x, y, params)
            if cross is None or cross.actor != "agent":
                continue
            demand = forced_yield(scene, x, y, params=params).forced
            if demand is not None:
                rows.append({"agent": scene.track_ids[x], "other": scene.track_ids[y], "kind": "crossing",
                             "demand": demand, "step": cross.step, "is_sdc": int(x == scene.sdc or y == scene.sdc)})
    return rows


def scan(job):
    path, crossing_pairs, profile = job
    params = params_for(profile)
    try:
        scene, _ = load_scene(path, Scene)
        rows = same_direction_references(scene, params)
        if crossing_pairs == "marked":
            pair = pick_pair(scene, "ooi")
            pairs = [pair[:2]] if pair else []
        else:
            pairs = candidate_pairs(scene)
        rows += crossing_reference(scene, pairs, params)
    except NeedsTorch:
        raise
    except Exception as e:  # one bad scene must not stop a calibration
        print(f"skip {Path(path).name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return []
    for r in rows:
        r["scene"] = Path(path).stem
    return rows


def summarise(rows, q: float):
    out = {}
    for kind in KINDS:
        v = np.array([min(float(r["demand"]), BIG) for r in rows if r["kind"] == kind], dtype=float)
        if not v.size:
            continue
        t = float(np.quantile(v, q, method="higher"))  # an observed value, never between a finite one and inf
        out[kind] = {"quantile": q, "threshold": float("inf") if t >= BIG else t, "n": int(v.size),
                     "n_sdc": int(sum(r["is_sdc"] for r in rows if r["kind"] == kind)),
                     "percentiles": {f"p{p}": round(float(np.quantile(v, p / 100, method="higher")), 3)
                                     for p in (10, 25, 50, 75, 90, 95)},
                     "reference": sorted(round(float(x), 4) for x in v)}
    return out


def calibrate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = scene_files(args.scenes)[: args.n]
    jobs = [(str(f), args.crossing_pairs, args.profile) for f in files]
    results = (map(scan, jobs) if args.workers <= 1
               else multiprocessing.Pool(args.workers).imap_unordered(scan, jobs, chunksize=4))
    rows = [r for rs in results for r in rs]
    with open(out / "references.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "agent", "other", "kind", "demand", "step", "is_sdc"])
        w.writeheader()
        w.writerows(rows)
    cal = {"scenes": str(args.scenes), "n_scenes": len(files), "crossing_pairs": args.crossing_pairs,
           "profile": args.profile, **summarise(rows, args.quantile)}
    (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    for kind in KINDS:
        if kind in cal:
            c = cal[kind]
            print(f"{kind}: n {c['n']}, q{args.quantile:g} = {c['threshold']:.2f} m/s^2, percentiles {c['percentiles']}")
        else:
            print(f"{kind}: no references; this kind cannot be judged with this calibration")
    return cal


# --- run ---------------------------------------------------------------------------------------

def judge_scene(job):
    path, cal, mode, profile = job
    params = params_for(profile)
    blank = {k: "" for k in FIELDS}
    try:
        scene, _ = load_scene(path, Scene)
        pair = pick_pair(scene, mode)
        if pair is None:
            return [{**blank, "scene": Path(path).stem, "error": "no pair"}]
        a, b, how = pair
        rows = []
        for direction, x, y in (("first", a, b), ("second", b, a)):
            row = {**blank, "scene": Path(path).stem, "pair": how, "direction": direction,
                   "agent": scene.track_ids[x], "other": scene.track_ids[y]}
            row.update({k: v for k, v in judge(scene, x, y, cal, params).as_row().items() if k in FIELDS})
            rows.append(row)
        return rows
    except NeedsTorch:
        raise
    except Exception as e:  # one bad scene must not stop a run
        return [{**blank, "scene": Path(path).stem, "error": f"{type(e).__name__}: {e}"}]


def run(args):
    cal = json.loads(Path(args.calibration).read_text())
    if cal.get("profile", args.profile) != args.profile:
        print(f"warning: calibrated with --profile {cal['profile']}, judging with {args.profile}", file=sys.stderr)
    jobs = [(str(f), cal, args.pairs, args.profile) for f in scene_files(args.scenes)[: args.n]]
    rows = [r for rs in (map(judge_scene, jobs) if args.workers <= 1
                         else multiprocessing.Pool(args.workers).imap(judge_scene, jobs, chunksize=4)) for r in rs]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("thresholds: " + ", ".join(f"{k} {cal[k]['threshold']:.2f} (q{cal[k]['quantile']:g}, n {cal[k]['n']})"
                                     for k in KINDS if k in cal))
    for direction in ("first", "second"):
        count = {}
        for r in rows:
            if r["direction"] == direction:
                key = r["verdict"] + (f" ({r['kind']})" if r["verdict"] == "aggressive" else "")
                count[key] = count.get(key, 0) + 1
        print(f"{direction} toward {'second' if direction == 'first' else 'first'}: {dict(sorted(count.items()))}")
    errors = [r for r in rows if r["error"]]
    print(f"{len(rows)} judgements, {len(errors)} errors -> {args.out}")
    return rows


# --- self-test ---------------------------------------------------------------------------------

def _self_test():
    from responsibility.forced_merge import following_demand, same_direction, tailgate

    n, dt, speed = 91, 0.1, 15.0

    def track(x0, y0, vx=speed):
        t = (np.arange(n) - 10) * dt
        pos = np.zeros((n, 3), np.float32)
        pos[:, 0], pos[:, 1] = x0 + vx * t, y0
        return {"type": "VEHICLE", "metadata": {},
                "state": {"position": pos, "heading": np.zeros(n, np.float32),
                          "velocity": np.tile(np.array([vx, 0.0], np.float32), (n, 1)),
                          "length": np.full(n, 4.8, np.float32), "width": np.full(n, 2.0, np.float32),
                          "height": np.full(n, 1.5, np.float32), "valid": np.ones(n, bool)}}

    def lane_change(x0, y0, y1, t0, t1, vx=speed):
        tr = track(x0, y0, vx)
        t = (np.arange(n) - 10) * dt
        w = np.clip((t - t0) / (t1 - t0), 0, 1)
        y = y0 + (y1 - y0) * (3 * w ** 2 - 2 * w ** 3)
        vy = np.gradient(y, dt)
        tr["state"]["position"][:, 1] = y
        tr["state"]["velocity"][:, 1] = vy
        tr["state"]["heading"][:] = np.arctan2(vy, vx)
        return tr

    def road(*ys):
        xs = np.arange(-50.0, 260.0, 10.0)
        line = lambda y: np.stack([xs, np.full_like(xs, y), 0 * xs], 1)
        feats = {f"l{k}": {"type": "LANE_SURFACE_STREET", "polyline": line(y)} for k, y in enumerate(ys)}
        feats.update(e0={"type": "ROAD_EDGE_BOUNDARY", "polyline": line(min(ys) - 1.8)},
                     e1={"type": "ROAD_EDGE_BOUNDARY", "polyline": line(max(ys) + 1.8)})
        return feats

    def scene(tracks, feats, flip=False):
        sd = {"id": "t", "tracks": tracks, "map_features": feats, "dynamic_map_states": {},
              "metadata": {"sdc_id": "0", "objects_of_interest": ["0", "1"], "scenario_id": "t"}}
        if flip:  # mirror left to right: y and headings negate
            import copy
            sd = copy.deepcopy(sd)
            for tr in sd["tracks"].values():
                for key in ("position", "velocity"):
                    tr["state"][key][:, 1] *= -1
                tr["state"]["heading"] = -tr["state"]["heading"]
            for f in sd["map_features"].values():
                f["polyline"][:, 1] *= -1
        return Scene.from_description(sd)

    # who moved: the car that cut in, not the one it cut in on
    s = scene({"0": lane_change(15.0, 3.5, 0.0, 2.0, 4.0), "1": track(0.0, 0.0)}, road(0.0, 3.5))
    a = attr.attribute(s, 0, 1)
    assert a.kind == "join" and a.actor == "agent", a
    s = scene({"0": track(15.0, 0.0), "1": lane_change(0.0, 3.5, 0.0, 2.0, 4.0)}, road(0.0, 3.5))
    assert attr.attribute(s, 0, 1).actor == "other"
    # following is the follower's doing
    s = scene({"0": track(0.0, 0.0), "1": track(20.0, 0.0)}, road(0.0, 3.5))
    assert attr.attribute(s, 0, 1).kind == "following" and attr.attribute(s, 0, 1).actor == "agent"
    # one response time of headway at equal speed asks exactly the front car's braking (4 m/s^2)
    s = scene({"0": track(0.0, 0.0), "1": track(speed + 4.8, 0.0)}, road(0.0))
    demand, _, headway = following_demand(s, 0, 1, np.array([20]))
    assert abs(headway[0] - 1.0) < 1e-3 and abs(demand[0] - 4.0) < 4e-3, (headway, demand)
    # tighter cut-ins demand more
    d = [same_direction(scene({"0": lane_change(15.0, 3.5, 0.0, 2.0, 4.0), "1": track(x, 0.0)}, road(0.0, 3.5)),
                        0, 1)["cut-in"].demand for x in (0.0, -10.0)]
    assert d[0] > d[1] > 0, d
    # the short gap right after being cut in on is excused
    s = scene({"0": track(0.0, 0.0), "1": lane_change(10.8, 3.5, 0.0, 2.0, 3.5)}, road(0.0, 3.5))
    k = attr.joins(s, 1, 0)[0]
    excused = tailgate(s, 0, 1, excused_from=[k])
    assert excused.demand is None or excused.step >= k + 30
    # the verdict, and the same verdict on the mirrored road under the left-hand table
    cal = {"cut-in": {"threshold": 5.0, "quantile": 0.9, "reference": [1.0, 2.0, 3.0, 5.0]},
           "tailgate": {"threshold": 4.5, "quantile": 0.9, "reference": [1.0, 2.0, 4.5]},
           "crossing": {"threshold": 2.5, "quantile": 0.9, "reference": [0.5, 2.5]}}
    tracks = {"0": lane_change(8.0, 3.5, 0.0, 2.0, 4.0, vx=speed - 1.0), "1": track(0.0, 0.0)}
    for flip, p in ((False, params_for("us-california")), (True, params_for("left-hand-draft"))):
        s = scene(tracks, road(0.0, 3.5), flip)
        j = judge(s, 0, 1, cal, p)
        assert (j.verdict, j.kind) == ("aggressive", "cut-in"), j
        assert judge(s, 1, 0, cal, p).verdict == "not aggressive"
    calm = scene({"0": track(0.0, 0.0), "1": track(40.0, 0.0)}, road(0.0, 3.5))
    assert judge(calm, 0, 1, cal).verdict == "not aggressive"
    print("judge self-test passed")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate", help="natural references and thresholds of a scene folder")
    c.add_argument("--scenes", required=True)
    c.add_argument("--out", required=True, help="output folder")
    c.add_argument("--quantile", type=float, default=0.9)
    c.add_argument("--crossing-pairs", choices=["marked", "all"], default="all",
                   help="crossing references from the marked pair only, or every pair of moving vehicles")
    r = sub.add_parser("run", help="judge each scene's pair both ways")
    r.add_argument("--scenes", required=True)
    r.add_argument("--calibration", required=True)
    r.add_argument("--out", required=True, help="output CSV")
    r.add_argument("--pairs", choices=PAIR_MODES, default="auto")
    for p in (c, r):
        p.add_argument("--profile", default="us-california", help=", ".join(SIDES))
        p.add_argument("--n", type=int, default=None, help="only the first n scenes")
        p.add_argument("--workers", type=int, default=1)
    sub.add_parser("self-test", help="synthetic checks, no scene files")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "self-test":
        return _self_test()
    params_for(args.profile)
    try:
        calibrate(args) if args.command == "calibrate" else run(args)
    except NeedsTorch as e:
        sys.exit(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
