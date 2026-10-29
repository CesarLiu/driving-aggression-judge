"""Screen WOMD-style scenes for same-direction aggression: cut-ins, tailgating and close passes.

For every scene it takes the self-driving car (SDC) and the other object of interest (OOI), or a
pair you name, and judges each toward the other ("sdc_*" columns: the SDC toward the OOI). No right
of way is involved here, so the criteria are physical (references/rubric.md, family B):

* cut-in: X moved sideways into Y's lane ahead of Y (X did the moving, measured against the lane
  direction from the map), and that made Y brake harder than ``--cut-in-brake`` (2 m/s², held 1 s)
  or left it a time headway below ``--cut-in-floor`` (0.5 s), or below ``--cut-in-thw`` (0.8 s)
  while X was not pulling away (less than ``--pull-away``, 2 m/s, faster). A lane change that left Y a comfortable gap is not aggressive,
  nor is one into a slow queue (Y below 2 m/s).
* tailgating: X followed Y in Y's path (the boxes overlap sideways) at 5 m/s or more with a time headway below ``--tailgate-thw``
  (0.9 s) for ``--tailgate-run`` (1 s) or longer, or closed in with a time to collision below
  ``--tailgate-ttc`` (2 s). The 3 s after Y moved in ahead of X (a lane change, or any following
  that starts with Y already close ahead while X itself did not move in) do not count: that short
  gap was Y's doing.
* close pass: X, in the next lane, came within ``--close-gap`` (0.5 m, box to box) of Y at a
  relative speed above ``--close-speed`` (3 m/s), and X was the one that closed in sideways.

Positions are measured along and across the map lane Y is in at that moment (its own track where no
lane matches), so curved roads work and Y's later lane changes do not leak in; who moved sideways is
measured against the nearest map lane. Scene files: ScenarioNet pickles or catk's cached WOMD (scenes.py). The output is candidates with the
numbers behind them; the reviewer still looks at each (SKILL.md, step 5).

    python screen_same_direction.py --scenes raw_scenes_500 --out same_direction.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenes import PAIR_MODES, NeedsTorch, load_scene, pick_pair  # noqa: E402
from screen_crossing import scene_files, smooth, sustained_decel  # noqa: E402
from responsibility import lanes, rss  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

DT = 0.1
SAME_DIRECTION = math.radians(30.0)  # headings closer than this travel the same way
IN_LANE = 1.3  # m: X's centre within this of Y's track is in Y's lane
ADJACENT = (2.3, 5.0)  # m: X's centre this far to the side of Y's track is in the next lane
LOOK_BACK = 50  # steps (5 s): how far back a cut-in's start (X still in the next lane) may lie
AFTER = 30  # steps (3 s): window after a cut-in in which its effect on Y is measured


class Track:
    """An agent's own path as a polyline, extended 30 m along its first and last headings, with
    arc length 0 at its first real point. ``project`` gives (along, signed lateral; left > 0)."""

    def __init__(self, xy, extend=30.0):
        keep = [0]
        for k in range(1, len(xy)):
            if np.linalg.norm(xy[k] - xy[keep[-1]]) >= 0.2:
                keep.append(k)
        pts = xy[keep]
        if len(pts) < 2:
            self.ok = False
            return
        head = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
        tail = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
        pts = np.vstack([pts[0] - extend * head, pts, pts[-1] + extend * tail])
        seg = np.diff(pts, axis=0)
        self.start, self.vec = pts[:-1], seg
        self.len2 = np.maximum((seg ** 2).sum(1), 1e-9)
        self.s0 = np.concatenate([[0.0], np.cumsum(np.sqrt(self.len2))])[:-1] - extend
        self.ok = True

    def project(self, p):
        rel = p - self.start
        u = np.clip((rel * self.vec).sum(1) / self.len2, 0.0, 1.0)
        foot = self.start + u[:, None] * self.vec
        dist = np.linalg.norm(p - foot, axis=1)
        k = int(np.argmin(dist))
        cross = self.vec[k, 0] * (p[1] - foot[k, 1]) - self.vec[k, 1] * (p[0] - foot[k, 0])
        return self.s0[k] + u[k] * math.sqrt(self.len2[k]), math.copysign(dist[k], cross)


def corners(c, h, length, width):
    f, l = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
    return np.array([c + f * length / 2 + l * width / 2, c + f * length / 2 - l * width / 2,
                     c - f * length / 2 - l * width / 2, c - f * length / 2 + l * width / 2])


def box_gap(a, b):
    """Exact distance between two rectangles given as 4 corners (0 if they overlap)."""
    for poly in (a, b):  # separating axis test
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                break
        else:
            continue
        break
    else:
        return 0.0

    def point_seg(p, s0, s1):
        e = s1 - s0
        t = np.clip(np.dot(p - s0, e) / max(np.dot(e, e), 1e-9), 0, 1)
        return np.linalg.norm(p - (s0 + t * e))

    return min(point_seg(p, q[k], q[(k + 1) % 4]) for p_set, q in ((a, b), (b, a)) for p in p_set for k in range(4))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def lane_chain(graph, lane, order):
    """Centreline of ``lane`` with two lanes before and after it (the ones the agent drives if it
    does, else the straightest), extended 30 m: the agent's lane as a reference path."""
    pts = [graph.points[lane]]
    cur = lane
    for _ in range(2):  # forward
        cands = graph.exits.get(cur, [])
        if not cands:
            break
        mine = [c for c in cands if c in order and order.index(c) > order.index(cur)] if cur in order else []
        cur = mine[0] if mine else min(cands, key=lambda c: abs(_wrap(graph.direction[c][0] - graph.direction[cur][-1])))
        pts.append(graph.points[cur])
    cur = lane
    for _ in range(2):  # backward
        cands = [k for k, ex in graph.exits.items() if cur in ex]
        if not cands:
            break
        mine = [c for c in cands if c in order and order.index(c) < order.index(cur)] if cur in order else []
        cur = mine[0] if mine else min(cands, key=lambda c: abs(_wrap(graph.direction[c][-1] - graph.direction[cur][0])))
        pts.insert(0, graph.points[cur])
    return Track(np.concatenate(pts))


def reference_paths(scene, i, graph):
    """Per step, the path that stands for agent i's lane: the map lane it is on (with what comes
    before and after), or its own track where no lane matches. Measuring the other car against the
    lane i is in *now* keeps i's own later lane changes out of it."""
    idx = np.flatnonzero(scene.valid[i])
    own = Track(scene.position[i, idx, :2].astype(float))
    own.lane = False
    lanes = {t: graph.best(scene.position[i, t, :2].astype(float), float(scene.heading[i, t])) if len(graph) else None
             for t in idx}
    order = []
    for t in idx:
        if lanes[t] is not None and lanes[t] not in order:
            order.append(lanes[t])
    cache, ref = {}, {}
    for t in idx:
        lane = lanes[t]
        if lane is None:
            ref[t] = own
            continue
        if lane not in cache:
            chain = lane_chain(graph, lane, order)
            chain.lane = True
            cache[lane] = chain if chain.ok else own
        ref[t] = cache[lane]
    return ref


def sideways(scene, i, data, lo, hi):
    """How far agent i moved across the road between steps lo and hi (m): the change of its offset
    from its lane. Integrating its speed across the lane instead would pile up the small heading
    differences on a curve into metres. Falls back to that integral off the map."""
    ref = data["ref"][i]
    valid = [t for t in range(lo, hi + 1) if scene.valid[i, t]]
    if len(valid) < 2:
        return 0.0
    a, b = valid[0], valid[-1]
    pa, pb = scene.position[i, a, :2].astype(float), scene.position[i, b, :2].astype(float)
    # against the lane it started in and the lane it ended in: a real lane change moves it across
    # both; lane matching that jumps between overlapping or branching lanes moves it across only one
    moves = [abs(r.project(pb)[1] - r.project(pa)[1]) for r in (ref.get(a), ref.get(b))
             if r is not None and getattr(r, "lane", False)]
    if moves:
        return min(moves)
    return abs(np.nansum(data["lat"][i][a:b + 1]) * DT)


def lateral_speed(scene, i, segments, rss):
    """Speed across the lane (m/s, left > 0) at every step, against the nearest map lane running the
    agent's way (its own heading where the map has none)."""
    pos = scene.position[i, :, :2].astype(float)
    lane_dir = rss.lane_frame(segments, pos, scene.heading[i].astype(float))
    v = scene.velocity[i].astype(float)
    return -v[:, 0] * np.sin(lane_dir) + v[:, 1] * np.cos(lane_dir)


def directed(scene, x, y, data, p):
    """X toward Y: cut-in, tailgating, close pass (dict of numbers and 0/1 candidates)."""
    both = scene.valid[x] & scene.valid[y]
    steps = np.flatnonzero(both)
    out = {"cut_in": 0, "tailgate": 0, "close_pass": 0}
    ref = data["ref"][y]
    steps = np.array([t for t in steps if ref.get(t) is not None and ref[t].ok], dtype=int)
    if len(steps) < 10:
        return out, []
    T = scene.n_steps
    s_x, d_x, s_y = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    for t in steps:  # X against the lane Y is in at that moment
        s_x[t], d_x[t] = ref[t].project(scene.position[x, t, :2].astype(float))
        s_y[t] = ref[t].project(scene.position[y, t, :2].astype(float))[0]
    half = (scene.length[x] + scene.length[y]) / 2
    v_x, v_y = data["speed"][x], data["speed"][y]
    same = np.abs((scene.heading[x] - scene.heading[y] + np.pi) % (2 * np.pi) - np.pi) < SAME_DIRECTION
    same &= both
    in_lane = same & (np.abs(d_x) <= IN_LANE)
    adjacent = same & (np.abs(d_x) >= ADJACENT[0]) & (np.abs(d_x) <= ADJACENT[1])
    ahead_gap = s_x - s_y - half  # X's rear to Y's front, X ahead
    behind_gap = s_y - s_x - half  # Y's rear to X's front, X behind
    lat_x, lat_y = data["lat"][x], data["lat"][y]
    dec_y = sustained_decel(v_y)  # braking held over 1 s, not frame-to-frame speed noise
    notes = []

    # cut-in: X enters Y's lane ahead of Y, having been in the next lane in the 5 s before
    best = None
    for t in steps[1:]:
        if not (in_lane[t] and not in_lane[t - 1] and ahead_gap[t] > -0.5):
            continue
        start = [k for k in range(max(0, t - LOOK_BACK), t) if adjacent[k]]
        if not start:
            continue
        t0 = start[-1]
        lo, hi = max(0, t0 - 20), min(T, t + 21)  # the whole manoeuvre, not just the line crossing
        move_x = sideways(scene, x, data, lo, hi - 1)
        move_y = sideways(scene, y, data, lo, hi - 1)
        if move_x < 1.5 or move_x < 1.5 * move_y:  # Y did the moving (or nobody clearly did)
            notes.append("lane_join_at_%.1fs_not_by_this_agent" % (t * DT))
            continue
        w = np.arange(t, min(T, t + AFTER + 1))
        w = w[in_lane[w] & (ahead_gap[w] > -0.5)]
        if v_y[t] < 2.0 or not len(w):  # into a slow queue, or straight out again
            continue
        thw = np.min(np.maximum(ahead_gap[w], 0) / np.maximum(v_y[w], 0.1))
        brake = float(np.nanmax(dec_y[max(0, t - 5):min(T, t + AFTER + 1)]))
        ev = dict(t=t * DT, start=t0 * DT, gap=float(np.nanmin(ahead_gap[w])), thw=float(thw), brake=brake,
                  move=move_x, v_y=float(v_y[t]), faster=float(v_x[t] - v_y[t]))
        if best is None or ev["thw"] < best["thw"]:
            best = ev
    if best:
        out.update(cut_in_t=round(best["t"], 1), cut_in_lateral_move=round(best["move"], 1),
                   cut_in_min_gap=round(best["gap"], 1), cut_in_min_thw=round(best["thw"], 2),
                   cut_in_other_brake=round(best["brake"], 1), cut_in_other_speed=round(best["v_y"], 1),
                   cut_in_start=round(best["start"], 1), cut_in_speed_difference=round(best["faster"], 1))
        # a short headway is a cost only if X is not pulling away: a car that merges in while clearly
        # faster opens the gap at once and Y need not react (unless Y did brake). Below cut_in_floor
        # (0.5 s) it is a cut-off however fast X is: Y has no margin if X brakes
        pulling_away = best["faster"] >= p.pull_away and best["brake"] < 1.0 and best["thw"] >= p.cut_in_floor
        implausible = best["move"] > 6.0  # more than a lane and a half sideways: a lane reference gone wrong
        out["cut_in"] = int(not implausible and (best["brake"] >= p.cut_in_brake
                                                 or (best["thw"] < p.cut_in_thw and not pulling_away)))
        if implausible:
            notes.append("implausible_lateral_move_%.0fm_check_by_eye" % best["move"])
        if best["thw"] < p.cut_in_thw and pulling_away:
            notes.append("cut_in_pulling_away_%.1fm/s" % best["faster"])
        elif not out["cut_in"] and best["thw"] < 1.0:
            notes.append("borderline_cut_in_thw_%.2fs" % best["thw"])

    # tailgating: X behind a moving Y in Y's lane at speed, not falling back (a gap opening by more
    # than 1 m/s is X dropping back, not following close by choice)
    # "in Y's path" for following: the boxes overlap sideways (a car half a lane over, straddling
    # the line right behind Y, is following it too), not only the 1.3 m of a lane change
    in_path = same & (np.abs(d_x) <= (scene.width[x] + scene.width[y]) / 2 - 0.1)
    follow = in_path & (behind_gap > 0) & (v_x >= 5.0) & (v_y >= 2.0) & (v_x - v_y >= -1.0)
    excused = np.zeros(T, bool)
    for t0, t in data["cut_ins_by"].get(y, []):  # Y moved in ahead of X: until 3 s after, Y's doing
        excused[int(t0):min(T, int(t) + 30)] = True
    # a following episode that starts with Y already close ahead began with Y arriving there (from
    # the next lane, a side road, a turn), not with X closing up: its first 3 s are excused too
    near = in_path & (behind_gap > 0)
    for t in range(T):
        if near[t] and (t == 0 or not near[t - 1]) and behind_gap[t] < 1.5 * max(v_x[t], 1.0):
            if sideways(scene, x, data, max(0, t - 20), min(T - 1, t + 5)) < 1.0:  # X did not move in itself
                excused[t:min(T, t + 30)] = True
    ok = follow & ~excused
    if ok.any():
        gap = smooth(np.where(ok, behind_gap, np.nan_to_num(behind_gap, nan=50.0)))
        thw = np.where(ok, gap / np.maximum(v_x, 0.1), np.inf)
        closing = v_x - v_y
        ttc = np.where(ok & (closing > 0.5), gap / np.maximum(closing, 1e-3), np.inf)
        run = longest = 0
        for flag in thw < p.tailgate_thw:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        out.update(tailgate_min_thw=round(float(thw.min()), 2), tailgate_run_s=round(longest * DT, 1),
                   tailgate_min_ttc=round(float(ttc.min()), 1) if np.isfinite(ttc.min()) else "",
                   tailgate_speed=round(float(v_x[ok].mean()), 1))
        out["tailgate"] = int(longest * DT >= p.tailgate_run or ttc.min() < p.tailgate_ttc)
        if thw.min() < 0.05:  # boxes touching at speed in one lane: two tracks of one car, or a bad lane match
            out["tailgate"] = 0
            notes.append("boxes_touch_while_following_check_tracks")
        if excused.any() and (follow & excused).any():
            notes.append("short_gap_after_other_cut_in_excused")

    # close pass: X in the next lane comes within close_gap of Y, closing in sideways itself
    best = None
    for t in steps:
        if not same[t] or abs(d_x[t]) <= IN_LANE or abs(v_x[t] - v_y[t]) < p.close_speed:
            continue
        g = box_gap(corners(scene.position[x, t, :2], scene.heading[x, t], scene.length[x, t], scene.width[x, t]),
                    corners(scene.position[y, t, :2], scene.heading[y, t], scene.length[y, t], scene.width[y, t]))
        if g > p.close_gap:
            continue
        k0 = max(steps[0], t - 20)
        closed = np.nanmax(np.abs(d_x[k0:t + 1])) - abs(d_x[t])
        mx, my = sideways(scene, x, data, k0, t), sideways(scene, y, data, k0, t)
        if closed >= 0.3 and mx > my and (best is None or g < best[1]):
            best = (t * DT, g, abs(v_x[t] - v_y[t]))
    if best:
        out.update(close_pass=1, close_pass_t=round(best[0], 1), close_pass_gap=round(best[1], 2),
                   close_pass_rel_speed=round(best[2], 1))
    return out, notes


def reason(me, other, r):
    parts = []
    if r.get("cut_in"):
        parts.append("%s moved %.1f m sideways into the %s's lane ahead of it at %.1f s, leaving it a headway of "
                     "%.2f s (gap %.1f m at %.1f m/s) and making it brake %.1f m/s²"
                     % (me, r["cut_in_lateral_move"], other, r["cut_in_t"], r["cut_in_min_thw"],
                        r["cut_in_min_gap"], r["cut_in_other_speed"], max(0.0, r["cut_in_other_brake"])))
    if r.get("tailgate"):
        ttc = r.get("tailgate_min_ttc")
        parts.append("%s followed the %s at about %.1f m/s with a headway below the limit for %.1f s (lowest %.2f s%s)"
                     % (me, other, r["tailgate_speed"], r["tailgate_run_s"], r["tailgate_min_thw"],
                        ", TTC %.1f s" % ttc if ttc != "" else ""))
    if r.get("close_pass"):
        parts.append("%s closed in sideways and passed the %s %.2f m apart at %.1f m/s relative speed at %.1f s"
                     % (me, other, r["close_pass_gap"], r["close_pass_rel_speed"], r["close_pass_t"]))
    return "; ".join(parts) + "." if parts else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="folder of scene pickles, or one file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", help="track id of the first agent (default: the SDC); needs --other")
    ap.add_argument("--other", help="track id of the second agent (default: the other object of interest)")
    ap.add_argument("--pairs", choices=PAIR_MODES, default="auto",
                    help="which pair to judge when --agent is not given (see scenes.pick_pair)")
    ap.add_argument("--cut-in-thw", type=float, default=0.8, help="s: a cut-in leaving less headway than this")
    ap.add_argument("--cut-in-brake", type=float, default=2.0, help="m/s²: or making the other brake this hard")
    ap.add_argument("--cut-in-floor", type=float, default=0.5,
                    help="s: below this headway a cut-in counts however fast the merging car is")
    ap.add_argument("--pull-away", type=float, default=2.0,
                    help="m/s: a merging car this much faster opens the gap itself; its short headway alone is not a cut-in")
    ap.add_argument("--tailgate-thw", type=float, default=0.9, help="s: following headway below this ...")
    ap.add_argument("--tailgate-run", type=float, default=1.0, help="s: ... for this long")
    ap.add_argument("--tailgate-ttc", type=float, default=2.0, help="s: or closing in with TTC below this")
    ap.add_argument("--close-gap", type=float, default=0.5, help="m: box-to-box gap of a close pass")
    ap.add_argument("--close-speed", type=float, default=3.0, help="m/s: relative speed of a close pass")
    args = ap.parse_args()


    rows = []
    for path in scene_files(args.scenes):
        try:
            scene, fmt = load_scene(path, Scene)
            if args.agent:
                a, o = scene.index(args.agent), scene.index(args.other)
                how = "named"
            else:
                pair = pick_pair(scene, args.pairs)
                if pair is None:
                    continue
                a, o, how = pair
            segments = rss.lane_segments(scene)
            graph = lanes.lane_graph(scene)
            data = {"ref": {}, "speed": {}, "lat": {}, "cut_ins_by": {}}
            for i in (a, o):
                data["ref"][i] = reference_paths(scene, i, graph)
                v = np.linalg.norm(scene.velocity[i], axis=1)
                data["speed"][i] = smooth(np.where(scene.valid[i], v, np.nan))
                data["lat"][i] = np.where(scene.valid[i], lateral_speed(scene, i, segments, rss), 0.0)
            # cut-ins first (with any effect), so that tailgating can excuse the gap a cut-in created
            first = {}
            for x, y in ((a, o), (o, a)):
                first[x], _ = directed(scene, x, y, data, args)
                if "cut_in_t" in first[x]:
                    data["cut_ins_by"][x] = [(first[x]["cut_in_start"] / DT, first[x]["cut_in_t"] / DT)]
            row = {"scene": path.stem, "scenario_id": scene.scenario_id, "format": fmt, "pair": how,
                   "first_id": scene.track_ids[a], "second_id": scene.track_ids[o]}
            names = {"sdc_with_ooi": ("SDC", "OOI"), "sdc_nearest": ("SDC", "other car"),
                     "ooi_pair": ("first OOI", "second OOI")}.get(how, ("first car", "second car"))
            for role, x, y, me, other in (("sdc", a, o) + names, ("ooi", o, a) + names[::-1]):
                res, notes = directed(scene, x, y, data, args)
                kinds = [k for k in ("cut_in", "tailgate", "close_pass") if res.get(k)]
                row.update({"%s_%s" % (role, k): v for k, v in res.items()})
                row["%s_kind" % role] = "+".join(kinds)
                row["%s_reason" % role] = reason(me, other, res)
                row["%s_notes" % role] = ";".join(notes)
            rows.append(row)
        except NeedsTorch as exc:  # the whole folder is in that format: stop with one clear message
            sys.exit(str(exc))
        except Exception as exc:  # one bad scene must not stop a batch
            print("skip %s: %s: %s" % (path.name, type(exc).__name__, exc), file=sys.stderr)

    fields = ["scene", "scenario_id", "format", "pair", "first_id", "second_id"]
    for role in ("sdc", "ooi"):
        fields += ["%s_%s" % (role, k) for k in (
            "kind", "cut_in", "tailgate", "close_pass", "cut_in_t", "cut_in_lateral_move", "cut_in_min_gap",
            "cut_in_min_thw", "cut_in_other_brake", "cut_in_other_speed", "cut_in_speed_difference",
            "cut_in_start", "tailgate_min_thw", "tailgate_run_s",
            "tailgate_min_ttc", "tailgate_speed", "close_pass_t", "close_pass_gap", "close_pass_rel_speed",
            "reason", "notes")]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    count = lambda role, k: sum(1 for r in rows if r.get("%s_%s" % (role, k)))
    print("%d scenes; SDC cut-in %d, tailgate %d, close pass %d; OOI cut-in %d, tailgate %d, close pass %d -> %s"
          % (len(rows), count("sdc", "cut_in"), count("sdc", "tailgate"), count("sdc", "close_pass"),
             count("ooi", "cut_in"), count("ooi", "tailgate"), count("ooi", "close_pass"), args.out))


if __name__ == "__main__":
    main()
