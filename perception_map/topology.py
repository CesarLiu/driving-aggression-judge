"""Rebuilding lane topology from a perception-only map and the tracks, so that right of way can be
judged where the map has lane lines but no links and nothing inside junctions.

A perception system sees painted lanes. It does not know which lane follows which, which lanes run
side by side, or how lanes connect across a junction, where nothing is painted. WOMD stands in for
such a map here: ``degrade`` keeps the lanes WOMD does not interpolate (the painted ones), the road
lines and the kerbs, and drops every link, the junction lanes, lights, stop signs, crosswalks and
speed limits. ``rebuild`` then restores:

  successors  a lane that starts within LINK_GAP m of another's end, heading the same way;
  neighbours  lanes running the same way LANE_MIN to LANE_MAX m apart over at least SIDE_OVERLAP m;
  junctions   a connector lane wherever vehicles left one lane at its end and joined another at its
              start without a successor link between them; its centreline is the mean of their
              paths, so only movements someone drove in the clip are rebuilt.

``score`` compares a rebuilt map with WOMD's junction lanes (movements found, false ones, centreline
error), and ``screen`` runs the right-of-way screen on rebuilt maps with WOMD's controls moved onto
the rebuilt connectors, which isolates what the topology costs. ``run`` does both with one rebuild
per scene (``score`` and ``screen`` do one each); rows are appended scene by scene, so a stopped part
resumes where it left off, and progress goes to stderr every 100 scenes.

  python topology.py run --scenes DIR --pairs screen_crossing.csv --score-out topo_score
      --screen-out topo_screen --part 0/8 [--only scenes.txt] [--modes rebuilt,rebuilt_geo,no_links]
  python topology.py report --real screen_crossing.csv --score topo_score --screen topo_screen
"""

import argparse
import copy
import csv
import glob
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "judge-driving-aggression" / "scripts"
sys.path.insert(0, str(SKILL))
from scenes import load_scene  # noqa: E402
from responsibility.scene import Scene  # noqa: E402
from responsibility.lanes import LANE_TYPES, lane_graph  # noqa: E402

KEEP = ("ROAD_LINE", "ROAD_EDGE")  # prefixes of the map features a perception system also sees
LINK_GAP = 1.0  # m between one lane's end and the next one's start
LINK_ANGLE = math.radians(20.0)
LANE_MIN, LANE_MAX = 2.5, 5.0  # m between the centrelines of neighbouring lanes
SIDE_ANGLE = math.radians(15.0)
SIDE_OVERLAP = 5.0  # m
MATCH = 2.0  # m: a track point is on a lane this close to its centreline
MATCH_ANGLE = math.radians(35.0)
MOVING = 1.0  # m/s: slower points are not used to match lanes
TRACK_MATCH, TRACK_ANGLE = MATCH, MATCH_ANGLE  # for finding where a vehicle left and joined lanes
END_NEAR = 8.0  # m: a vehicle left a lane at its end (joined at its start) within this
GAP_MIN, GAP_MAX = 3.0, 90.0  # m of path between leaving one lane and joining the next
TURN_MIN = math.radians(20.0)  # a move off or onto the middle of a lane must turn this much (else: lane change)
CUT_MERGE = 6.0  # m: cuts of one lane this close are one
RESAMPLE = 25  # points per connector path, by arc length
REGION = 60.0  # m around the judged pair for scoring
MATCH_LINE, MATCH_END = 2.0, 4.0  # m: a connector matches a true junction lane (mean distance, ends)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _heading(poly):
    d = np.diff(poly, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    return np.append(h, h[-1])


def _resample(poly, n):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, poly[:, k]) for k in range(poly.shape[1])], axis=1)


# --- degrade ---------------------------------------------------------------------------------

def degrade(scene):
    """A copy of ``scene`` with a perception-like map: painted lanes without links, road lines and
    kerbs; no junction lanes, controls, crosswalks or speed limits."""
    feats = {}
    for k, f in scene.map_features.items():
        t = str(f.get("type", ""))
        if t in LANE_TYPES and not f.get("interpolating"):
            feats[k] = {"type": t, "polyline": np.asarray(f["polyline"]), "interpolating": False,
                        "entry_lanes": [], "exit_lanes": [], "left_neighbor": [], "right_neighbor": []}
        elif t.startswith(KEEP):
            feats[k] = f
    out = copy.copy(scene)
    out.map_features = feats
    out.dynamic_map_states = {}
    return out


# --- rebuild ---------------------------------------------------------------------------------

def _lanes(feats):
    out = {}
    for k, f in feats.items():
        if f.get("type") in LANE_TYPES:
            p = np.asarray(f["polyline"], float)[:, :2]
            if len(p) >= 2:
                out[k] = p
    return out


def rebuild(scene):
    """``scene`` (a degraded one) with successors, neighbours and junction connectors rebuilt."""
    from scipy.spatial import cKDTree

    feats = {str(k): dict(f) for k, f in scene.map_features.items()}  # string ids, as LaneGraph and the truth use
    lanes = _lanes(feats)
    keys = list(lanes)
    heads = {k: _heading(lanes[k]) for k in keys}
    for k in keys:
        feats[k]["exit_lanes"], feats[k]["entry_lanes"] = [], []
        feats[k]["left_neighbor"], feats[k]["right_neighbor"] = [], []
    if not keys:
        out = copy.copy(scene)
        out.map_features = feats
        return out, {}
    starts = np.array([lanes[k][0] for k in keys])
    st_tree = cKDTree(starts)
    for a in keys:  # successors
        for j in st_tree.query_ball_point(lanes[a][-1], LINK_GAP):
            b = keys[j]
            if b != a and abs(wrap(heads[b][0] - heads[a][-1])) < LINK_ANGLE:
                feats[a]["exit_lanes"].append(b)
                feats[b]["entry_lanes"].append(a)
    pts = np.concatenate([lanes[k] for k in keys])
    owner = np.concatenate([[i] * len(lanes[k]) for i, k in enumerate(keys)])
    hs = np.concatenate([heads[k] for k in keys])
    tree = cKDTree(pts)
    # neighbours: b runs the same way LANE_MIN..LANE_MAX m to the side of a over SIDE_OVERLAP m of a
    pp = tree.query_pairs(LANE_MAX, output_type="ndarray")
    if len(pp):
        pp = np.vstack([pp, pp[:, ::-1]])
        i, j = pp[:, 0], pp[:, 1]
        r = pts[j] - pts[i]
        lat = r[:, 1] * np.cos(hs[i]) - r[:, 0] * np.sin(hs[i])
        keep = ((owner[i] != owner[j]) & (np.abs(wrap(hs[j] - hs[i])) <= SIDE_ANGLE)
                & (np.abs(lat) >= LANE_MIN) & (np.abs(lat) <= LANE_MAX))
        i, j, left = i[keep], j[keep], lat[keep] > 0
        seglen = np.r_[np.linalg.norm(np.diff(pts, axis=0), axis=1), 0.0]  # metres each point of a stands for
        seglen[np.r_[owner[1:] != owner[:-1], True]] = 0.0
        # each point of a counted once per (a, b, side): one int64 code per pair, sorted once
        group = (owner[i].astype(np.int64) * len(keys) + owner[j]) * 2 + left
        code = np.unique(group * len(pts) + i)
        groups, inv = np.unique(code // len(pts), return_inverse=True)
        metres = np.bincount(inv, weights=seglen[code % len(pts)], minlength=len(groups))
        for g_, m in zip(groups, metres):
            if m >= SIDE_OVERLAP:
                a_, b_, l_ = g_ // 2 // len(keys), g_ // 2 % len(keys), g_ % 2
                feats[keys[a_]]["left_neighbor" if l_ else "right_neighbor"].append({"feature_id": keys[b_]})

    # junction connectors from the tracks
    linked = {a: set(feats[a]["exit_lanes"]) | {n["feature_id"] for s in ("left_neighbor", "right_neighbor")
                                                for n in feats[a][s]} for a in keys}
    arcs = {k: np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(lanes[k], axis=0), axis=1))]) for k in keys}
    moves = []  # (left lane, where (None: its end), joined lane, where (None: its start), path between)
    speed = np.linalg.norm(scene.velocity, axis=-1)
    for n in range(scene.n_agents):
        if scene.types[n] != "VEHICLE":
            continue
        ok = scene.valid[n] & (speed[n] > MOVING)
        idx = np.flatnonzero(ok)
        if len(idx) < 5:
            continue
        P = scene.position[n, idx, :2].astype(float)
        H = scene.heading[n, idx].astype(float)
        dist, j = tree.query(P, distance_upper_bound=TRACK_MATCH)
        lane = []
        for q in range(len(P)):
            if np.isfinite(dist[q]) and abs(wrap(hs[j[q]] - H[q])) < TRACK_ANGLE:
                lane.append(keys[owner[j[q]]])
            else:
                lane.append(None)
        # runs of one lane; a connector between consecutive runs A ... B
        runs = []
        for q, l in enumerate(lane):
            if l is None:
                continue
            if runs and runs[-1][0] == l:
                runs[-1][2] = q
            else:
                runs.append([l, q, q])
        for (a, _, qa), (b, qb, _) in zip(runs, runs[1:]):
            turn = abs(wrap(H[qb] - H[qa]))
            if a == b or b in feats[a]["exit_lanes"] or ((b in linked[a] or a in linked.get(b, ())) and turn < TURN_MIN):
                continue
            # where it left a and joined b, in metres along them; near a's end or b's start counts as
            # the end or the start, elsewhere the lane is cut there (a through lane turned off or onto)
            ia = int(np.argmin(np.linalg.norm(lanes[a] - P[qa], axis=1)))
            ib = int(np.argmin(np.linalg.norm(lanes[b] - P[qb], axis=1)))
            sa, sb = arcs[a][ia], arcs[b][ib]
            cut_a = None if arcs[a][-1] - sa <= END_NEAR else sa
            cut_b = None if sb <= END_NEAR else sb
            if (cut_a is not None or cut_b is not None) and turn < TURN_MIN:
                continue  # a lane change between lanes not found to be neighbours, not a junction movement
            path = np.vstack([P[qa:qb + 1]])
            length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum() if len(path) > 1 else 0.0
            if GAP_MIN <= length + 2 * END_NEAR and length <= GAP_MAX:
                moves.append((a, cut_a, b, cut_b, path))

    # cut lanes where vehicles left or joined them (nearby cuts merged), then connect the pieces
    cuts = defaultdict(list)
    for a, ca, b, cb, _ in moves:
        if ca is not None:
            cuts[a].append(ca)
        if cb is not None:
            cuts[b].append(cb)
    pieces = {}  # lane -> (cut positions, piece ids)
    for lane, cs in cuts.items():
        merged = []
        for c in sorted(cs):
            if merged and c - merged[-1][-1] <= CUT_MERGE:
                merged[-1].append(c)
            else:
                merged.append([c])
        at = [float(np.mean(m)) for m in merged if 1.0 < np.mean(m) < arcs[lane][-1] - 1.0]
        if at:
            pieces[lane] = (at, _split(feats, lane, arcs[lane], at))

    def piece_ending(lane, cut):
        if lane not in pieces:
            return lane
        at, ids = pieces[lane]
        return ids[-1] if cut is None else ids[int(np.argmin([abs(c - cut) for c in at]))]

    def piece_starting(lane, cut):
        if lane not in pieces:
            return lane
        at, ids = pieces[lane]
        return ids[0] if cut is None else ids[int(np.argmin([abs(c - cut) for c in at])) + 1]

    paths = defaultdict(list)
    for a, ca, b, cb, path in moves:
        x, y = piece_ending(a, ca), piece_starting(b, cb)
        p0 = np.asarray(feats[x]["polyline"], float)[-1, :2]
        p1 = np.asarray(feats[y]["polyline"], float)[0, :2]
        paths[(x, y)].append(_resample(np.vstack([p0, path, p1]), RESAMPLE))
    made = {}
    for (x, y), ps in paths.items():
        cid = "rc_%s_%s" % (x, y)
        poly = np.median(np.stack(ps), axis=0)
        poly[0] = np.asarray(feats[x]["polyline"], float)[-1, :2]
        poly[-1] = np.asarray(feats[y]["polyline"], float)[0, :2]
        feats[cid] = {"type": "LANE_SURFACE_STREET", "polyline": np.column_stack([poly, np.zeros(len(poly))]),
                      "interpolating": True, "entry_lanes": [x], "exit_lanes": [y], "left_neighbor": [],
                      "right_neighbor": [], "metadata": {"rebuilt_from": len(ps)}}
        feats[x]["exit_lanes"].append(cid)
        feats[y]["entry_lanes"].append(cid)
        made[cid] = (x, y, len(ps))
    out = copy.copy(scene)
    out.map_features = feats
    return out, made


def _split(feats, lane, arc, at):
    """Cut ``lane`` at the arc positions ``at``: the first piece keeps the lane's id (so links and
    controls on it still hold), the others are <lane>_s1, _s2, ...; each piece leads to the next and
    the last keeps the lane's exits. Returns the piece ids."""
    f = feats[lane]
    poly = np.asarray(f["polyline"], float)
    bounds = [0.0] + list(at) + [float(arc[-1])]
    ids = [lane] + ["%s_s%d" % (lane, k) for k in range(1, len(bounds) - 1)]
    exits = list(f["exit_lanes"])
    for k, pid in enumerate(ids):
        lo, hi = bounds[k], bounds[k + 1]
        inner = poly[(arc > lo) & (arc < hi)]
        ends = [np.array([np.interp(v, arc, poly[:, c]) for c in range(poly.shape[1])]) for v in (lo, hi)]
        piece = np.vstack([ends[0], inner, ends[1]])
        g = dict(f, polyline=piece, entry_lanes=[ids[k - 1]] if k else list(f["entry_lanes"]),
                 exit_lanes=[ids[k + 1]] if k + 1 < len(ids) else exits)
        feats[pid] = g
    return ids


# --- truth: WOMD's junction movements ----------------------------------------------------------

def movements(scene):
    """WOMD's junction lanes as {lane: (entry, exit)} for interpolating lanes between painted ones."""
    feats = scene.map_features
    plain = {str(k) for k, f in feats.items() if f.get("type") in LANE_TYPES and not f.get("interpolating")}
    out = {}
    for k, f in feats.items():
        if f.get("type") not in LANE_TYPES or not f.get("interpolating"):
            continue
        for a in (str(x) for x in f.get("entry_lanes", [])):
            for b in (str(x) for x in f.get("exit_lanes", [])):
                if a in plain and b in plain:
                    out.setdefault(str(k), []).append((a, b))
    return out


def moving_points(scene):
    """Positions and headings of every vehicle seen moving (faster than MOVING) at three steps or more."""
    speed = np.linalg.norm(scene.velocity, axis=-1)
    P, H = [], []
    for n in range(scene.n_agents):
        if scene.types[n] != "VEHICLE":
            continue
        ok = scene.valid[n] & (speed[n] > MOVING)
        if ok.sum() >= 3:
            P.append(scene.position[n, ok, :2].astype(float))
            H.append(scene.heading[n, ok])
    return (np.concatenate(P), np.concatenate(H)) if P else (np.zeros((0, 2)), np.zeros(0))


def driven(moving, lane_pts, lane_dir):
    """Whether some vehicle drove most of ``lane``: matched to it on at least half its points
    (``moving``: moving_points of the scene)."""
    from scipy.spatial import cKDTree

    P, H = moving
    covered = np.zeros(len(lane_pts), bool)
    if len(P):
        d, j = cKDTree(lane_pts).query(P, distance_upper_bound=MATCH)
        hit = np.isfinite(d)
        hit[hit] &= np.abs(wrap(lane_dir[j[hit]] - H[hit])) < MATCH_ANGLE
        covered[j[hit]] = True
    return covered.mean() >= 0.5


def along(true_pts, true_dir, conns):
    """(error, connector) of the connector that runs along a true junction lane: its points lie within
    MATCH_LINE m of the lane on average, heading its way, and it covers at least half of the lane.
    None when no connector does."""
    from scipy.spatial import cKDTree

    tree = cKDTree(true_pts)
    best = None
    for c, p in conns.items():
        d, j = tree.query(p)
        h = _heading(p)
        if d.mean() > MATCH_LINE or np.abs(wrap(true_dir[j] - h)).mean() > MATCH_ANGLE:
            continue
        cover = (cKDTree(p).query(true_pts)[0] <= MATCH_LINE).mean()
        if cover >= 0.5 and (best is None or d.mean() < best[0]):
            best = (float(d.mean()), c)
    return best


def read_pairs(path):
    with open(path, encoding="utf-8") as f:
        return {r["scene"]: r for r in csv.DictReader(f)}


def files_of(path, part):
    files = sorted(Path(p) for p in glob.glob(str(Path(path) / "*.pkl")))
    if part:
        k, n = (int(x) for x in part.split("/"))
        files = files[k::n]
    return files


SCORE_COLS = ["scene", "true", "driven", "lost_driven", "found_lost_driven", "rebuilt", "rebuilt_true", "found",
              "found_driven", "centreline_err_m"]
SCREEN_COLS = ["scene", "mode", "case", "holder", "went_first", "sdc_violation", "ooi_violation",
               "sdc_aggressive_candidate", "ooi_aggressive_candidate", "sdc_control", "ooi_control", "sdc_turn",
               "ooi_turn", "connectors"]


def score_scene(scene, a, o, rebuilt, made):
    """The SCORE_COLS row of one scene: WOMD's junction lanes near the judged pair against the rebuild."""
    from scipy.spatial import cKDTree

    focus = cKDTree(np.concatenate([scene.position[i, scene.valid[i], :2] for i in (a, o)]).astype(float))
    truth = movements(scene)
    g = lane_graph(scene)
    near = lambda pts: focus.query(pts, distance_upper_bound=REGION)[0].min() < REGION
    true_lanes = [l for l in truth if l in g.points and near(g.points[l])]
    moving = moving_points(scene)
    is_driven = {l: driven(moving, g.points[l], g.direction[l]) for l in true_lanes}
    # a junction lane is lost only where no painted lane runs along it (WOMD often has both); only a lane
    # with a point within MATCH_LINE of it can (along's cover test), so a k-d tree picks those first
    plain = {k: v for k, v in _lanes(rebuilt.map_features).items() if not str(k).startswith("rc_")}
    keys = list(plain)
    tree = cKDTree(np.concatenate([plain[k] for k in keys])) if keys else None
    owner = np.repeat(np.arange(len(keys)), [len(plain[k]) for k in keys])
    is_lost = {}
    for l in true_lanes:
        hits = [] if tree is None else [i for h in tree.query_ball_point(g.points[l], MATCH_LINE + 1e-6) for i in h]
        cand = {keys[i] for i in set(owner[hits].tolist())}
        is_lost[l] = along(g.points[l], g.direction[l],
                           {k: plain[k] for k in keys if k in cand
                            and np.linalg.norm(plain[k] - g.points[l][0], axis=1).min() < 60}) is None
    conns = {cid: _resample(np.asarray(rebuilt.map_features[cid]["polyline"], float)[:, :2], RESAMPLE) for cid in made}
    conns = {c: p for c, p in conns.items() if near(p)}
    hit, err, used = set(), [], set()
    for l in true_lanes:
        best = along(g.points[l], g.direction[l], conns)
        if best:
            hit.add(l)
            err.append(best[0])
            used.add(best[1])
    lost_driven = [l for l in true_lanes if is_lost[l] and is_driven[l]]
    return {"true": len(true_lanes), "driven": sum(is_driven.values()), "lost_driven": len(lost_driven),
            "found_lost_driven": sum(l in hit for l in lost_driven), "rebuilt": len(conns), "rebuilt_true": len(used),
            "found": len(hit), "found_driven": sum(is_driven[l] for l in hit),
            "centreline_err_m": round(float(np.mean(err)), 2) if err else ""}


def screen_scene(scene, a, o, deg, rebuilt, made, modes, profile):
    """SCREEN_COLS rows of one scene: the right-of-way screen on each map of ``modes``."""
    import screen_crossing as SC
    from responsibility import lanes as lanes_mod
    from responsibility import right_of_way as R

    maps = {"rebuilt": lambda: with_true_controls(scene, rebuilt),
            "rebuilt_geo": lambda: with_true_controls(scene, rebuilt, geometric=True),
            "no_links": lambda: with_true_controls(scene, deg)}
    rows = []
    for mode in modes:
        out = SC.screen_pair(R, lanes_mod, maps[mode](), a, o, profile, 10)
        out.update(mode=mode, connectors=len(made))
        rows.append(out)
    return rows


def cmd_run(args):
    """Score and/or screen (args.cmd: score, screen or run for both) with one load and one rebuild per
    scene; rows are appended as each scene finishes, and a part already partly written resumes."""
    import jurisdiction

    profile = jurisdiction.load_profile(args.profile)
    pairs = read_pairs(args.pairs)
    only = set(Path(args.only).read_text().split()) if args.only else None
    modes = args.modes.split(",")
    outs = {}
    if args.cmd in ("score", "run"):
        outs["score"] = _Out(args.score_out if args.cmd == "run" else args.out, args.part, SCORE_COLS)
    if args.cmd in ("screen", "run"):
        outs["screen"] = _Out(args.screen_out if args.cmd == "run" else args.out, args.part, SCREEN_COLS)
    done = set.intersection(*(o.done for o in outs.values()))
    todo = [p for p in files_of(args.scenes, args.part)
            if p.stem in pairs and (only is None or p.stem in only) and p.stem not in done]
    t0 = time.perf_counter()
    for k, path in enumerate(todo, 1):
        row = pairs[path.stem]
        try:
            scene, _ = load_scene(path, Scene)
            a, o = scene.index(row["first_id"]), scene.index(row["second_id"])
            deg = degrade(scene)
            rebuilt, made = rebuild(deg)
            if "score" in outs and path.stem not in outs["score"].done:
                outs["score"].add([dict(score_scene(scene, a, o, rebuilt, made), scene=path.stem)])
            if "screen" in outs and path.stem not in outs["screen"].done:
                outs["screen"].add([dict(r, scene=path.stem) for r in screen_scene(scene, a, o, deg, rebuilt, made, modes, profile)])
        except Exception as exc:
            print("skip %s: %s: %s" % (path.stem, type(exc).__name__, exc), file=sys.stderr)
        if k % 100 == 0 or k == len(todo):
            rate = (time.perf_counter() - t0) / k
            print("%d/%d scenes, %.2f s/scene, %.0f min left" % (k, len(todo), rate, rate * (len(todo) - k) / 60),
                  file=sys.stderr, flush=True)
    for o in outs.values():
        o.close()


class _Out:
    """A part's CSV, appended to scene by scene; ``done`` holds the scenes already in it."""

    def __init__(self, folder, part, cols):
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.name = Path(folder) / ("part_%s.csv" % (part.replace("/", "of") if part else "all"))
        self.done = set()
        if self.name.exists():
            with open(self.name, encoding="utf-8") as f:
                self.done = {r["scene"] for r in csv.DictReader(f)}
        fresh = not self.name.exists() or self.name.stat().st_size == 0
        self.f = open(self.name, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=cols, extrasaction="ignore")
        if fresh:
            self.w.writeheader()
        self.n = 0

    def add(self, rows):
        self.w.writerows(rows)
        self.f.flush()
        self.n += len(rows)

    def close(self):
        self.f.close()
        print("%d rows (%d scenes before) -> %s" % (self.n, len(self.done), self.name))


def with_true_controls(scene, rebuilt, geometric=False):
    """WOMD's lights and stop signs moved onto the rebuilt map: a junction lane's control goes to the
    rebuilt connector of the same movement (and, ``geometric``, to one that runs along it: connectors
    often join split or stub lanes whose ids differ from WOMD's); controls on painted lanes stay."""
    truth = movements(scene)
    by_move = defaultdict(list)
    for lane, mv in truth.items():
        for m in mv:
            by_move[m].append(lane)
    conn = {}
    for cid, f in rebuilt.map_features.items():
        if str(cid).startswith("rc_"):
            conn[cid] = [lane for lane in by_move.get((f["entry_lanes"][0], f["exit_lanes"][0]), [])]
    if geometric:  # also any junction lane the connector runs along, whatever lanes it joins
        from scipy.spatial import cKDTree

        true = {k: p for k, p in _lanes(scene.map_features).items() if k in truth}
        trees = {k: (cKDTree(p), _heading(p)) for k, p in true.items()}
        for cid in conn:
            p = np.asarray(rebuilt.map_features[cid]["polyline"], float)[:, :2]
            h = _heading(p)
            for k, (tree, hd) in trees.items():
                d, j = tree.query(p)
                if d.mean() <= MATCH_LINE and np.abs(wrap(hd[j] - h)).mean() <= MATCH_ANGLE and k not in conn[cid]:
                    conn[cid].append(k)
    lanes_now = {str(k) for k in rebuilt.map_features}
    out = copy.copy(rebuilt)
    feats = dict(rebuilt.map_features)
    for k, f in scene.map_features.items():
        if f.get("type") != "STOP_SIGN":
            continue
        controlled = {str(x) for x in np.atleast_1d(f.get("lane", []))}
        new = [l for l in controlled if l in lanes_now] + [c for c, src in conn.items() if controlled & set(src)]
        if new:
            feats["stop_%s" % k] = dict(f, lane=new)
    out.map_features = feats
    lights = {}
    for k, v in scene.dynamic_map_states.items():
        lane = str(v.get("lane"))
        if lane in lanes_now:
            lights[k] = v
        for c, src in conn.items():
            if lane in src:
                lights["%s_%s" % (k, c)] = dict(v, lane=c)
    out.dynamic_map_states = lights
    return out


def _read_parts(folder):
    out = []
    for p in sorted(glob.glob(str(Path(folder) / "part_*.csv"))):
        with open(p, encoding="utf-8") as f:
            out += list(csv.DictReader(f))
    return out


NOT_JUNCTION = ("same-direction", "following", "no-conflict", "paths-do-not-cross", "too-short", "driveway")


def cmd_report(args):
    lines = []
    if args.score:
        s = _read_parts(args.score)
        tot = lambda k: sum(int(r[k]) for r in s)
        errs = [float(r["centreline_err_m"]) for r in s if r["centreline_err_m"]]
        lines += ["## Topology, %d scenes (WOMD junction lanes within %d m of the judged pair)" % (len(s), REGION),
                  "- WOMD junction lanes: %d, of which some vehicle drove (half their length): %d" % (
                      tot("true"), tot("driven")),
                  "- rebuilt connectors: %d; along a WOMD junction lane: %d (precision %.1f%%)" % (
                      tot("rebuilt"), tot("rebuilt_true"), 100.0 * tot("rebuilt_true") / max(1, tot("rebuilt"))),
                  "- recall: %.1f%% of all junction lanes, %.1f%% of driven ones" % (
                      100.0 * tot("found") / max(1, tot("true")), 100.0 * tot("found_driven") / max(1, tot("driven"))),
                  "- driven junction lanes the degraded map lost (no painted lane along them): %d; rebuilt: %d (%.1f%%)" % (
                      tot("lost_driven"), tot("found_lost_driven"),
                      100.0 * tot("found_lost_driven") / max(1, tot("lost_driven"))),
                  "- centreline error of true rebuilt connectors: mean %.2f m, median %.2f m" % (
                      float(np.mean(errs)) if errs else float("nan"), float(np.median(errs)) if errs else float("nan")),
                  ""]
    if args.screen:
        real = read_pairs(args.real)
        ev = defaultdict(dict)
        for r in _read_parts(args.screen):
            ev[r["mode"]][r["scene"]] = r
        for where in ("all", "junction"):
            for mode in [m for m in ("rebuilt", "rebuilt_geo", "no_links") if m in ev]:
                got = ev.get(mode, {})
                common = [x for x in got if x in real and (where == "all" or real[x]["case"] not in NOT_JUNCTION)]
                if not common:
                    continue
                same = lambda k: sum(real[x][k] == got[x][k] for x in common)
                lines.append("## %s map, true controls, %s scenes: %d" % (mode, where, len(common)))
                for k in ("case", "holder", "sdc_control", "ooi_control", "sdc_turn", "ooi_turn"):
                    lines.append("- %s unchanged: %.1f%%" % (k, 100.0 * same(k) / len(common)))
                for k in ("sdc_violation", "ooi_violation", "sdc_aggressive_candidate", "ooi_aggressive_candidate"):
                    r1 = {x for x in common if real[x][k] == "1"}
                    g1 = {x for x in common if got[x][k] == "1"}
                    lines.append("- %s: full map %d, here %d, both %d (kept %.0f%%, new %d)" % (
                        k, len(r1), len(g1), len(r1 & g1), 100.0 * len(r1 & g1) / max(1, len(r1)), len(g1 - r1)))
                moves = Counter((real[x]["case"], got[x]["case"]) for x in common if real[x]["case"] != got[x]["case"])
                lines.append("- commonest case changes: " + "; ".join(
                    "%s -> %s: %d" % (a, b, n) for (a, b), n in moves.most_common(6)))
                lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "screen", "run"):
        p = sub.add_parser(name)
        p.add_argument("--scenes", required=True)
        p.add_argument("--pairs", required=True, help="screen_crossing.csv: the judged pair of each scene")
        if name == "run":
            p.add_argument("--score-out", required=True)
            p.add_argument("--screen-out", required=True)
        else:
            p.add_argument("--out", required=True)
        p.add_argument("--part")
        p.add_argument("--profile", default="us-california")
        p.add_argument("--only", help="a file of scene names: only these")
        p.add_argument("--modes", default="rebuilt,no_links", help="screen: comma list of rebuilt, rebuilt_geo, no_links")
    r = sub.add_parser("report")
    r.add_argument("--real", help="screen_crossing.csv on the full map")
    r.add_argument("--score")
    r.add_argument("--screen")
    r.add_argument("--out")
    args = ap.parse_args()
    (cmd_report if args.cmd == "report" else cmd_run)(args)


if __name__ == "__main__":
    main()
