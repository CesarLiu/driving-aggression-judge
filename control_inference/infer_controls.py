"""Inferring the traffic controls a map lacks (traffic lights, stop signs, light states) from the lanes
and from how the other vehicles drive, so that right of way can be judged on a perception-only map.

WOMD maps are the teacher: they carry lights per lane (with a state at every step) and stop signs with
the lanes they control. The student sees only the lane centrelines and the tracks. Two models:

  static   per junction-entry lane: signal, stop sign or none.
  dynamic  per signalised lane and moment: go, arrow (protected turn) or red.

The right-of-way rules (screen_crossing.py with a jurisdiction profile) then run unchanged on a scene
whose controls were replaced by the inferred ones. That is the measure that counts: how many
right-of-way verdicts change against the full map, especially the violations.

Candidate lanes (which lanes get a prediction) are found from geometry alone, never from the
controls: lanes within REGION m of the judged pair's tracks that start where a predecessor branches,
or that cross another lane. Features come from the other vehicles only; the judged pair (SDC and the
other object of interest) is left out, or the model would learn "who went first had the right of
way", which is exactly wrong for the violations the screen looks for.

  python infer_controls.py extract --scenes DIR --pairs screen_crossing.csv --out feats --part 0/8
  python infer_controls.py train --features feats --out models.pkl
  python infer_controls.py evaluate --scenes DIR --pairs screen_crossing.csv --models models.pkl \
      --out eval --part 0/8 [--modes gated --junction-only]
  python infer_controls.py compare --real screen_crossing.csv --eval eval [eval_gated]

A third model works per junction (lanes grouped by their starts): none, two-way stop, all-way stop or
signal; it did worse than the lane model, so mode "gated" uses it only with --junction-model (to decide
all-way stops). Gated leaves a control unknown where the lane model is unsure (an always-unknown light,
so the rules give no verdict instead of a guessed one). Unless trained with --no-context, a second
lane model sees each lane's row with the first one's probabilities for every approach of its junction:
whether the crossing approaches stop too is what tells an all-way stop from a two-way one. Feature parts
extracted before lane starts were stored get them with: geometry --scenes DIR --features feats.
The perception-only setting: --map rebuilt (extract, evaluate) infers on the map with its topology
rebuilt from the tracks (perception_map/topology.py), the true controls moved onto the rebuilt lanes as
labels; --pool adds the tracks of the other scenes at the same place (perception_map/pool.py) to the
rebuild and to the static evidence. Models must be trained on features extracted the same way.
Set OMP_NUM_THREADS=1 when other jobs share the machine: sklearn's threads otherwise stall each other.
"""

import argparse
import copy
import csv
import glob
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "judge-driving-aggression" / "scripts"
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception_map"))  # pool, topology
from scenes import load_scene  # noqa: E402
from responsibility.scene import Scene  # noqa: E402
from responsibility.lanes import lane_graph  # noqa: E402
from responsibility.right_of_way import LIGHTS  # noqa: E402

REGION = 60.0  # m around the judged pair's tracks in which lanes get a prediction
CROSS_R = 1.0  # m: points of two lanes this close, at an angle, make the lanes cross
CROSS_ANGLE = math.radians(30.0)
ZONE = 25.0  # m before a lane's start: its approach
NEAR_LINE = 8.0  # m before the start: at the stop line
LAT = 1.8  # m either side of the approach axis
ALIGN = math.radians(30.0)  # a vehicle heading within this of the lane is on its approach
STOP_V = 0.5  # m/s: stopped
LONG_STOP = 40  # steps (4 s) of waiting at the line: a red, rarely a stop sign
CONTEXT = 40.0  # m between lane starts for the parallel, opposite and crossing approaches
W1, W3 = 10, 30  # steps either side of a moment for the dynamic features
STEP_EVERY = 5  # steps between the moments the dynamic model is applied at (0.5 s)
STATIC = ["none", "stop", "signal"]
DYNAMIC = ["go", "arrow", "red"]
JUNCTION = ["none", "two-way", "all-way", "signal"]
JUNCTION_R = 30.0  # m between lane starts of one junction (single linkage)
APPROACH_ANGLE = math.radians(45.0)  # lanes of one junction starting within this of each other: one approach
APPROACH_KEYS = ("seen", "crossed", "cross_stop", "cross_nostop", "front_stops", "front_stop_long", "waiting_end",
                 "queue_max")
WOMD_STATE = {"go": "LANE_STATE_GO", "arrow": "LANE_STATE_ARROW_GO", "red": "LANE_STATE_STOP"}
CLASSES = ("parallel", "opposite", "crossing")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --- candidate lanes -------------------------------------------------------------------------

def candidates(graph, focus):
    """Junction-entry lanes near ``focus`` points, from geometry only: a lane that starts where its
    predecessor branches, or that crosses another lane (not one it continues, sits beside or
    branches with)."""
    from scipy.spatial import cKDTree

    preds = defaultdict(set)
    for lane, exits in graph.exits.items():
        for x in exits:
            preds[x].add(lane)
    tree = cKDTree(focus)
    near = [l for l in graph.points if tree.query(graph.points[l], distance_upper_bound=REGION)[0].min() < REGION]
    if not near:
        return []
    pts = np.concatenate([graph.points[l] for l in near])
    dirs = np.concatenate([graph.direction[l] for l in near])
    owner = np.concatenate([[k] * len(graph.points[l]) for k, l in enumerate(near)])
    pairs = cKDTree(pts).query_pairs(CROSS_R, output_type="ndarray").reshape(-1, 2)
    i, j = pairs[:, 0], pairs[:, 1]
    keep = (owner[i] != owner[j]) & (np.abs(wrap(dirs[i] - dirs[j])) >= CROSS_ANGLE)
    lane_pairs, counts = np.unique(np.stack([owner[i[keep]], owner[j[keep]]], axis=1).reshape(-1, 2), axis=0,
                                   return_counts=True)  # point pairs per (a, b): the related test is per lane pair
    crosses = Counter()
    related = {}
    for (ka, kb), c in zip(lane_pairs, counts):
        a, b = near[ka], near[kb]
        if a not in related:
            related[a] = (set(graph.exits[a]) | preds[a] | set(graph.sides[a])
                          | {s for p in preds[a] for s in graph.exits[p]})
        if b not in related[a]:
            crosses[a] += int(c)
            crosses[b] += int(c)
    return [l for l in near if preds[l] and (any(len(graph.exits[p]) >= 2 for p in preds[l]) or crosses[l])], \
        {l: crosses[l] for l in near}, preds


# --- labels from the full map ----------------------------------------------------------------

def labels(scene, graph, lanes):
    lights = {str(v["lane"]): v["state"]["object_state"] for v in scene.dynamic_map_states.values()
              if v.get("type") == "TRAFFIC_LIGHT" and "lane" in v}
    stops = set()
    for f in scene.map_features.values():
        if f.get("type") == "STOP_SIGN":
            stops |= {str(x) for x in np.atleast_1d(f.get("lane", []))}
    lit = [l for l in lights if l in graph.points]
    static = {}
    for l in lanes:
        s0, h0 = graph.points[l][0], graph.direction[l][0]
        # a lane without a light of its own at a stop line with lights is signalised (WOMD leaves
        # some intersection lanes without one)
        shared = any(np.linalg.norm(graph.points[m][0] - s0) < 6.0 and abs(wrap(graph.direction[m][0] - h0)) < 0.35
                     for m in lit)
        static[l] = 2 if l in lights or shared else 1 if l in stops else 0
    return static, lights


# --- evidence per lane -----------------------------------------------------------------------

class Kin:
    """The other vehicles' tracks: everything but the judged pair and non-vehicles."""

    def __init__(self, scene, exclude=()):
        """``scene``: a Scene, or a dict of vehicle track arrays (pool.others_tracks)."""
        if isinstance(scene, dict):
            P, H, V, vel = scene["position"], scene["heading"], scene["valid"], scene["velocity"]
        else:
            keep = [i for i in range(scene.n_agents) if scene.types[i] == "VEHICLE" and i not in exclude]
            P, H, V, vel = scene.position[keep], scene.heading[keep], scene.valid[keep], scene.velocity[keep]
        self.P = P[:, :, :2].astype(float)
        self.H = H.astype(float)
        self.V = V
        self.S = np.linalg.norm(vel, axis=-1)
        self.N, self.T = self.V.shape


PER_SUM = ("crossed", "entered", "cross_stop", "cross_nostop", "waiting_end", "seen")
PER_JOIN = ("cross_vmin", "front_stops")


def pooled_per(per, others):
    """``per`` (evidence of the scene's own vehicles) with the per-vehicle evidence of other scenes'
    vehicles at the same lane added: counts summed, per-vehicle values joined."""
    out = dict(per)
    for o in others:
        for k in PER_SUM:
            out[k] = out[k] + o[k]
        for k in PER_JOIN:
            out[k] = out[k] + o[k]
    return out


def evidence(kin, graph, lane):
    """What the other vehicles did at the start of ``lane``: per step, and per vehicle."""
    T = kin.T
    s0, h0 = graph.points[lane][0], graph.direction[lane][0]
    u = np.array([math.cos(h0), math.sin(h0)])
    rel = kin.P - s0
    d = rel @ u
    lat = rel[..., 1] * u[0] - rel[..., 0] * u[1]
    align = np.abs(wrap(kin.H - h0)) < ALIGN
    inzone = kin.V & align & (np.abs(lat) < LAT) & (d > -ZONE) & (d < 1.0)
    near = inzone & (d > -NEAR_LINE)
    stopped = kin.S < STOP_V
    ev = {"cross": [], "enter": [], "front_stopped": np.zeros(T, bool), "front_elapsed": np.zeros(T),
          "front_speed": np.full(T, -1.0), "n_in": inzone.sum(0) if kin.N else np.zeros(T),
          "n_stop_in": (inzone & stopped).sum(0) if kin.N else np.zeros(T)}
    per = {"crossed": 0, "entered": 0, "cross_stop": 0, "cross_nostop": 0, "cross_vmin": [], "front_stops": [],
           "waiting_end": 0, "seen": int(inzone.any(1).sum()) if kin.N else 0}
    if not kin.N:
        return ev, per
    ar = np.arange(T)
    dz = np.where(inzone, d, -np.inf)
    front = dz.argmax(0)
    has = np.isfinite(dz.max(0))
    ev["front_stopped"] = has & stopped[front, ar] & near[front, ar]
    ev["front_speed"] = np.where(has, kin.S[front, ar], -1.0)
    run_start, cur = None, None
    for t in range(T):  # waiting at the head of the queue: how long, per vehicle
        if ev["front_stopped"][t]:
            if cur != front[t]:
                cur, run_start = front[t], t
            ev["front_elapsed"][t] = (t - run_start + 1) * 0.1
        else:
            if cur is not None:
                per["front_stops"].append(t - run_start)
            cur = None
    if cur is not None:
        per["front_stops"].append(T - run_start)
        per["waiting_end"] = 1
    lane_pts = graph.points[lane][2:20]
    step = (kin.V[:, 1:] & (d[:, 1:] >= 0) & (d[:, :-1] < 0) & inzone[:, :-1] & (np.abs(lat[:, 1:]) < 2.5))
    for n in range(kin.N):
        ts = np.flatnonzero(step[n])
        if not ts.size:
            continue
        t = int(ts[0]) + 1
        ev["cross"].append((n, t))
        per["crossed"] += 1
        before = near[n, :t]
        if before.sum() >= 3:
            vmin = float(kin.S[n, :t][before].min())
            per["cross_vmin"].append(vmin)
            per["cross_stop"] += vmin < STOP_V
            per["cross_nostop"] += vmin > 2.0
        after = np.arange(t, min(t + 30, T))
        after = after[kin.V[n, after]]
        if after.size and len(lane_pts):
            dd = np.linalg.norm(kin.P[n, after][:, None] - lane_pts[None], axis=-1)
            if dd.min() < 1.5:
                ev["enter"].append((n, t))
                per["entered"] += 1
    return ev, per


def relations(graph, lanes):
    """For each candidate lane, the candidates within CONTEXT m by relative direction."""
    out = {l: {c: [] for c in CLASSES} for l in lanes}
    if not lanes:
        return out
    s = np.array([graph.points[l][0] for l in lanes], float)
    h = np.array([graph.direction[l][0] for l in lanes], float)
    u = np.stack([np.cos(h), np.sin(h)], axis=1)
    r = s[None, :, :] - s[:, None, :]  # [a, b]: from a's start to b's
    dist = np.sqrt((r * r).sum(-1))
    dh = np.abs(wrap(h[None, :] - h[:, None]))
    lat = np.abs(r[..., 1] * u[:, None, 0] - r[..., 0] * u[:, None, 1])
    lon = np.abs(r[..., 0] * u[:, None, 0] + r[..., 1] * u[:, None, 1])
    ok = (dist <= CONTEXT) & ~np.eye(len(lanes), dtype=bool)
    par = ok & (dh < math.radians(20)) & (lat > 2.0) & (lon < 15)
    opp = ok & ~par & (dh > math.radians(150))
    cro = ok & ~par & ~opp & (dh > math.radians(60)) & (dh < math.radians(120))
    for k, a in enumerate(lanes):
        for c, m in (("parallel", par), ("opposite", opp), ("crossing", cro)):
            out[a][c] = [lanes[x] for x in np.flatnonzero(m[k])]
    return out


# --- junctions -------------------------------------------------------------------------------
#
# Whether a stop-controlled junction is all-way or two-way decides most stop-sign verdicts, and the
# lane model gets it wrong where it misses (or adds) the other approach's sign. The question is one of
# the whole junction: does traffic stop on every approach, or does some approach drive through? So a
# second model answers it per junction, from every approach's evidence together.

def starts_of(graph, lanes):
    """x, y and heading at the start of each lane."""
    return np.asarray([[*graph.points[l][0][:2], graph.direction[l][0]] for l in lanes], np.float32).reshape(-1, 3)


def junction_groups(starts):
    """Junction index per candidate lane (single linkage of lane starts within JUNCTION_R), and the
    approach index per lane within its junction (start headings within APPROACH_ANGLE)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(starts)
    pairs = cKDTree(starts[:, :2]).query_pairs(JUNCTION_R, output_type="ndarray").reshape(-1, 2)
    junction = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)),
                                    directed=False)[1]
    approach = np.zeros(n, int)
    for j in np.unique(junction):
        heads = []
        for i in np.flatnonzero(junction == j):
            k = next((k for k, h in enumerate(heads) if abs(wrap(starts[i, 2] - h)) < APPROACH_ANGLE), None)
            if k is None:
                k = len(heads)
                heads.append(starts[i, 2])
            approach[i] = k
    return junction, approach


def junction_label(y, approach):
    """From the lane labels: signal if any lane has a light; all-way if every approach (two or more)
    has a stop sign; two-way if some do; none otherwise."""
    if (y == 2).any():
        return 3
    stopped = set(approach[y == 1])
    if not stopped:
        return 0
    return 2 if stopped == set(approach) and len(stopped) >= 2 else 1


JUNCTION_NAMES = ["lanes", "approaches", "app_seen", "app_crossed", "app_nostop", "app_stop", "app_only_stop",
                  "cross_stop", "cross_nostop", "crossed", "front_stops", "front_stop_long", "waiting_end",
                  "queue_max", "front_stop_max", "frac_stop_min", "frac_stop_max", "frac_stop_mean", "app_unseen"]


def junction_rows(X, names, junction, approach):
    """One row per junction (JUNCTION_NAMES) from its lanes' static rows; and the lanes of each."""
    col = {k: names.index(k) for k in APPROACH_KEYS + ("front_stop_max",)}
    rows, members = [], []
    for j in np.unique(junction):
        idx = np.flatnonzero(junction == j)
        apps = []
        for a in np.unique(approach[idx]):
            sel = idx[approach[idx] == a]
            apps.append({k: float(X[sel, col[k]].max() if k in ("queue_max", "front_stop_max") else X[sel, col[k]].sum())
                         for k in col})
        crossed = [p for p in apps if p["cross_stop"] + p["cross_nostop"] > 0]
        frac = [p["cross_stop"] / (p["cross_stop"] + p["cross_nostop"]) for p in crossed]
        tot = lambda k: sum(p[k] for p in apps)
        rows.append([len(idx), len(apps), sum(p["seen"] > 0 for p in apps), len(crossed),
                     sum(p["cross_nostop"] > 0 for p in apps), sum(p["cross_stop"] > 0 for p in apps),
                     sum(p["cross_stop"] > 0 and p["cross_nostop"] == 0 for p in apps),
                     tot("cross_stop"), tot("cross_nostop"), tot("crossed"), tot("front_stops"), tot("front_stop_long"),
                     tot("waiting_end"), max(p["queue_max"] for p in apps), max(p["front_stop_max"] for p in apps),
                     min(frac, default=-1.0), max(frac, default=-1.0), float(np.mean(frac)) if frac else -1.0,
                     len(apps) - len(crossed)])
        members.append(idx)
    return np.asarray(rows, np.float32).reshape(-1, len(JUNCTION_NAMES)), members


# A second stage per lane: its static row and the first stage's probabilities, beside those of the
# other approaches of its junction. Whether an approach has a stop sign of an all-way stop depends on
# whether the crossing approaches stop too, which the first stage sees only as raw counts.

CONTEXT_NAMES = ["ctx_approaches", "ctx_own_none", "ctx_own_stop", "ctx_own_signal",
                 "ctx_stop_min", "ctx_stop_max", "ctx_stop_mean", "ctx_signal_min", "ctx_signal_max", "ctx_signal_mean",
                 "ctx_n_stop", "ctx_n_none", "ctx_cross_stop_p", "ctx_cross_signal_p", "ctx_opp_stop_p",
                 "ctx_others_cross_stop", "ctx_others_cross_nostop", "ctx_others_frac_stop",
                 "ctx_cro_cross_stop", "ctx_cro_cross_nostop"]


def context_rows(X, names, starts, p):
    """Second-stage rows for one scene's candidate lanes: static rows ``X``, first-stage probabilities
    ``p`` (none, stop, signal), and CONTEXT_NAMES from the other approaches of the lane's junction."""
    junction, approach = junction_groups(starts)
    cs, cn = names.index("cross_stop"), names.index("cross_nostop")
    out = np.zeros((len(X), len(CONTEXT_NAMES)), np.float32)
    agg = lambda v: [min(v), max(v), float(np.mean(v))] if v else [-1.0, -1.0, -1.0]
    for j in np.unique(junction):
        idx = np.flatnonzero(junction == j)
        lanes = {a: idx[approach[idx] == a] for a in np.unique(approach[idx])}
        head = {a: starts[m[0], 2] for a, m in lanes.items()}
        mp = {a: p[m].mean(0) for a, m in lanes.items()}
        stops = {a: (X[m, cs].sum(), X[m, cn].sum()) for a, m in lanes.items()}
        for a, m in lanes.items():
            others = [b for b in lanes if b != a]
            rel = {b: abs(float(wrap(head[b] - head[a]))) for b in others}
            cro = [b for b in others if math.radians(60) < rel[b] < math.radians(120)]
            opp = [b for b in others if rel[b] > math.radians(150)]
            s, n = sum(stops[b][0] for b in others), sum(stops[b][1] for b in others)
            out[m] = [len(lanes), *mp[a], *agg([mp[b][1] for b in others]), *agg([mp[b][2] for b in others]),
                      sum(mp[b][1] > 0.5 for b in others), sum(mp[b][0] > 0.5 for b in others),
                      float(np.mean([mp[b][1] for b in cro])) if cro else -1.0,
                      float(np.mean([mp[b][2] for b in cro])) if cro else -1.0,
                      float(np.mean([mp[b][1] for b in opp])) if opp else -1.0,
                      s, n, s / (s + n) if s + n else -1.0,
                      sum(stops[b][0] for b in cro), sum(stops[b][1] for b in cro)]
    return np.hstack([X, p.astype(np.float32), out])


def context_data(d, p, rows):
    """context_rows for the static rows ``rows`` (indices into ``d``) of every scene, in that order."""
    by = defaultdict(list)
    for i in rows:
        by[d["static_scene"][i]].append(i)
    out = np.zeros((len(d["static_y"]), len(d["names"][0]) + 3 + len(CONTEXT_NAMES)), np.float32)
    for idx in by.values():
        idx = np.asarray(idx)
        out[idx] = context_rows(d["static_X"][idx], d["names"][0], d["static_start"][idx], p[idx])
    return out[np.asarray(rows)]


def junction_of_lanes(d, pred):
    """junction_label of every junction of junction_data (same order) from lane predictions ``pred``."""
    by = defaultdict(list)
    for i, s in enumerate(d["static_scene"]):
        by[s].append(i)
    out = []
    for idx in by.values():
        idx = np.asarray(idx)
        junction, approach = junction_groups(d["static_start"][idx])
        out += [junction_label(pred[idx][junction == j], approach[junction == j]) for j in np.unique(junction)]
    return np.asarray(out, np.int8)


def junction_data(d):
    """Junction rows, labels and scenes for every scene of the extracted features ``d``."""
    by = defaultdict(list)
    for i, s in enumerate(d["static_scene"]):
        by[s].append(i)
    X, y, scenes = [], [], []
    for s, idx in by.items():
        idx = np.asarray(idx)
        junction, approach = junction_groups(d["static_start"][idx])
        rows, members = junction_rows(d["static_X"][idx], d["names"][0], junction, approach)
        X.append(rows)
        scenes += [s] * len(rows)
        y += [junction_label(d["static_y"][idx][m], approach[m]) for m in members]
    return np.concatenate(X), np.asarray(y, np.int8), scenes


def _count(events, t, w):
    return sum(1 for _, s in events if t - w <= s <= t + w)


def _union(evs, key):
    return sorted({e for ev in evs for e in ev[key]})


def scene_rows(scene, exclude, focus, dyn_lanes=None, dyn_steps=None, cache=None, pool=()):
    """Static rows for every candidate lane, and dynamic rows for ``dyn_lanes`` (default: the lanes
    with a light in the map, at their known states) at ``dyn_steps``. A ``cache`` dict passed to
    several calls on one scene keeps the candidates and their evidence from the first. ``pool``: Kin
    of other scenes at the same place, whose per-vehicle evidence adds to the static rows (anything
    timed, and the dynamic rows, come from the scene's own vehicles)."""
    if cache is not None and "ctx" in cache:
        ctx = cache["ctx"]
    else:
        graph = lane_graph(scene)
        found = candidates(graph, focus)
        if not found or not found[0]:
            ctx = None
        else:
            lanes = found[0]
            kin = Kin(scene, exclude)
            ev = {l: evidence(kin, graph, l) for l in lanes}
            if pool:
                ev = {l: (e, pooled_per(p, [evidence(k, graph, l)[1] for k in pool])) for l, (e, p) in ev.items()}
            starts = np.array([graph.points[l][0] for l in lanes], float)
            gap = np.sqrt(((starts[:, None] - starts[None]) ** 2).sum(-1))
            ctx = (graph, found, {l: e for l, (e, _) in ev.items()}, {l: p for l, (_, p) in ev.items()},
                   relations(graph, lanes), labels(scene, graph, lanes),
                   dict(zip(lanes, ((gap < 1.5).sum(1) - 1).tolist())))  # siblings: other starts within 1.5 m
        if cache is not None:
            cache["ctx"] = ctx
    if ctx is None:
        return None
    graph, (lanes, crosses, preds), evs, pers, rel, (static_y, lights), siblings = ctx
    T = scene.n_steps

    def own_static(l):
        p = pers[l]
        fs = p["front_stops"]
        turn = math.degrees(float(wrap(graph.direction[l][-1] - graph.direction[l][0])))
        sib = siblings[l]
        return {"seen": p["seen"], "crossed": p["crossed"], "entered": p["entered"], "cross_stop": p["cross_stop"],
                "cross_nostop": p["cross_nostop"],
                "frac_cross_stop": p["cross_stop"] / max(1, p["cross_stop"] + p["cross_nostop"]),
                "cross_vmin_mean": float(np.mean(p["cross_vmin"])) if p["cross_vmin"] else -1.0,
                "front_stops": len(fs), "front_stop_max": max(fs) * 0.1 if fs else 0.0,
                "front_stop_mean": float(np.mean(fs)) * 0.1 if fs else 0.0,
                "front_stop_long": sum(f >= LONG_STOP for f in fs), "waiting_end": p["waiting_end"],
                "queue_max": int(evs[l]["n_stop_in"].max()), "turn_deg": turn, "abs_turn": abs(turn),
                "length": graph.length[l], "siblings": sib, "crosses": crosses.get(l, 0),
                "pred_exits": max(len(graph.exits[p]) for p in preds[l])}

    own = {l: own_static(l) for l in lanes}
    static_rows, names_s = [], None
    for l in lanes:
        row = dict(own[l])
        for c in CLASSES:
            group = rel[l][c]
            row["%s_n" % c] = len(group)
            for k in ("seen", "crossed", "cross_stop", "cross_nostop", "front_stops", "front_stop_long", "waiting_end"):
                row["%s_%s" % (c, k)] = sum(own[m][k] for m in group)
            row["%s_front_stop_max" % c] = max([own[m]["front_stop_max"] for m in group], default=0.0)
            ce = _union([evs[m] for m in group], "cross")
            fst = evs[l]["front_stopped"]
            # others going while this approach waits at the line (a red), or while it goes too
            row["%s_cross_while_waiting" % c] = sum(1 for _, t in ce if fst[min(t, T - 1)])
            mine = [t for _, t in evs[l]["cross"]]
            row["%s_cross_with_mine" % c] = sum(1 for _, t in ce if any(abs(t - s) <= W1 for s in mine))
        if names_s is None:
            names_s = list(row)
        static_rows.append([row[k] for k in names_s])

    dyn_rows, dyn_y, dyn_meta, names_d = [], [], [], None
    if dyn_lanes is None:
        dyn_lanes = [l for l in lanes if l in lights]
    for l in dyn_lanes:
        if l not in evs:
            continue
        states = lights.get(l)
        e = evs[l]
        groups = {c: [evs[m] for m in rel[l][c]] for c in CLASSES}
        unions = {c: (_union(groups[c], "cross"), _union(groups[c], "enter")) for c in CLASSES}
        steps = dyn_steps if dyn_steps is not None else range(0, T, 10)
        for t in steps:
            y = None
            if dyn_steps is None:  # training: the lane's own recorded state
                s = states[t] if states is not None and t < len(states) else None
                y = LIGHTS.get(s) if s else None
                if y not in DYNAMIC:
                    continue
            row = {"t": t * 0.1, "cross1": _count(e["cross"], t, W1), "cross3": _count(e["cross"], t, W3),
                   "enter1": _count(e["enter"], t, W1), "enter3": _count(e["enter"], t, W3),
                   "front_stopped": int(e["front_stopped"][t]), "front_elapsed": e["front_elapsed"][t],
                   "front_speed": e["front_speed"][t], "n_in": int(e["n_in"][t]), "n_stop_in": int(e["n_stop_in"][t])}
            for c in CLASSES:
                cr, en = unions[c]
                row["%s_cross1" % c] = _count(cr, t, W1)
                row["%s_cross3" % c] = _count(cr, t, W3)
                row["%s_enter1" % c] = _count(en, t, W1)
                row["%s_front_stopped" % c] = sum(int(g["front_stopped"][t]) for g in groups[c])
                row["%s_front_elapsed" % c] = max([g["front_elapsed"][t] for g in groups[c]], default=0.0)
            row.update({"s_" + k: v for k, v in own[l].items()})
            if names_d is None:
                names_d = list(row)
            dyn_rows.append([row[k] for k in names_d])
            dyn_y.append(DYNAMIC.index(y) if y else -1)
            dyn_meta.append((l, t))
    return {"lanes": lanes, "static_X": np.asarray(static_rows, np.float32), "names_static": names_s,
            "static_start": starts_of(graph, lanes),
            "static_y": np.asarray([static_y[l] for l in lanes], np.int8),
            "dyn_X": np.asarray(dyn_rows, np.float32) if dyn_rows else np.zeros((0, 0), np.float32), "names_dyn": names_d,
            "dyn_y": np.asarray(dyn_y, np.int8), "dyn_meta": dyn_meta}


# --- the judged pairs ------------------------------------------------------------------------

def read_pairs(path):
    with open(path, encoding="utf-8") as f:
        return {r["scene"]: r for r in csv.DictReader(f)}


def pair_of(scene, row):
    a, o = scene.index(row["first_id"]), scene.index(row["second_id"])
    focus = np.concatenate([scene.position[i, scene.valid[i], :2] for i in (a, o)]).astype(float)
    return a, o, focus


def map_of(name, scene, args):
    """The scene the controls are inferred on, carrying its true controls (the labels), and the Kin of
    the other scenes pooled into it (``args.pool``: pool.py's folder). ``args.map``: "full", WOMD's
    map; or "rebuilt", the perception-only map with its topology rebuilt from the tracks (pooled ones
    too) and the true controls moved onto the rebuilt lanes by geometry."""
    groups = []
    if args.pool:
        import pool as PL

        groups = PL.others_tracks(scene, _places(args.pool).get(name, []), args.pool)
    if args.map == "full":
        return scene, [Kin(g) for g in groups]
    import topology as TP

    rebuilt, _ = TP.rebuild_pooled(TP.degrade(scene), groups)
    return TP.with_true_controls(scene, rebuilt, geometric=True), [Kin(g) for g in groups]


_PLACES = {}


def _places(folder):
    if folder not in _PLACES:
        import pool as PL

        _PLACES[folder] = PL.load_places(folder)
    return _PLACES[folder]


def files_of(path, part):
    files = sorted(Path(p) for p in glob.glob(str(Path(path) / "*.pkl")))
    if part:
        k, n = (int(x) for x in part.split("/"))
        files = files[k::n]
    return files


def cell_of(scene):
    """A spatial group for the split: scenes in one 500 m cell share streets, so they stay on one side."""
    t = min(10, scene.n_steps - 1)
    x, y = scene.position[scene.sdc, t, :2]
    return "%d_%d" % (math.floor(x / 500.0), math.floor(y / 500.0))


# --- extract ---------------------------------------------------------------------------------

def cmd_extract(args):
    pairs = read_pairs(args.pairs)
    out = {"scenes": [], "cells": {}, "static_X": [], "static_y": [], "static_scene": [], "static_lane": [],
           "dyn_X": [], "dyn_y": [], "dyn_scene": [], "names_static": None, "names_dyn": None}
    for path in files_of(args.scenes, args.part):
        row = pairs.get(path.stem)
        if row is None:
            continue
        try:
            scene, _ = load_scene(path, Scene)
            a, o, focus = pair_of(scene, row)
            on, pool = map_of(path.stem, scene, args)
            r = scene_rows(on, {a, o}, focus, pool=pool)
        except Exception as exc:  # one bad scene must not stop the part
            print("skip %s: %s: %s" % (path.stem, type(exc).__name__, exc), file=sys.stderr)
            continue
        out["scenes"].append(path.stem)
        out["cells"][path.stem] = cell_of(scene)
        if r is None:
            continue
        out["names_static"] = out["names_static"] or r["names_static"]
        out["static_X"].append(r["static_X"])
        out["static_y"].append(r["static_y"])
        out["static_scene"] += [path.stem] * len(r["lanes"])
        out["static_lane"] += r["lanes"]
        out.setdefault("static_start", []).append(r["static_start"])
        if len(r["dyn_y"]):
            out["names_dyn"] = out["names_dyn"] or r["names_dyn"]
            out["dyn_X"].append(r["dyn_X"])
            out["dyn_y"].append(r["dyn_y"])
            out["dyn_scene"] += [path.stem] * len(r["dyn_y"])
    for k in ("static_X", "static_y", "static_start", "dyn_X", "dyn_y"):
        out[k] = np.concatenate(out[k]) if out.get(k) else np.zeros((0,))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    name = "part_%s.pkl" % (args.part.replace("/", "of") if args.part else "all")
    with open(Path(args.out) / name, "wb") as f:
        pickle.dump(out, f)
    print("%d scenes; %d candidate lanes (%s); %d dynamic rows (%s) -> %s" % (
        len(out["scenes"]), len(out["static_y"]), dict(Counter(STATIC[i] for i in out["static_y"])),
        len(out["dyn_y"]), dict(Counter(DYNAMIC[i] for i in out["dyn_y"])), Path(args.out) / name))


def cmd_geometry(args):
    """Add the lane starts to feature parts extracted before scene_rows returned them."""
    files = {f.stem: f for f in files_of(args.scenes, None)}
    for p in sorted(glob.glob(str(Path(args.features) / "part_*.pkl"))):
        if args.part and Path(p).stem != "part_" + args.part.replace("/", "of"):
            continue
        d = pickle.load(open(p, "rb"))
        if len(d.get("static_start", ())) == len(d["static_y"]):
            continue
        starts, graph, current = [], None, None
        for s, lane in zip(d["static_scene"], d["static_lane"]):
            if s != current:
                graph, current = lane_graph(load_scene(files[s], Scene)[0]), s
            starts.append(starts_of(graph, [lane])[0])
        d["static_start"] = np.asarray(starts, np.float32).reshape(-1, 3)
        with open(p, "wb") as f:
            pickle.dump(d, f)
        print("%d lane starts -> %s" % (len(starts), p), flush=True)


# --- train -----------------------------------------------------------------------------------

def _load_features(folder):
    parts = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(str(Path(folder) / "part_*.pkl")))]
    cat = lambda k: np.concatenate([p[k] for p in parts if len(p[k])])
    lst = lambda k: [x for p in parts for x in p[k]]
    cells = {}
    for p in parts:
        cells.update(p["cells"])
    names = next(p["names_static"] for p in parts if p["names_static"]), next(p["names_dyn"] for p in parts if p["names_dyn"])
    return {"static_X": cat("static_X"), "static_y": cat("static_y"), "static_scene": lst("static_scene"),
            "static_lane": lst("static_lane"), "dyn_X": cat("dyn_X"), "dyn_y": cat("dyn_y"),
            "static_start": cat("static_start") if all("static_start" in p for p in parts) else None,
            "dyn_scene": lst("dyn_scene"), "cells": cells, "names": names}


def _report(title, y, p, classes):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p, labels=range(len(classes)))
    lines = ["%s: accuracy %.3f on %d" % (title, float((y == p).mean()), len(y)),
             "  true \\ predicted  " + "  ".join("%8s" % c for c in classes)]
    for i, c in enumerate(classes):
        recall = cm[i, i] / max(1, cm[i].sum())
        precision = cm[i, i] / max(1, cm[:, i].sum())
        lines.append("  %-16s  " % c + "  ".join("%8d" % x for x in cm[i]) + "   recall %.3f precision %.3f" % (
            recall, precision))
    return "\n".join(lines)


def cmd_train(args):
    from sklearn.ensemble import HistGradientBoostingClassifier

    d = _load_features(args.features)
    scenes = sorted(d["cells"])
    cells = sorted(set(d["cells"].values()))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(cells))
    fold_of_cell = {cells[i]: k % args.folds for k, i in enumerate(order)}
    fold = {s: fold_of_cell[d["cells"][s]] for s in scenes}
    fs = np.array([fold[s] for s in d["static_scene"]])
    fd = np.array([fold[s] for s in d["dyn_scene"]])
    models, ps, pd_ = {}, np.zeros(len(fs), int), np.zeros(len(fd), int)
    static_model = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
    context = args.context and d["static_start"] is not None
    pp, p2 = np.zeros((len(fs), len(STATIC))), np.zeros(len(fs), int)  # first-stage probabilities, second stage
    if d["static_start"] is not None:
        jX, jy, jscene = junction_data(d)
        fj = np.array([fold[s] for s in jscene])
        pj = np.zeros((len(jy), len(JUNCTION)))
    for k in range(args.folds):
        tr, te = fs != k, fs == k
        sm = static_model().fit(d["static_X"][tr], d["static_y"][tr])
        pp[te] = sm.predict_proba(d["static_X"][te])
        ps[te] = pp[te].argmax(1)
        s2 = None
        if context:  # its training rows get first-stage probabilities from models that never saw their cell either
            inner = np.zeros_like(pp)
            for m in range(args.folds):
                if m != k:
                    fit = (fs != k) & (fs != m)
                    inner[fs == m] = static_model().fit(d["static_X"][fit], d["static_y"][fit]).predict_proba(
                        d["static_X"][fs == m])
            trs, tes = np.flatnonzero(tr), np.flatnonzero(te)
            s2 = static_model().fit(context_data(d, inner, trs), d["static_y"][trs])
            p2[tes] = s2.predict(context_data(d, pp, tes))
        trd = np.flatnonzero(fd != k)
        if len(trd) > args.max_dyn:
            trd = rng.choice(trd, args.max_dyn, replace=False)
        dm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
        dm.fit(d["dyn_X"][trd], d["dyn_y"][trd])
        ted = fd == k
        pd_[ted] = dm.predict(d["dyn_X"][ted])
        jm = None
        if d["static_start"] is not None:
            jm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
            jm.fit(jX[fj != k], jy[fj != k])
            pj[np.ix_(fj == k, jm.classes_)] = jm.predict_proba(jX[fj == k])
        models[k] = (sm, dm, jm, s2)
        print("fold %d: %d/%d static, %d/%d dynamic rows held out" % (k, te.sum(), len(fs), ted.sum(), len(fd)), flush=True)
    parts = [
        "%d scenes in %d spatial cells (500 m), %d folds; every prediction below is from a model that never saw "
        "its cell" % (len(scenes), len(cells), args.folds),
        _report("static (candidate lanes)", d["static_y"], ps, STATIC),
        *([_report("static, second stage (with the junction's other approaches)", d["static_y"], p2, STATIC)]
          if context else []),
        _report("dynamic (signalised lanes, every 1 s, known states)", d["dyn_y"], pd_, DYNAMIC)]
    if d["static_start"] is not None:
        parts.append(_report("junction, from the lane model's predictions", jy, junction_of_lanes(d, ps), JUNCTION))
        if context:
            parts.append(_report("junction, from the second stage's predictions", jy, junction_of_lanes(d, p2), JUNCTION))
        parts.append(_report("junction model", jy, pj.argmax(1), JUNCTION))
        stop = np.isin(jy, (1, 2))  # stop-controlled junctions: is it all-way?
        p_all, all_way = pj[stop, 2], jy[stop] == 2
        lines = ["all-way vs two-way on %d stop-controlled junctions (%d all-way): P(all-way) >= hi -> all-way, "
                 "<= lo -> two-way, between -> unknown" % (stop.sum(), all_way.sum())]
        for lo, hi in ((0.5, 0.5), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)):
            yes, no = p_all >= hi, p_all <= lo
            lines.append("  lo %.1f hi %.1f: decided %.0f%%, all-way precision %.3f recall %.3f, two-way precision %.3f" % (
                lo, hi, 100.0 * (yes | no).mean(), (all_way & yes).sum() / max(1, yes.sum()),
                (all_way & yes).sum() / max(1, all_way.sum()), (~all_way & no).sum() / max(1, no.sum())))
        parts.append("\n".join(lines))
    report = "\n\n".join(parts)
    print(report)
    with open(args.out, "wb") as f:
        pickle.dump({"models": models, "fold": fold, "names": d["names"]}, f)
    Path(args.out).with_suffix(".txt").write_text(report + "\n", encoding="utf-8")


# --- evaluate: the right-of-way screen on inferred controls ----------------------------------

def with_controls(scene, stop_lanes, lights_by_lane):
    """A copy of ``scene`` whose stop signs and traffic lights are the given ones (none at all when
    both are empty)."""
    out = copy.copy(scene)
    feats = {k: v for k, v in scene.map_features.items() if v.get("type") != "STOP_SIGN"}
    graph = lane_graph(scene)
    for lane in stop_lanes:
        feats["inferred_stop_" + lane] = {"type": "STOP_SIGN", "lane": [int(lane) if lane.isdigit() else lane],
                                          "position": np.append(graph.points[lane][0], 0.0)}
    out.map_features = feats
    out.dynamic_map_states = {}
    for lane, states in lights_by_lane.items():
        light = {"type": "TRAFFIC_LIGHT", "lane": int(lane) if lane.isdigit() else lane,
                 "state": {"object_state": states}, "stop_point": np.append(graph.points[lane][0], 0.0),
                 "metadata": {"inferred": True}}
        if states is None:  # control unknown: no stop point, or every uncontrolled approach near it turns unknown too
            light.update(state={"object_state": ["LANE_STATE_UNKNOWN"] * scene.n_steps}, metadata={"unknown": True})
            del light["stop_point"]
        out.dynamic_map_states["inferred_" + lane] = light
    return out


def infer(scene, exclude, focus, sm, dm, min_conf, jm=None, static_conf=0.0, all_way=(1.0, 1.0), cache=None,
          pool=(), s2=None):
    """Stop lanes, light states per lane, and (truth, prediction) per candidate lane. With ``jm`` the
    junction model decides all-way stops: P(all-way) >= hi makes every lane of the junction a stop
    lane, and between lo and hi its stop and none lanes are left unknown. Lanes the lane model gives
    less than ``static_conf`` are unknown too. An unknown lane gets a light whose state is unknown at
    every step, which the rules read as "control unknown": no verdict rather than a guessed one."""
    cache = {} if cache is None else cache  # scene_rows' context, shared by its calls (and by callers' modes)
    r = scene_rows(scene, exclude, focus, cache=cache, pool=pool)
    if r is None:
        return set(), {}, {}
    prob = sm.predict_proba(r["static_X"])
    if s2 is not None:  # the second stage, from the first one's probabilities across each junction
        prob = s2.predict_proba(context_rows(r["static_X"], r["names_static"], r["static_start"], prob))
    cls = prob.argmax(1)
    unsure = prob.max(1) < static_conf
    if jm is not None:
        junction, approach = junction_groups(r["static_start"])
        rows, members = junction_rows(r["static_X"], r["names_static"], junction, approach)
        pj = jm.predict_proba(rows)[:, list(jm.classes_).index(2)]
        lo, hi = all_way
        for m, p_all in zip(members, pj):
            m = m[cls[m] != 2]
            if p_all >= hi:
                cls[m], unsure[m] = 1, False
            elif p_all > lo:
                unsure[m] = True
    stop = {l for l, c, u in zip(r["lanes"], cls, unsure) if STATIC[c] == "stop" and not u}
    sig = [l for l, c, u in zip(r["lanes"], cls, unsure) if STATIC[c] == "signal" and not u]
    lights = {}
    if sig:
        steps = list(range(0, scene.n_steps, STEP_EVERY))
        rd = scene_rows(scene, exclude, focus, dyn_lanes=sig, dyn_steps=steps, cache=cache)
        pp = dm.predict_proba(rd["dyn_X"])
        at = defaultdict(dict)
        for (l, t), p in zip(rd["dyn_meta"], pp):
            at[l][t] = DYNAMIC[int(p.argmax())] if p.max() >= min_conf else None
        for l in sig:
            seq = []
            for t in range(scene.n_steps):
                s = at[l].get(min(steps, key=lambda x: abs(x - t)))
                seq.append(WOMD_STATE[s] if s else "LANE_STATE_UNKNOWN")
            lights[l] = seq
    for l in np.asarray(r["lanes"], object)[unsure]:
        lights[l] = None  # with_controls: a light of unknown state, and no stop point
    truth = dict(zip(r["lanes"], (STATIC[c] for c in r["static_y"])))
    return stop, lights, {l: (truth[l], "unknown" if u else STATIC[c])
                          for l, c, u in zip(r["lanes"], cls, unsure)}


def cmd_evaluate(args):
    sys.path.insert(0, str(SKILL))
    import jurisdiction
    import screen_crossing as SC
    from responsibility import lanes as lanes_mod
    from responsibility import right_of_way as R

    profile = jurisdiction.load_profile(args.profile)
    pairs = read_pairs(args.pairs)
    bundle = pickle.load(open(args.models, "rb"))
    rows = []
    for path in files_of(args.scenes, args.part):
        row = pairs.get(path.stem)
        if row is None or path.stem not in bundle["fold"] or (args.junction_only and row["case"] in NOT_JUNCTION):
            continue
        sm, dm, jm, s2 = (bundle["models"][bundle["fold"][path.stem]] + (None, None))[:4]
        jm = jm if args.junction_model else None
        try:
            scene, _ = load_scene(path, Scene)
            a, o, focus = pair_of(scene, row)
            on, pool = map_of(path.stem, scene, args)
            cache = {}
            for mode in args.modes.split(","):
                if mode == "no_controls":
                    stop, lights, lane_pred = (), {}, {}
                elif mode == "inferred":
                    stop, lights, lane_pred = infer(on, {a, o}, focus, sm, dm, args.min_conf, cache=cache, pool=pool,
                                                    s2=s2)
                elif mode == "gated":
                    stop, lights, lane_pred = infer(on, {a, o}, focus, sm, dm, args.min_conf, jm, args.static_conf,
                                                    (args.all_way_lo, args.all_way_hi), cache=cache, pool=pool, s2=s2)
                out = SC.screen_pair(R, lanes_mod, with_controls(on, stop, lights), a, o, profile, 10)
                out.update(scene=path.stem, mode=mode, n_stop=len(stop),
                           n_signal=sum(v is not None for v in lights.values()),
                           n_unknown=sum(v is None for v in lights.values()),
                           lane_errors=sum(t != p for t, p in lane_pred.values()), lanes=len(lane_pred))
                rows.append(out)
        except Exception as exc:
            print("skip %s: %s: %s" % (path.stem, type(exc).__name__, exc), file=sys.stderr)
    cols = ["scene", "mode", "case", "holder", "went_first", "sdc_violation", "ooi_violation", "sdc_aggressive_candidate",
            "ooi_aggressive_candidate", "sdc_control", "ooi_control", "sdc_red_entry", "ooi_red_entry", "n_stop",
            "n_signal", "n_unknown", "lanes", "lane_errors"]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    name = Path(args.out) / ("part_%s.csv" % (args.part.replace("/", "of") if args.part else "all"))
    with open(name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("%d rows -> %s" % (len(rows), name))


# full-map cases whose verdict does not depend on traffic controls
NOT_JUNCTION = ("same-direction", "following", "no-conflict", "paths-do-not-cross", "too-short", "driveway")


def cmd_compare(args):
    real = read_pairs(args.real)
    ev = defaultdict(dict)
    for p in sorted(p for folder in args.eval for p in glob.glob(str(Path(folder) / "part_*.csv"))):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ev[r["mode"]][r["scene"]] = r
    lines = []
    for mode, where in ((m, w) for w in ("all", "junction") for m in ("gated", "inferred", "no_controls")):
        got = ev.get(mode, {})
        common = [s for s in got if s in real and (where == "all" or real[s]["case"] not in NOT_JUNCTION)]
        if not common:
            continue
        same = lambda k: sum(real[s][k] == got[s][k] for s in common)
        lines.append("## %s controls, %s scenes: %d" % (mode, where, len(common)))
        for k in ("case", "holder", "sdc_control", "ooi_control"):
            lines.append("- %s unchanged: %d (%.1f%%)" % (k, same(k), 100.0 * same(k) / len(common)))
        for k in ("sdc_violation", "ooi_violation", "sdc_aggressive_candidate", "ooi_aggressive_candidate"):
            r1 = {s for s in common if real[s][k] == "1"}
            g1 = {s for s in common if got[s][k] == "1"}
            lines.append("- %s: full map %d, here %d, both %d (kept %.0f%%, new %d, precision %.0f%%)" % (
                k, len(r1), len(g1), len(r1 & g1), 100.0 * len(r1 & g1) / max(1, len(r1)), len(g1 - r1),
                100.0 * len(r1 & g1) / max(1, len(g1))))
        held = [s for s in common if real[s]["holder"] in ("sdc", "ooi")]
        kept = sum(real[s]["holder"] == got[s]["holder"] for s in held)
        none = sum(got[s]["holder"] not in ("sdc", "ooi") for s in held)
        lines.append("- where the full map names a holder (%d): same %d, the other one %d, none %d" % (
            len(held), kept, len(held) - kept - none, none))
        moves = Counter((real[s]["case"], got[s]["case"]) for s in common if real[s]["case"] != got[s]["case"])
        lines.append("- commonest case changes: " + "; ".join("%s -> %s: %d" % (a, b, n) for (a, b), n in moves.most_common(8)))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--scenes", required=True)
    e.add_argument("--pairs", required=True, help="screen_crossing.csv: the judged pair of each scene")
    e.add_argument("--out", required=True)
    e.add_argument("--part")
    e.add_argument("--map", choices=("full", "rebuilt"), default="full",
                   help="rebuilt: the perception-only map with the topology rebuilt from the tracks")
    e.add_argument("--pool", help="pool.py folder: add the tracks of other scenes at the same place")
    t = sub.add_parser("train")
    t.add_argument("--features", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--max-dyn", type=int, default=400000, help="dynamic rows per fold to train on")
    t.add_argument("--no-context", dest="context", action="store_false", help="no second-stage lane model")
    v = sub.add_parser("evaluate")
    v.add_argument("--scenes", required=True)
    v.add_argument("--pairs", required=True)
    v.add_argument("--models", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--part")
    v.add_argument("--map", choices=("full", "rebuilt"), default="full",
                   help="rebuilt: the perception-only map with the topology rebuilt from the tracks")
    v.add_argument("--pool", help="pool.py folder: add the tracks of other scenes at the same place")
    v.add_argument("--profile", default="us-california")
    v.add_argument("--min-conf", type=float, default=0.6, help="below this a light state stays unknown")
    v.add_argument("--modes", default="inferred,no_controls", help="comma list of inferred, gated, no_controls")
    v.add_argument("--static-conf", type=float, default=0.6, help="gated: below this a lane's control is unknown")
    v.add_argument("--all-way-lo", type=float, default=0.2, help="gated: P(all-way) at or below this: two-way")
    v.add_argument("--all-way-hi", type=float, default=0.8, help="gated: P(all-way) at or above this: all-way")
    v.add_argument("--junction-model", action="store_true", help="gated: let the junction model decide all-way stops")
    v.add_argument("--junction-only", action="store_true", help="skip pairs whose full-map case needs no controls")
    c = sub.add_parser("compare")
    c.add_argument("--real", required=True)
    c.add_argument("--eval", required=True, nargs="+")
    c.add_argument("--out")
    g = sub.add_parser("geometry", help="add lane starts to features extracted without them")
    g.add_argument("--scenes", required=True)
    g.add_argument("--features", required=True)
    g.add_argument("--part", help="k/n: only part_kofn.pkl")
    args = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train, "evaluate": cmd_evaluate, "compare": cmd_compare,
     "geometry": cmd_geometry}[args.cmd](args)


if __name__ == "__main__":
    main()
