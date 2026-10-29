"""Who caused a pair's conflict, read from the two tracks and the map, without a motion model: the
kinematic half (A1) of the attribution in aggression_generalisation_plan.md. The counterfactual half
(A2) is the two-sided safety score of aggression_test.pair_scores (rel = beta_s(a -> b) - beta_s(b ->
a)). Safety responsibility alone does not attribute: the closest approach is shared, so any
alternative of either agent widens it (README, "Can it judge aggression?"). The two halves fail for
different reasons (A1 on map geometry, A2 on the model's symmetry and inertia), which is why they are
checked against each other.

Three kinds of conflict, each with its own reading of "who caused it":

  crossing   the paths cross or merge and the right of way (right_of_way.py, judged at the first
             crossing of the whole tracks as forced_yield.pair_priority does) names a holder. The
             actor is the yielder if it entered its conflict zone first; nobody if the holder did.
             Without a holder the actor is unknown: who moved does not decide a crossing.
  join       the two travel the same way side by side and then one is in the other's path ahead of
             it: one or both moved over. Each agent's sideways movement is the integral of its
             speed times the sine of its course against the road's local direction at its own
             position: the median direction of the lane centrelines and road edges within
             ROAD_RADIUS that run its way. Not its own lane: a car following a lane that branches
             off across the road (CAT scenes 293, 337, 483) does not move against that lane but does
             against the road, and the median over every nearby lane and edge is not swayed by one
             branch. The road's curvature drops out because the direction is read where the car is.
             The window runs from SIDE_PAD before the last step the two were side by side to
             JOIN_AFTER after the join, so an earlier, separate lane change (CAT scene 342: the
             other car changed lanes behind first) is not mixed in. The actor is the agent with at
             least ACTOR_SHARE of the sideways closing; both when neither has; none when the two
             closed in by less than MIN_CLOSING; unknown when the map gives no direction for most of
             the window.
  following  one is in the other's path throughout, behind it: the follower chose its distance.

Checked on 12 CAT scenes whose actor was known from the review of the driving-aggression-judge
skill (scripts/responsibility/attribute_pairs.py --check): see responsibility/README.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from responsibility.forced_yield import pair_priority
from responsibility.right_of_way import RightOfWayParams, _Path, _signals, _window
from responsibility.scene import Scene

DT = 0.1  # s per step
SAME_WAY = np.deg2rad(30.0)  # rad: headings closer than this travel the same way
NEAR = 40.0  # m: longitudinal reach of a join or of following
SIDE_MARGIN = 0.5  # m beyond half the two widths: side by side, not in path
SIDE_LOOKBACK = 30  # steps (3 s): the two must have been side by side this soon before the join
SIDE_PAD = 10  # steps (1 s) of the window before the last side-by-side step
JOIN_AFTER = 10  # steps (1 s) of the window after the join
MIN_CLOSING = 1.0  # m: the two must have closed in sideways by at least this much
ACTOR_SHARE = 0.7  # share of the sideways closing that makes one agent the actor
ROAD_RADIUS = 6.0  # m: lane centrelines and road edges this close give the road's direction
ROAD_ALIGN = np.deg2rad(30.0)  # rad: a segment runs the car's way (either direction) within this
MIN_COVER = 0.8  # share of a window's moving steps that must have a road direction
MOVING = 1.0  # m/s: slower steps are left out of the sideways movement
EDGE_TYPES = ("ROAD_EDGE_BOUNDARY", "ROAD_EDGE_MEDIAN")
LANE_TYPES = ("LANE_FREEWAY", "LANE_SURFACE_STREET", "LANE_UNDEFINED")


@dataclass
class Attribution:
    kind: str  # "crossing", "join", "following" or "none"
    actor: str  # "agent", "other", "both", "none" or "unknown" (the judged pair is (agent, other))
    rule: str  # how the actor was decided, or why it is unknown
    step: Optional[int] = None  # the join's first in-path step; the actor's zone entry in a crossing
    evidence: Dict = field(default_factory=dict)

    def as_row(self) -> Dict:
        return {"kind": self.kind, "actor": self.actor, "rule": self.rule,
                "step": "" if self.step is None else self.step, **self.evidence}


# --- geometry --------------------------------------------------------------------------------

def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rel(scene: Scene, i: int, j: int, k) -> Tuple[np.ndarray, np.ndarray]:
    """j's centre in i's frame at step(s) k: (ahead, left) in metres."""
    k = np.atleast_1d(k)
    d = scene.position[j, k, :2].astype(float) - scene.position[i, k, :2].astype(float)
    h = scene.heading[i, k].astype(float)
    return d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h), -d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)


def same_way(scene: Scene, i: int, j: int, k) -> np.ndarray:
    return np.abs(_wrap(scene.heading[i, k].astype(float) - scene.heading[j, k].astype(float))) < SAME_WAY


def map_segments(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    """The segments of the lane centrelines and road edges: starts [M, 2], vectors [M, 2]."""
    starts, vectors = [], []
    for f in scene.map_features.values():
        poly = np.asarray(f.get("polyline", []), dtype=float)
        if poly.ndim == 2 and len(poly) >= 2 and (f.get("type") in EDGE_TYPES or f.get("type") in LANE_TYPES):
            starts.append(poly[:-1, :2])
            vectors.append(np.diff(poly[:, :2], axis=0))
    if not starts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    starts, vectors = np.concatenate(starts), np.concatenate(vectors)
    keep = np.linalg.norm(vectors, axis=1) > 1e-3
    return starts[keep], vectors[keep]


def road_direction(segments, points: np.ndarray, courses: np.ndarray) -> np.ndarray:
    """Per point, the road's direction: the course plus the median deviation of the segments within
    ROAD_RADIUS that run within ROAD_ALIGN of it (either direction); nan where there is none."""
    starts, vectors = segments
    out = np.full(len(points), np.nan)
    if len(starts) == 0:
        return out
    angles, length2 = np.arctan2(vectors[:, 1], vectors[:, 0]), (vectors ** 2).sum(1)
    for i, (p, c) in enumerate(zip(points, courses)):
        u = np.clip(((p - starts) * vectors).sum(1) / length2, 0.0, 1.0)
        dist = np.linalg.norm(starts + u[:, None] * vectors - p, axis=1)
        dev = _wrap(angles - c)
        dev = np.where(np.abs(dev) > np.pi / 2, _wrap(dev + np.pi), dev)  # a segment drawn the other way
        ok = (dist <= ROAD_RADIUS) & (np.abs(dev) <= ROAD_ALIGN)
        if ok.any():
            out[i] = c + np.median(dev[ok])
    return out


def sideways(scene: Scene, i: int, steps: np.ndarray, segments) -> Tuple[float, float]:
    """How far agent i moved to its left against the road over ``steps`` (m), and the share of its
    moving steps that had a road direction."""
    pos = scene.position[i, steps, :2].astype(float)
    step = np.diff(pos, axis=0)
    length = np.linalg.norm(step, axis=1)
    course = np.arctan2(step[:, 1], step[:, 0])
    road = road_direction(segments, (pos[1:] + pos[:-1]) / 2, course)
    moving = length / (np.diff(steps) * DT) > MOVING
    known = moving & np.isfinite(road)
    left = np.where(known, length * np.sin(_wrap(course - np.where(known, road, 0.0))), 0.0)
    return float(left.sum()), float(known.sum() / max(moving.sum(), 1))


# --- joins and following ---------------------------------------------------------------------

def in_path(scene: Scene, follower: int, leader: int, k) -> np.ndarray:
    """Whether ``leader`` is ahead of ``follower`` within NEAR, its centre within half the two
    widths of the follower's centreline, travelling the same way."""
    k = np.atleast_1d(k)
    ahead, left = rel(scene, follower, leader, k)
    half = (scene.width[follower, k] + scene.width[leader, k]) / 2
    return (ahead > 0) & (ahead < NEAR) & (np.abs(left) < half) & same_way(scene, follower, leader, k)


def side_by_side(scene: Scene, i: int, j: int, k) -> np.ndarray:
    k = np.atleast_1d(k)
    ahead, left = rel(scene, i, j, k)
    half = (scene.width[i, k] + scene.width[j, k]) / 2
    return (np.abs(ahead) < NEAR) & (np.abs(left) >= half + SIDE_MARGIN) & same_way(scene, i, j, k)


def joins(scene: Scene, merger: int, other: int) -> List[int]:
    """Steps at which ``merger`` came into ``other``'s path ahead of it, having been side by side
    with it within SIDE_LOOKBACK steps before."""
    both = np.flatnonzero(scene.valid[merger] & scene.valid[other])
    if both.size < 2:
        return []
    inside = np.zeros(scene.n_steps, dtype=bool)
    beside = np.zeros(scene.n_steps, dtype=bool)
    inside[both] = in_path(scene, other, merger, both)
    beside[both] = side_by_side(scene, other, merger, both)
    return [int(k) for k in both
            if inside[k] and (k == 0 or not inside[k - 1]) and beside[max(0, k - SIDE_LOOKBACK):k].any()]


def join_window(scene: Scene, a: int, b: int, k: int) -> np.ndarray:
    """The steps both are seen from SIDE_PAD before the last side-by-side step before join k to
    JOIN_AFTER after it."""
    both = np.flatnonzero(scene.valid[a] & scene.valid[b])
    before = both[(both < k) & (both >= k - SIDE_LOOKBACK)]
    beside = before[side_by_side(scene, a, b, before)] if before.size else before
    start = int(beside[-1]) - SIDE_PAD if beside.size else k - SIDE_LOOKBACK
    return both[(both >= start) & (both <= k + JOIN_AFTER)]


def join_actor(scene: Scene, a: int, b: int, k: int, segments) -> Tuple[str, Dict]:
    """("a" / "b" / "both" / "none" / "unknown", evidence) for the join at step k."""
    steps = join_window(scene, a, b, k)
    if steps.size < 3:
        return "unknown", {"window": int(steps.size)}
    la, cover_a = sideways(scene, a, steps, segments)
    lb, cover_b = sideways(scene, b, steps, segments)
    side = np.sign(rel(scene, a, b, steps[0])[1][0]) or 1.0  # +1: b on a's left at the start
    toward_a, toward_b = la * side, -lb * side
    ev = {"window_start": int(steps[0]), "window_end": int(steps[-1]), "toward_agent": round(float(toward_a), 2),
          "toward_other": round(float(toward_b), 2), "road_cover": round(float(min(cover_a, cover_b)), 2)}
    if min(cover_a, cover_b) < MIN_COVER:
        return "unknown", ev
    closing = max(toward_a, 0.0) + max(toward_b, 0.0)
    if closing < MIN_CLOSING:
        return "none", ev
    share = max(toward_a, 0.0) / closing
    ev["share_agent"] = round(float(share), 2)
    return ("a" if share >= ACTOR_SHARE else "b" if share <= 1 - ACTOR_SHARE else "both"), ev


# --- the attribution -------------------------------------------------------------------------

def _first_entry(in_zone: np.ndarray) -> Optional[int]:
    inside = np.flatnonzero(in_zone > 0)
    return None if not inside.size or in_zone[0] > 0 else int(inside[0])


def crossing(scene: Scene, agent: int, other: int, params: Optional[RightOfWayParams] = None) -> Optional[Attribution]:
    """The crossing reading, or None when the pair's paths do not cross or merge."""
    p = params or RightOfWayParams()
    prio = pair_priority(scene, agent, other, params=p)
    if prio is None or prio.conflict is None or prio.approaches is None or prio.case in ("same-direction", "following"):
        return None
    if prio.holder is None:
        return Attribution("crossing", "unknown", "no-holder", evidence={"case": prio.case, "holder": "none"})
    steps = _window(scene, agent, other, scene.n_steps - 1)
    if steps.size < 2:
        return Attribution("crossing", "unknown", "too-short", evidence={"case": prio.case})
    c = prio.conflict
    entry = {}
    for x, o, s_star in ((agent, other, c.s[0]), (other, agent, c.s[1])):
        in_zone, _, _ = _signals(scene, x, o, _Path(scene, x, int(steps[-1]), p.extend), s_star, c.angle, steps, p)
        k = _first_entry(in_zone)
        entry[x] = None if k is None else int(steps[k])
    holder = "agent" if prio.holder == agent else "other"
    ea, eo = entry[agent], entry[other]
    ev = {"case": prio.case, "holder": holder, "entry_agent": "" if ea is None else ea,
          "entry_other": "" if eo is None else eo}
    if ea is None and eo is None:
        return Attribution("crossing", "unknown", "neither-entered", evidence=ev)
    first = "agent" if eo is None or (ea is not None and ea < eo) else "other" if ea is None or eo < ea else "tie"
    ev["first"] = first
    if first == "tie":
        return Attribution("crossing", "unknown", "simultaneous", ea, ev)
    k = ea if first == "agent" else eo
    if first == holder:
        return Attribution("crossing", "none", "holder-first", k, ev)
    return Attribution("crossing", first, "yielder-first", k, ev)


def attribute(scene: Scene, agent: int, other: int, params: Optional[RightOfWayParams] = None,
              segments=None) -> Attribution:
    """Who caused the pair's conflict (module docstring), as "agent" / "other" / "both" / "none" /
    "unknown" for the pair (agent, other)."""
    both = np.flatnonzero(scene.valid[agent] & scene.valid[other])
    if both.size < 2:
        return Attribution("none", "unknown", "not-seen-together")
    events = [(k, agent) for k in joins(scene, agent, other)] + [(k, other) for k in joins(scene, other, agent)]
    if events:
        k, merger = min(events)
        segments = map_segments(scene) if segments is None else segments
        actor, ev = join_actor(scene, agent, other, k, segments)
        ev.update(ahead="agent" if merger == agent else "other", joins=len(events))
        name = {"a": "agent", "b": "other"}.get(actor, actor)
        rule = {"agent": "moved-over", "other": "moved-over", "both": "both-moved-over",
                "none": "neither-moved-over"}.get(name, "no-road-direction")
        return Attribution("join", name, rule, int(k), ev)
    found = crossing(scene, agent, other, params)
    if found is not None:
        return found
    follow_ao, follow_oa = in_path(scene, agent, other, both), in_path(scene, other, agent, both)
    if follow_ao.mean() > 0.5 or follow_oa.mean() > 0.5:
        follower = "agent" if follow_ao.sum() >= follow_oa.sum() else "other"
        return Attribution("following", follower, "follower",
                           evidence={"in_path_share": round(float(max(follow_ao.mean(), follow_oa.mean())), 2)})
    return Attribution("none", "none", "no-conflict")
