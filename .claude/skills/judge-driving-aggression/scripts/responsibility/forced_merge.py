"""How hard one car's closeness in the same lane would force the car behind it to brake: the
same-direction counterpart of forced_yield.py, for cut-ins and tailgating, open loop on the tracks.

forced_yield.py measures the braking an agent's entry into a crossing demands of the holder of the
right of way. Between cars in one lane there is no conflict zone and no holder (right_of_way.py
leaves same-direction pairs to RSS), so the duty is the one the law states for these cases: a car
may move over only when it can do so with reasonable safety (CVC 21658(a), 22107, as recalled), and
may not follow more closely than is reasonable and prudent (CVC 21703). The cost is read the same
way as forced_yield's a_req: the constant deceleration the rear car would need to stop behind the
front car if the front car braked at ``brake`` (RightOfWayParams.brake, RSS's brake_min, 4 m/s^2)
and the rear car responded after ``rho`` (1 s):

    a_req = v_r^2 / (2 (gap + v_f^2 / (2 brake) - v_r rho))                          [m/s^2]

gap being the box-to-box distance along the rear car's heading. It is infinite when the gap is
gone within the response time. At equal speeds a_req equals ``brake`` at a headway of rho: the rear
car can match the front car's braking exactly; a shorter headway asks more of it than the front car
does. A slower front car asks more, a faster one less, so a car that cuts in while pulling away asks
little (the rubric's pull-away exemption) unless it leaves almost no gap (its 0.5 s floor).

  cut-in    a join (attribution.joins) in which the car that came in ahead is the one that moved
            over (attribution.attribute: the actor). ``demand``: the worst a_req of the car behind,
            over JOIN_EFFECT steps from the join.
  tailgate  the car behind, by its own choice: ``demand`` is the highest a_req it kept for
            SUSTAIN steps (1 s; WOMD speeds jitter by 1-2 m/s between frames) at MIN_SPEED or more
            with the boxes apart, excluding the EXCUSED steps after the car ahead moved in in front
            of it.

As with forced_yield, the thresholds are calibrated on logged driving (calibrate_aggression.py),
not fixed here.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from responsibility import attribution as attr
from responsibility.right_of_way import RightOfWayParams
from responsibility.scene import Scene

JOIN_EFFECT = 10  # steps (1 s, the response time) after a join over which the cut-in's demand is read
SUSTAIN = 10  # steps (1 s) a tailgating demand must be kept
EXCUSED = 30  # steps (3 s) after the car ahead moved in in front, not the follower's doing
MIN_SPEED = 5.0  # m/s: a slower rear car is queueing, not tailgating (the rubric's 5 m/s; slow queues are not
                 # aggression, and there the response time alone makes every short gap look demanding)


@dataclass
class ForcedMerge:
    kind: str  # "cut-in", "tailgate" or "none"
    demand: Optional[float] = None  # m/s^2 (inf: the gap is gone within the response time)
    step: Optional[int] = None  # where the demand was read
    evidence: Dict = field(default_factory=dict)

    def as_row(self) -> Dict:
        def num(x):
            return "" if x is None else "inf" if np.isinf(x) else round(float(x), 3)
        return {"kind": self.kind, "demand": num(self.demand), "step": "" if self.step is None else self.step,
                **self.evidence}


def following_demand(scene: Scene, rear: int, front: int, steps: np.ndarray, params: Optional[RightOfWayParams] = None,
                     front_brake: Optional[float] = None):
    """Per step, a_req of ``rear`` behind ``front`` (module docstring); nan where ``front`` is not in
    ``rear``'s path ahead. Also the gap (m) and the headway (s). ``front_brake``: the front car's
    braking assumed (default ``params.brake``; sensitivity_cut_in.py varies it)."""
    p = params or RightOfWayParams()
    brake = p.brake if front_brake is None else front_brake
    steps = np.atleast_1d(steps)
    ahead, _ = attr.rel(scene, rear, front, steps)
    gap = ahead - (scene.length[rear, steps] + scene.length[front, steps]) / 2
    v_r = np.linalg.norm(scene.velocity[rear, steps], axis=-1).astype(float)
    v_f = np.linalg.norm(scene.velocity[front, steps], axis=-1).astype(float)
    room = gap + v_f ** 2 / (2 * brake) - v_r * p.rho
    with np.errstate(divide="ignore", invalid="ignore"):
        demand = np.where(room > 0, v_r ** 2 / (2 * np.where(room > 0, room, 1.0)), np.inf)
    demand = np.where(attr.in_path(scene, rear, front, steps), demand, np.nan)
    headway = np.where(v_r > 0.5, gap / np.maximum(v_r, 0.5), np.nan)
    return demand, gap, headway


def cut_in(scene: Scene, merger: int, rear: int, join: int, params: Optional[RightOfWayParams] = None,
           front_brake: Optional[float] = None) -> ForcedMerge:
    """The demand on ``rear`` of ``merger``'s join at step ``join``."""
    both = np.flatnonzero(scene.valid[merger] & scene.valid[rear])
    steps = both[(both >= join) & (both <= join + JOIN_EFFECT)]
    if not steps.size:
        return ForcedMerge("cut-in")
    demand, gap, headway = following_demand(scene, rear, merger, steps, params, front_brake)
    if not np.isfinite(demand).any() and not np.isinf(demand).any():
        return ForcedMerge("cut-in", evidence={"gap_at_join": round(float(gap[0]), 2)})
    worst = int(np.nanargmax(np.where(np.isnan(demand), -1.0, demand)))
    v_r = float(np.linalg.norm(scene.velocity[rear, steps[0]]))
    v_m = float(np.linalg.norm(scene.velocity[merger, steps[0]]))
    return ForcedMerge("cut-in", float(demand[worst]), int(steps[worst]),
                       {"gap_at_join": round(float(gap[0]), 2), "headway_at_join": round(float(headway[0]), 2),
                        "min_headway": round(float(np.nanmin(headway)), 2) if np.isfinite(headway).any() else "",
                        "speed_rear": round(v_r, 1), "speed_difference": round(v_m - v_r, 1)})


def tailgate(scene: Scene, rear: int, front: int, excused_from=(), params: Optional[RightOfWayParams] = None) -> ForcedMerge:
    """The highest a_req ``rear`` kept behind ``front`` for SUSTAIN steps, outside the EXCUSED steps
    after each step in ``excused_from`` (joins in which ``front`` moved in ahead of it)."""
    both = np.flatnonzero(scene.valid[rear] & scene.valid[front])
    if both.size < SUSTAIN:
        return ForcedMerge("tailgate")
    demand, gap, headway = following_demand(scene, rear, front, both, params)
    speed = np.linalg.norm(scene.velocity[rear, both], axis=-1)
    usable = np.isfinite(demand) | np.isinf(demand)
    usable &= (speed >= MIN_SPEED) & (gap > 0)  # boxes that touch while following: side by side, or a tracking glitch
    for k in excused_from:
        usable &= ~((both >= k) & (both < k + EXCUSED))
    d = np.where(usable, np.where(np.isinf(demand), 1e6, demand), np.nan)
    best, at = None, None
    for i in range(len(both) - SUSTAIN + 1):
        w = d[i:i + SUSTAIN]
        if np.isnan(w).any() or both[i + SUSTAIN - 1] - both[i] != SUSTAIN - 1:
            continue
        kept = float(w.min())
        if best is None or kept > best:
            best, at = kept, i
    if best is None:
        return ForcedMerge("tailgate", evidence={"following_share": round(float(usable.mean()), 2)})
    seg = slice(at, at + SUSTAIN)
    return ForcedMerge("tailgate", np.inf if best >= 1e6 else best, int(both[at]),
                       {"headway": round(float(np.nanmax(headway[seg])), 2), "gap": round(float(np.max(gap[seg])), 2),
                        "speed_rear": round(float(speed[seg].min()), 1),
                        "following_share": round(float(usable.mean()), 2)})


def same_direction(scene: Scene, agent: int, other: int, params: Optional[RightOfWayParams] = None,
                   segments=None) -> Dict[str, ForcedMerge]:
    """The agent's same-direction demands on ``other``: {"cut-in": the worst of its joins in front
    of other that it made itself, "tailgate": behind other}; with the attribution of each join."""
    segments = attr.map_segments(scene) if segments is None else segments
    out = {"cut-in": ForcedMerge("cut-in"), "tailgate": ForcedMerge("tailgate")}
    worst = None
    for k in attr.joins(scene, agent, other):  # agent came in ahead of other
        actor, ev = attr.join_actor(scene, agent, other, k, segments)
        if actor != "a":
            continue
        m = cut_in(scene, agent, other, k, params)
        m.evidence.update(join=k, toward_agent=ev.get("toward_agent"), toward_other=ev.get("toward_other"))
        if m.demand is not None and (worst is None or m.demand > worst.demand):
            worst = m
    if worst is not None:
        out["cut-in"] = worst
    excused = []
    for k in attr.joins(scene, other, agent):  # other came in ahead of agent
        actor, _ = attr.join_actor(scene, other, agent, k, segments)
        if actor in ("a", "both", "unknown"):  # it moved over: the short gap that follows is its doing
            excused.append(k)
    out["tailgate"] = tailgate(scene, agent, other, excused, params)
    return out
