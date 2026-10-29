"""Aggression as duty x cost x attribution, with thresholds taken from logged driving
(aggression_generalisation_plan.md): an agent is aggressive toward another when it caused a conflict
in which it owed the other something (to yield, to move over only with reasonable safety, not to
follow too closely), and the braking that conflict asked of the other is above what logged drivers
in the same situation ask (the --quantile, 0.9 by default, of a natural reference).

  crossing  attribution.crossing names the agent as the yielder that entered first; the cost is
            forced_yield.forced (the braking its entry demanded of the holder). Reference: logged
            yielders that went first (make_go_first.py's log_yielder_first).
  cut-in    attribution names the agent as the car that moved over into the other's path ahead of
            it; the cost is forced_merge.cut_in. Reference: every logged join made by the car that
            ended up ahead.
  tailgate  the agent followed the other by its own choice; the cost is forced_merge.tailgate.
            Reference: every logged following episode.

Only the legal content is written by hand (who owes what, the module docstrings of right_of_way.py
and forced_merge.py); every threshold is the reference's quantile on the data at hand, so a new
dataset, a new region or another motion model gets its own. A verdict is "abstain" when a conflict
exists but who caused it cannot be read (a crossing without a holder or with a simultaneous entry,
a join where the map gives no road direction) and no other duty was broken.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from responsibility import attribution as attr
from responsibility.forced_merge import same_direction
from responsibility.forced_yield import forced_yield
from responsibility.right_of_way import RightOfWayParams
from responsibility.scene import Scene

KINDS = ("crossing", "cut-in", "tailgate")
NEAR = 40.0  # m: pairs that never come this close are not scanned
MOVING = 2.0  # m/s: agents never this fast are not scanned


@dataclass
class Judgement:
    verdict: str  # "aggressive", "not aggressive" or "abstain"
    kind: str  # the kind that decided: "crossing", "cut-in", "tailgate" or "none"
    reason: str
    costs: Dict = field(default_factory=dict)  # kind -> demand (m/s^2) where the duty applied
    quantiles: Dict = field(default_factory=dict)  # kind -> the demand's quantile in its reference
    evidence: Dict = field(default_factory=dict)

    def as_row(self) -> Dict:
        row = {"verdict": self.verdict, "kind": self.kind, "reason": self.reason}
        for k in KINDS:
            c = self.costs.get(k)
            row[f"{k}_demand"] = "" if c is None else ("inf" if np.isinf(c) else round(float(c), 3))
            q = self.quantiles.get(k)
            row[f"{k}_quantile"] = "" if q is None else round(float(q), 3)
        return {**row, **self.evidence}


def costs(scene: Scene, agent: int, other: int, params: Optional[RightOfWayParams] = None, segments=None) -> Dict:
    """The agent's demands on the other under each duty it owed, with what decided them:
    {"crossing": (demand or None, attribution), "cut-in": ForcedMerge, "tailgate": ForcedMerge}."""
    p = params or RightOfWayParams()
    segments = attr.map_segments(scene) if segments is None else segments
    out = {}
    cross = attr.crossing(scene, agent, other, p)
    demand = None
    if cross is not None and cross.actor == "agent":
        demand = forced_yield(scene, agent, other, params=p).forced
    out["crossing"] = (demand, cross)
    out.update(same_direction(scene, agent, other, p, segments))
    return out


def quantile_in(value: float, reference: np.ndarray) -> float:
    """Share of the reference at or below ``value`` (inf above everything finite)."""
    reference = np.asarray(reference, dtype=float)
    if reference.size == 0:
        return float("nan")
    return float(np.mean(reference <= value))


def judge(scene: Scene, agent: int, other: int, calibration: Dict, params: Optional[RightOfWayParams] = None,
          segments=None) -> Judgement:
    """The verdict on ``agent`` toward ``other`` against ``calibration`` (calibrate_aggression.py:
    per kind its "threshold" and its "reference" values)."""
    segments = attr.map_segments(scene) if segments is None else segments
    c = costs(scene, agent, other, params, segments)
    demand, cross = c["crossing"]
    found = {}
    ev = {"crossing_case": "" if cross is None else cross.evidence.get("case", ""),
          "crossing_actor": "" if cross is None else cross.actor}
    if demand is not None:
        found["crossing"] = demand
    for kind in ("cut-in", "tailgate"):
        m = c[kind]
        if m.demand is not None:
            found[kind] = m.demand
            for key in ("headway_at_join", "headway", "speed_difference", "join"):
                if key in m.evidence:
                    ev[f"{kind}_{key}"] = m.evidence[key]
    quantiles = {k: quantile_in(v, calibration[k]["reference"]) for k, v in found.items() if k in calibration}
    over = {k: v for k, v in found.items() if k in calibration and v > calibration[k]["threshold"]}
    if over:
        kind = max(over, key=lambda k: quantiles[k])
        return Judgement("aggressive", kind, f"{kind} demand {over[kind]:.2f} m/s^2 above the "
                         f"q{calibration[kind]['quantile']:g} threshold {calibration[kind]['threshold']:.2f}",
                         found, quantiles, ev)
    if cross is not None and cross.actor == "unknown" and not found:
        return Judgement("abstain", "crossing", "crossing without a readable cause: " + cross.rule, found, quantiles, ev)
    a = attr.attribute(scene, agent, other, params, segments)
    if a.kind == "join" and a.actor == "unknown" and not found:
        return Judgement("abstain", "cut-in", "join without a road direction to tell who moved", found, quantiles, ev)
    return Judgement("not aggressive", "none", "no demand above its threshold" if found else "owed nothing",
                     found, quantiles, ev)


# --- natural references ----------------------------------------------------------------------

def candidate_pairs(scene: Scene) -> List[tuple]:
    """Unordered pairs of moving vehicles that come within NEAR of each other."""
    veh = [i for i in range(scene.n_agents) if scene.types[i] == "VEHICLE" and scene.valid[i].sum() >= 20
           and np.linalg.norm(scene.velocity[i, scene.valid[i]], axis=-1).max() >= MOVING]
    out = []
    for n, i in enumerate(veh):
        for j in veh[n + 1:]:
            both = scene.valid[i] & scene.valid[j]
            if both.sum() < 10:
                continue
            d = np.linalg.norm(scene.position[i, both, :2] - scene.position[j, both, :2], axis=-1)
            if d.min() <= NEAR:
                out.append((i, j))
    return out


def same_direction_references(scene: Scene, params: Optional[RightOfWayParams] = None) -> List[Dict]:
    """Every logged cut-in (a join made by the car that came in ahead) and following episode between
    any two moving vehicles of the scene: rows with the pair, the kind and the demand."""
    segments = attr.map_segments(scene)
    rows = []
    for i, j in candidate_pairs(scene):
        both = np.flatnonzero(scene.valid[i] & scene.valid[j])
        if not (attr.in_path(scene, i, j, both).any() or attr.in_path(scene, j, i, both).any()):
            continue
        for x, y in ((i, j), (j, i)):
            r = same_direction(scene, x, y, params, segments)
            for kind in ("cut-in", "tailgate"):
                m = r[kind]
                if m.demand is not None:
                    rows.append({"agent": scene.track_ids[x], "other": scene.track_ids[y], "kind": kind,
                                 "demand": m.demand, "step": m.step,
                                 "is_sdc": int(x == scene.sdc or y == scene.sdc)})
    return rows
