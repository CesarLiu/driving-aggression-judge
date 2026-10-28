"""Who had the right of way where two road users' paths met, and who failed
the duty it gave them: the traffic-law baseline for conflicts between
different paths (crossing, turning across, merging, changing lanes), which
RSS (responsibility/rss.py) and the rear-end rule (responsibility/blame.py)
leave undecided. The rules are those of the place the data comes from:
WOMD was recorded in US cities, and the California Vehicle Code (CVC)
stands in for US law.

Everything is read from a ``Scene``: the two trajectories, the lane graph
(responsibility/lanes.py), the traffic-light states and stop signs, so
logged scenes and scenes rebuilt from policy rollouts are judged alike.

Conflict. Each agent's path is its centre's logged (or simulated) track up
to the step judged (``at``: the collision), extended EXTEND metres straight
on along its last heading, because a collision ends a track before its
centre reaches the other's path. The conflict point P* is where the two
paths cross (the crossing nearest to the two agents at ``at``); paths that
converge without crossing (a merge, a completed lane change) meet where
one first comes within ``merge_gap`` of the other; paths that do neither
meet at their closest approach. The conflict zone of agent x is the stretch
of its own path within

    h_x = L_x / 2 + (W_o + W_x |cos(theta)|) / (2 sin(theta))       [m]

of P*, where its footprint (length L_x, width W_x) overlaps the other's
swept strip (width W_o) crossing its path at the angle theta, with
sin(theta) held at 0.5 or more so that shallow merges stay finite.

Priority. In WOMD a traffic light and a stop sign control the intersection
lane that starts at the stop line (a light's stop point is that lane's
first point; a stop sign stands ~5 m to its side). Each agent is followed
along the lane graph from CONTROL_REACH metres before P* (lane_sequence:
it keeps to its lane, then its exit lanes, and of two branches leaving one
stop line the one it took). Its control is the light of the controlled lane
it entered last, in the state it showed when the agent crossed that stop
line (CVC 21451-21453: the light at the stop line counts; entering on
yellow is lawful), or of the controlled exit lane ahead of it that passes
P*, in its last seen state, when it has not crossed the line yet; else a
stop sign on such a lane (or a sign the map gives no lanes for, at the
right kerb of its approach); else "unknown" when lights stand within
SIGNAL_RADIUS of P* (a signalised intersection whose light for this
movement was not observed); else "none". The holder of the right of way is
decided by the first rule that applies:

    following         the two were on one path from the start (one behind
                      the other): rear-end territory, no priority
    same-direction    approach headings within SAME_DIRECTION (one road):
                      no priority and no verdict; RSS (responsibility/rss.py)
                      judges these (cut-in, side-swipe). With
                      ``same_direction_rules`` they get a holder instead:
      lane-change       one moved into the other's lane from a neighbouring
                        lane: it yields (21658(a))
      merge             paths converging from different lanes: the first
                        on the lane they share goes first
                      They are off by default: on held-out WOMD pairs the
                      lane-change holder passed first in only 38% of 85, and
                      the yield duty, whose conflict zone assumes crossing
                      paths, failed for 56% of logged lane-change yielders
                      and 52% of merging ones (below)
    driveway          one comes from off the mapped lanes (a driveway, a car
                      park): it yields to the one on the road (21804(a))
    oncoming          both go straight from opposite directions: passing
                      each other, no priority
    signal-unknown    either control is unknown: no priority
    red-light         one entered on red (or a red arrow), the other on
                      green, yellow or a green arrow: red yields (21453)
    both on green / yellow / arrow:
      protected-arrow   a green arrow over a circular light
      left-turn         one turns left (or makes a U-turn) across the other
                        coming from the opposite direction: the turner yields
                        (21801(a), 21451(a))
      first-in          the one that entered the intersection later yields to
                        the one lawfully within it (21451(a))
      simultaneous      neither: no priority
    mixed-control     a light against a stop sign or no control: no priority
    stop-sign         one faces a stop sign (or a flashing red, 21457), the
                      other no control: the stopped one yields (21802)
    all-way-stop/...  both face stop signs (or both turn on red):
      first-in          the first to arrive (within ARRIVE of its stop line)
                        goes first
      order-unknown     both were waiting when the clip began: no priority.
                        With ``order_unknown_tie`` they count as simultaneous
                        and the left-turn and yield-right rules below decide
                        (order-unknown-left-turn, order-unknown-yield-right)
      left-turn, yield-right   arrived within TIE seconds of each other:
                        left turns yield to oncoming traffic (21801(a)),
                        then yield to the vehicle on the right (21800(b))
    neither controlled:
      through-road      a T junction: the road that ends there yields to the
                        one that continues (21800(c)); a road continues when
                        its lane, or a neighbour, has an exit within TURN of
                        straight on
      left-turn         as above (21801(a))
      uncontrolled      otherwise no priority. With ``uncontrolled_order``
                        the CVC's order applies instead: first-in (21800(a))
                        then yield-right (21800(b)). It is off by default:
                        a junction the map shows no control for is mostly
                        controlled in reality by signs or lights WOMD lacks,
                        and over CAT's 500 logged scenes (below) the holder
                        these two rules name went first in 40% (of 15) and
                        25% (of 8) of the pairs, no better than chance

"Entered" is the step the agent crossed the stop line into the
intersection (the first controlled lane), or else began its run on the
lane P* lies on (for a merge, the lane the two share after it).

Duties, as Signal Temporal Logic (responsibility/stl.py) over the steps
both are seen, every predicate a signed distance in metres:

    in_C(x)       h_x - |s_x - s*_x|          x's footprint is in the zone
    cleared(x)    s_x - s*_x - h_x            x has passed it
    can_stop(x)   (s*_x - h_x - s_x) - (v_x rho + v_x^2 / (2 brake))
                  x could still stop before the zone after the response
                  time rho, braking at ``brake`` (RSS's rho and brake_min)
    enters(x)     in_C(x) and not prev in_C(x)

    yield(y, p)   G( enters(y) -> cleared(p) or can_stop(p) )
                  the yielder y enters the zone only when the holder p has
                  passed it or can still stop before it: "yield to vehicles
                  close enough to constitute an immediate hazard"
                  (21800-21802)
    avoid(x, z)   G( in_C(z) and can_stop(x) and not in_C(x)
                     -> (not in_C(x)) W (not in_C(z)) )
                  nobody drives into a zone the other occupies while it
                  can still stop before it (21451(a): yield to vehicles
                  lawfully within the intersection), right of way or not

The yielder owes yield and avoid, the holder avoid; without a holder both
owe avoid only (between same-direction paths neither duty is judged
unless ``same_direction_rules``). An agent inside the zone when the two are first seen
entered it before then: its entry is not judged. A violated duty (negative
robustness) puts the collision on that agent: "ego", "other", "shared" when
both failed, "n/a" when neither did (".../no-violation"; e.g. a yielder
that entered lawfully and was hit downstream of the zone, which is a
rear-end collision) or no conflict was found. Robustness of an
event-triggered formula is bounded by how sharply the event is observed
(one 10 Hz step), so only its sign is the verdict; ``yield_gap`` gives the
holder's can_stop margin when the yielder entered, the informative number.

Checked on CAT's 500 logged scenes, the self-driving car against the other
object of interest at their closest approach (logged drivers mostly keep
the right of way, so its holder should usually pass P* first): of 220
pairs whose paths cross or merge and where one passed P* at least 0.5 s
before the other, 113 get a holder, and the holder passed first in 89%
(stop-sign 94% of 49, through-road 68% of 19, all-way-stop/first-in 100%
of 16, driveway 100% of 10, red-light 88% of 8, left-turn 100% of 7).
Most disagreements are a yielder that went first lawfully, the holder
still far away; the rest are left without a holder (signal-unknown 38,
uncontrolled 23, same-direction 20, all-way-stop/order-unknown 17,
oncoming 9). On those logged pairs the yield duty fails for 2% of the
yielders (2 of 112 where one entered the zone): the false-alarm rate of
the duties on lawful driving. It fails as rarely for the holders, had they
been the yielders (4 of 112): logged conflicts are seldom close enough for
the duty to tell the two apart, so this checks the duties, not the
priority. These scenes were the development set. Held out, on 3119 pairs
of WOMD's validation_interactive split (the two objects of interest;
scripts/responsibility/validate_right_of_way.py), the holder passed first
in 83% of 330 (stop-sign 92% of 130, driveway 74% of 62, through-road 85%
of 40, all-way-stop/first-in 91% of 35, left-turn 64% of 28, red-light
87% of 23) and the yield duty failed for 7 of 323 yielders (2%); with
``same_direction_rules`` the lane-change holder passed first in 38% of 85
and the duty failed for 51 of 91 lane-change and 32 of 61 merging yielders.

Approximations: WOMD has no yield signs, so a yield-controlled approach
counts as uncontrolled; roundabouts and lane drops have no rule of their
own (a roundabout's entry can come out as through-road the wrong way
round); lane changes are read from the lane matching, which also jitters
between parallel turn lanes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from responsibility import stl
from responsibility.lanes import HEADING_WEIGHT, lane_graph
from responsibility.scene import Scene

DT = 0.1  # s per step
SAME_DIRECTION = np.deg2rad(45.0)  # rad: approach headings closer than this travel the same way
ONCOMING = np.deg2rad(135.0)  # rad: approach headings further apart than this come from opposite directions
TURN = np.deg2rad(30.0)  # rad: a lane turning more than this is a turn
U_TURN = np.deg2rad(150.0)  # rad: and more than this a U-turn
SIGN_OFFSET = 7.0  # m: a stop sign without lanes controls an approach it stands this close to, on its right
LANE_NEAR = 3.0  # m: a controlled lane not yet entered leads to P* when it passes this close to it
GO, RED, ARROW, STOP, NONE, UNKNOWN = "go", "red", "arrow", "stop", "none", "unknown"
LIGHTS = {  # WOMD lane state -> the control it imposes
    "LANE_STATE_GO": GO, "LANE_STATE_CAUTION": GO, "LANE_STATE_ARROW_CAUTION": GO,
    "LANE_STATE_ARROW_GO": ARROW,
    "LANE_STATE_STOP": RED, "LANE_STATE_ARROW_STOP": RED,
    "LANE_STATE_FLASHING_STOP": STOP,  # a flashing red is a stop sign (21457(a))
    "LANE_STATE_FLASHING_CAUTION": NONE,  # a flashing yellow: proceed with caution
    "LANE_STATE_UNKNOWN": UNKNOWN,
    # MetaDrive's names, should a converter store those (they drop the arrows)
    "TRAFFIC_LIGHT_GREEN": GO, "TRAFFIC_LIGHT_YELLOW": GO, "TRAFFIC_LIGHT_RED": RED, "TRAFFIC_LIGHT_UNKNOWN": UNKNOWN,
}
PERMISSIVE = (GO, ARROW)


@dataclass(frozen=True)
class RightOfWayParams:
    rho: float = 1.0  # s: response time of can_stop (RSS's)
    brake: float = 4.0  # m/s^2: braking of can_stop (RSS's brake_min)
    tie: float = 1.0  # s: entries closer than this are simultaneous
    extend: float = 15.0  # m: paths go on this far along their last heading
    merge_gap: float = 1.0  # m: paths this close have merged
    near: float = 30.0  # m: a crossing further than this from the agents at ``at`` is not their conflict
    approach: float = 30.0  # m before P* where the approach heading is read
    control_reach: float = 50.0  # m before P* in which a lane's light or stop sign controls the agent
    signal_radius: float = 40.0  # m: lights this close to P* make it a signalised intersection
    arrive: float = 5.0  # m before its stop line where an agent has arrived at an all-way stop
    uncontrolled_order: bool = False  # first-in and yield-right where the map shows no control (21800(a), (b))
    same_direction_rules: bool = False  # lane-change and merge priorities, and duties, between same-direction paths
    order_unknown_tie: bool = False  # an all-way stop both were already waiting at: treat as simultaneous (21801, 21800(b))


@dataclass
class Conflict:
    point: np.ndarray  # P* [2]
    s: Tuple[float, float]  # s*_a, s*_b: P* along each path (m)
    angle: float  # rad, in [0, pi]: between the paths at P*
    kind: str  # "crossing", "merge" or "closest"


@dataclass
class Approach:
    """What one agent brought to the conflict."""
    control: str  # GO, ARROW, RED, STOP, NONE or UNKNOWN
    heading: float  # rad: approach heading, APPROACH metres before P*
    turn: str  # "left", "right", "u-turn", "straight"
    entered: Optional[int]  # step it entered the intersection (its conflict lane; None: never seen to)
    changed_lane: bool  # moved in from a neighbouring lane before P*
    arrived: Optional[int] = None  # step it came within ARRIVE metres of its stop line (-1: before it was
                                   # first seen; None: not seen to)
    on_lane: bool = True  # on a mapped lane within APPROACH metres before P* (not a driveway or a car park)
    through: Optional[bool] = None  # its road continues across the junction (False: it ends there; None: unknown)


@dataclass
class Priority:
    holder: Optional[int]  # the agent with the right of way, None when no rule gives it
    case: str
    conflict: Optional[Conflict] = None
    approaches: Optional[Tuple[Approach, Approach]] = None


@dataclass
class RightOfWay:
    verdict: str  # "ego", "other", "shared" or "n/a"
    case: str  # the rule that gave the priority (or why there is none), "/no-violation" without a verdict
    priority: str  # "ego", "other" or "none"
    rho_yield: Optional[float] = None  # robustness of yield(yielder, holder); None without a holder
    rho_avoid_ego: Optional[float] = None
    rho_avoid_other: Optional[float] = None
    yield_gap: Optional[float] = None  # m: the holder's can_stop when the yielder entered (worst entry)

    def as_row(self) -> Dict:
        return {"right_of_way": self.verdict, "right_of_way_case": self.case, "priority": self.priority}


def _angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class _Path:
    """An agent's centre track over its valid steps up to ``last``, then
    ``extend`` metres on along its last heading."""

    def __init__(self, scene: Scene, agent: int, last: int, extend: float):
        self.steps = np.flatnonzero(scene.valid[agent, :last + 1])
        real = scene.position[agent, self.steps, :2].astype(float)
        heading = float(scene.heading[agent, self.steps[-1]])
        ahead = real[-1] + np.arange(1, int(np.ceil(extend)) + 1)[:, None] * [np.cos(heading), np.sin(heading)]
        self.points = np.concatenate([real, ahead])
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))])
        self.n_real = len(real)
        self.headings = scene.heading[agent, self.steps].astype(float)

    def s_at(self, steps: np.ndarray) -> np.ndarray:
        return self.s[np.searchsorted(self.steps, steps)]

    def project(self, q: np.ndarray, real: bool = False) -> Tuple[float, float]:
        """(arc length, distance) of the path point closest to q (on the
        real track only, without the extension, if ``real``)."""
        n = self.n_real + 1 if real else len(self.points)
        a, b = self.points[:n - 1], self.points[1:n]
        d = b - a
        length2 = np.maximum(np.sum(d ** 2, -1), 1e-12)
        u = np.clip(np.sum((q - a) * d, -1) / length2, 0.0, 1.0)
        dist = np.linalg.norm(a + u[:, None] * d - q, axis=-1)
        i = int(np.argmin(dist))
        return float(self.s[i] + u[i] * np.sqrt(length2[i])), float(dist[i])

    def point_at(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.s, self.points[:, 0]), np.interp(s, self.s, self.points[:, 1])])

    def step_index(self, s: float) -> int:
        """Index (into ``steps``) of the last real point at or before arc length s."""
        return int(np.clip(np.searchsorted(self.s[:self.n_real], s, side="right") - 1, 0, self.n_real - 1))

    def heading_at(self, s: float) -> float:
        return float(self.headings[self.step_index(s)])


def _crossings(pa: _Path, pb: _Path) -> List[Tuple[float, float, np.ndarray]]:
    """[(s_a, s_b, point)] where the two polylines cross."""
    a0, a1 = pa.points[:-1], pa.points[1:]
    b0, b1 = pb.points[:-1], pb.points[1:]
    da, db = a1 - a0, b1 - b0
    r = b0[None] - a0[:, None]  # [A, B, 2]
    den = da[:, None, 0] * db[None, :, 1] - da[:, None, 1] * db[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (r[..., 0] * db[None, :, 1] - r[..., 1] * db[None, :, 0]) / den
        u = (r[..., 0] * da[:, None, 1] - r[..., 1] * da[:, None, 0]) / den
    hit = (np.abs(den) > 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    out = []
    for i, j in zip(*np.nonzero(hit)):
        out.append((float(pa.s[i] + t[i, j] * (pa.s[i + 1] - pa.s[i])),
                    float(pb.s[j] + u[i, j] * (pb.s[j + 1] - pb.s[j])), a0[i] + t[i, j] * da[i]))
    return out


def find_conflict(pa: _Path, pb: _Path, ref: np.ndarray, p: RightOfWayParams) -> Optional[Conflict]:
    """P* of two paths (module docstring); ``ref``: where the agents were."""
    def conflict(s_a, s_b, point, kind):
        angle = abs(float(_angle(pa.heading_at(s_a) - pb.heading_at(s_b))))
        return Conflict(np.asarray(point, dtype=float), (float(s_a), float(s_b)), angle, kind)

    crossings = [c for c in _crossings(pa, pb) if np.linalg.norm(c[2] - ref) <= p.near]
    if crossings:
        s_a, s_b, point = min(crossings, key=lambda c: np.linalg.norm(c[2] - ref))
        return conflict(s_a, s_b, point, "crossing")
    gaps = np.array([pb.project(q)[1] for q in pa.points])
    close = gaps <= p.merge_gap
    if np.any(close):  # the run of merged points nearest to the agents; P* where it starts
        starts = np.flatnonzero(close & ~np.concatenate([[False], close[:-1]]))
        ends = np.flatnonzero(close & ~np.concatenate([close[1:], [False]]))
        k = min(range(len(starts)), key=lambda k: np.linalg.norm(pa.points[starts[k]:ends[k] + 1] - ref, axis=1).min())
        if np.linalg.norm(pa.points[starts[k]:ends[k] + 1] - ref, axis=1).min() <= p.near:
            point = pa.points[starts[k]]
            return conflict(pa.s[starts[k]], pb.project(point)[0], point, "merge")
    i = int(np.argmin(gaps))
    s_b, _ = pb.project(pa.points[i])
    point = (pa.points[i] + pb.point_at(s_b)) / 2
    if np.linalg.norm(point - ref) > p.near:
        return None
    return conflict(pa.s[i], s_b, point, "closest")


def _lights(scene: Scene) -> Dict[str, Dict]:
    return {str(light["lane"]): light for light in scene.dynamic_map_states.values()
            if light.get("type") == "TRAFFIC_LIGHT" and "lane" in light}


def _best_of(graph, xy: np.ndarray, heading: float, lanes: Sequence[str]) -> str:
    """The lane of ``lanes`` that fits a vehicle at ``xy`` best, by
    LaneGraph.best's score (distance plus HEADING_WEIGHT m per rad)."""
    def score(lane):
        d = np.linalg.norm(graph.points[lane] - xy, axis=1)
        i = int(np.argmin(d))
        return d[i] + HEADING_WEIGHT * abs(float(_angle(graph.direction[lane][i] - heading)))
    return min(sorted(lanes), key=score)


def _runs(seq: Sequence[Optional[str]]) -> List[List]:
    """[[lane, first index, last index]] of the runs of equal lanes (None skipped)."""
    runs = []
    for k, lane in enumerate(seq):
        if lane is None:
            continue
        if runs and runs[-1][0] == lane and runs[-1][2] == k - 1:
            runs[-1][2] = k
        else:
            runs.append([lane, k, k])
    return runs


def _past_end(graph, lane: Optional[str], xy: np.ndarray) -> bool:
    """Whether xy lies beyond the end of ``lane`` (its centreline still
    matches there for MATCH_RADIUS metres)."""
    if lane is None:
        return False
    pts = graph.points[lane]
    d = pts[-1] - pts[-2]
    return int(np.argmin(np.linalg.norm(pts - xy, axis=1))) == len(pts) - 1 and float(np.dot(xy - pts[-1], d)) > 0


def lane_sequence(graph, points: np.ndarray, headings: Sequence[float]) -> List[Optional[str]]:
    """The lane a vehicle is on at each of ``points``, followed through the
    lane graph: it stays on its lane while that still matches
    (LaneGraph.matched) and it has not passed the lane's end, then moves to
    a matching exit lane, else a matching neighbour, else the best match;
    None where no lane matches. Lanes that branch at one stop line (straight
    on, turning) overlap for metres, so the branch first matched may be the
    one not taken: a run on lane A directly followed by B, a sibling of A
    (a common predecessor) rather than its exit or neighbour, is relabelled
    B."""
    seq, cur = [], None
    for xy, h in zip(points, headings):
        matched = graph.matched(xy, float(h))
        if not matched:
            seq.append(None)
            continue
        if cur not in matched or _past_end(graph, cur, xy):
            onward = [x for x in graph.exits.get(cur, []) if x in matched]
            side = [x for x in graph.sides.get(cur, []) if x in matched]
            if onward or side or cur not in matched:
                cur = _best_of(graph, xy, float(h), onward or side or list(matched))
        seq.append(cur)
    preds: Dict[str, set] = {}
    for lane, exits in graph.exits.items():
        for x in exits:
            preds.setdefault(x, set()).add(lane)
    runs = _runs(seq)
    for k in range(len(runs) - 2, -1, -1):  # backwards, so a chain of siblings ends on the branch taken
        a, b = runs[k][0], runs[k + 1][0]
        if (runs[k + 1][1] == runs[k][2] + 1 and preds.get(a, set()) & preds.get(b, set())
                and b not in graph.exits.get(a, []) and b not in graph.sides.get(a, [])):
            runs[k][0] = b
            seq[runs[k][1]:runs[k][2] + 1] = [b] * (runs[k][2] - runs[k][1] + 1)
    return seq


def _stop_signs(scene: Scene) -> Tuple[set, List[np.ndarray]]:
    """The lanes stop signs control, and the positions of the signs whose
    lanes the map lacks."""
    graph = lane_graph(scene)
    lanes, loose = set(), []
    for f in scene.map_features.values():
        if f.get("type") != "STOP_SIGN":
            continue
        controlled = {str(lane) for lane in np.atleast_1d(f.get("lane", []))}
        lanes |= controlled
        if not controlled & set(graph.points):
            loose.append(np.asarray(f["position"], dtype=float)[:2])
    return lanes, loose


def _approach(scene: Scene, agent: int, path: _Path, s_star: float, p: RightOfWayParams) -> Approach:
    """What ``agent`` brought to the conflict at s_star along its path
    (module docstring): its lanes from CONTROL_REACH before P* to P* tell
    its control, its turn, when it entered and whether it changed lanes."""
    graph = lane_graph(scene)
    first = path.step_index(s_star - p.control_reach)
    star = min(int(np.searchsorted(path.s, s_star)), len(path.points) - 1)  # first path point at or past P*
    idx = np.arange(first, star + 1)
    seq = lane_sequence(graph, path.points[idx], path.headings[np.minimum(idx, path.n_real - 1)])
    runs = _runs(seq)
    last_seen = path.n_real - 1

    def step_of(k: int) -> int:  # a point of seq -> its step; points of the extension lie after the last seen
        i = first + k
        return int(path.steps[i]) if i <= last_seen else int(path.steps[-1]) + 1

    lights = _lights(scene)
    stop_lanes, loose_signs = _stop_signs(scene)
    controlled = [r[0] in lights or r[0] in stop_lanes for r in runs]
    block = []  # the last stretch of consecutive controlled lanes: the intersection entered last
    for k in range(len(runs) - 1, -1, -1):
        if not controlled[k] or (block and runs[k][2] + 1 != runs[block[0]][1]):
            if block:
                break
            continue
        block.insert(0, k)

    control, line = None, None  # line: the controlled lane the agent entered, or is about to
    if block:
        lit = [k for k in block if runs[k][0] in lights]
        line, k0, _ = runs[lit[0] if lit else block[0]]
        step = step_of(k0)
    elif runs:  # not yet across the stop line: the controlled exit of its lane that leads to P*
        ahead = [e for e in graph.exits.get(runs[-1][0], []) if e in lights or e in stop_lanes]
        ahead = [e for e in ahead if np.linalg.norm(graph.points[e] - path.point_at(s_star), axis=1).min() <= LANE_NEAR]
        if ahead:
            line = min(ahead, key=lambda e: np.linalg.norm(graph.points[e] - path.point_at(s_star), axis=1).min())
            step = int(path.steps[-1]) + 1
    if line is not None and line in lights:  # the light at the stop line when the agent crossed it (or the last seen)
        states = lights[line]["state"]["object_state"]
        step = min(step, int(path.steps[-1]))
        state = states[step] if step < len(states) else None
        control = LIGHTS.get(state, UNKNOWN) if state else UNKNOWN
    elif line is not None:
        control = STOP
    else:
        for sign in loose_signs:  # a sign the map gives no lanes for: at the right kerb of the approach
            s_sign, dist = path.project(sign, real=True)
            i = path.step_index(s_sign)
            h = path.headings[i]
            right = np.cos(h) * (sign[1] - path.points[i, 1]) - np.sin(h) * (sign[0] - path.points[i, 0]) < 0
            if dist <= SIGN_OFFSET and right and s_star - p.control_reach <= s_sign <= s_star:
                control = STOP
        if control is None:
            near = [np.asarray(light["stop_point"], float)[:2] for light in lights.values() if "stop_point" in light]
            close = any(np.linalg.norm(x - path.point_at(s_star)) <= p.signal_radius for x in near)
            control = UNKNOWN if close else NONE

    conflict_lane = runs[-1][0] if runs else None
    entered = step_of(runs[block[0]][1]) if block else step_of(runs[-1][1]) if runs else None
    arrived = None
    if line is not None:
        s_line, _ = path.project(graph.points[line][0])
        there = np.flatnonzero(path.s[:path.n_real] >= s_line - p.arrive)
        arrived = None if not there.size else -1 if there[0] == 0 else int(path.steps[there[0]])
    heading = path.heading_at(max(s_star - p.approach, 0.0))
    end = graph.direction[conflict_lane][-1] if conflict_lane is not None else path.headings[-1]
    turn = float(_angle(end - heading))
    kind = ("u-turn" if abs(turn) > U_TURN else "left" if turn > TURN else "right" if turn < -TURN
            else "straight")
    changed = any(b[0] in graph.sides.get(a[0], []) or a[0] in graph.sides.get(b[0], [])
                  for a, b in zip(runs, runs[1:]))
    near = [r for r in runs if path.s[first + r[2]] >= s_star - p.approach]  # runs within APPROACH of P*
    return Approach(control, heading, kind, entered, changed, arrived=arrived, on_lane=bool(near),
                    through=_through(graph, near[0][0], conflict_lane) if near else None)


def _through(graph, approach: str, conflict_lane: Optional[str]) -> Optional[bool]:
    """Whether the road of lane ``approach`` continues across the junction
    ahead: the agent drives on along it to P*, or it or a neighbour has an
    exit lane that ends within TURN of its direction (a turn pocket's
    neighbour goes straight on). False where every exit turns (the road
    ends at the junction), None where the map shows no exit."""
    if approach == conflict_lane:
        return True
    lanes = [approach] + graph.sides.get(approach, [])
    exits = [(lane, e) for lane in lanes for e in graph.exits.get(lane, [])]
    if not exits:
        return None
    return any(abs(float(_angle(graph.direction[e][-1] - graph.direction[lane][-1]))) < TURN for lane, e in exits)


def _following(scene: Scene, a: int, b: int, t0: int, pa: _Path, pb: _Path, p: RightOfWayParams) -> bool:
    """Whether a and b were on one path when first seen together: one
    behind the other, travelling the same way."""
    if abs(float(_angle(scene.heading[a, t0] - scene.heading[b, t0]))) > SAME_DIRECTION:
        return False
    return (pb.project(scene.position[a, t0, :2].astype(float), real=True)[1] <= p.merge_gap
            or pa.project(scene.position[b, t0, :2].astype(float), real=True)[1] <= p.merge_gap)


def _rules(a: Approach, b: Approach, merge: bool, p: RightOfWayParams) -> Tuple[Optional[str], str]:
    """("a" / "b" / None, case): who holds the right of way (module docstring)."""
    def first(ta: Optional[int], tb: Optional[int]) -> Optional[str]:
        if ta is None and tb is None:
            return None
        if ta is None or tb is None:
            return "b" if ta is None else "a"
        return None if abs(ta - tb) * DT <= p.tie else "a" if ta < tb else "b"

    rel = float(_angle(b.heading - a.heading))  # > 0: b comes from a's right (heads to a's left)
    if abs(rel) < SAME_DIRECTION:  # one road: signals and signs do not part them
        if not p.same_direction_rules:
            return None, "same-direction"
        if a.changed_lane != b.changed_lane:
            return ("b" if a.changed_lane else "a"), "lane-change"
        holder = first(a.entered, b.entered) if merge else None
        return holder, "merge" if holder is not None else "same-direction"
    if a.on_lane != b.on_lane:  # entering the road from a driveway or a car park (21804(a))
        return ("a" if a.on_lane else "b"), "driveway"
    a_left, b_left = a.turn in ("left", "u-turn"), b.turn in ("left", "u-turn")
    oncoming = abs(rel) > ONCOMING
    if oncoming and a.turn == b.turn == "straight":  # passing each other: no conflict of priority
        return None, "oncoming"
    if UNKNOWN in (a.control, b.control):
        return None, "signal-unknown"
    if a.control == RED and b.control in PERMISSIVE:
        return "b", "red-light"
    if b.control == RED and a.control in PERMISSIVE:
        return "a", "red-light"
    left_turn = ("b" if a_left else "a") if oncoming and a_left != b_left else None
    right = "b" if SAME_DIRECTION < rel < ONCOMING else "a" if -ONCOMING < rel < -SAME_DIRECTION else None

    if a.control in PERMISSIVE and b.control in PERMISSIVE:
        if (a.control == ARROW) != (b.control == ARROW):
            return ("a" if a.control == ARROW else "b"), "protected-arrow"
        for holder, case in ((left_turn, "left-turn"), (first(a.entered, b.entered), "first-in")):
            if holder is not None:
                return holder, case
        return None, "simultaneous"
    if a.control in PERMISSIVE or b.control in PERMISSIVE:
        return None, "mixed-control"
    stop_a, stop_b = a.control in (STOP, RED), b.control in (STOP, RED)
    if stop_a != stop_b:
        return ("b" if stop_a else "a"), "stop-sign"
    if stop_a:  # all-way stop: the first to arrive goes first; on a tie left turns yield, then yield to the right
        rules = ((left_turn, "left-turn"), (right, "yield-right"))
        if a.arrived == b.arrived == -1:  # both were waiting when the clip began
            if not p.order_unknown_tie:
                return None, "all-way-stop/order-unknown"
            return next(((h, "all-way-stop/order-unknown-" + c) for h, c in rules if h is not None),
                        (None, "all-way-stop/order-unknown"))
        holder = first(a.arrived, b.arrived)
        if holder is not None:
            return holder, "all-way-stop/first-in"
        return next(((h, "all-way-stop/" + c) for h, c in rules if h is not None), (None, "all-way-stop/simultaneous"))
    if {a.through, b.through} == {True, False}:  # a T junction: the ending road yields (21800(c))
        return ("a" if a.through else "b"), "through-road"
    if left_turn is not None:
        return left_turn, "left-turn"
    if p.uncontrolled_order:
        for holder, case in ((first(a.entered, b.entered), "first-in"), (right, "yield-right")):
            if holder is not None:
                return holder, case
    return None, "uncontrolled"


def _window(scene: Scene, a: int, b: int, at: int) -> np.ndarray:
    """The run of steps both are seen that ends last at or before ``at``."""
    both = scene.valid[a] & scene.valid[b]
    last = next((k for k in range(min(at, scene.n_steps - 1), -1, -1) if both[k]), None)
    if last is None:
        return np.zeros(0, dtype=int)
    first = last
    while first > 0 and both[first - 1]:
        first -= 1
    return np.arange(first, last + 1)


def priority(scene: Scene, a: int, b: int, at: int, params: Optional[RightOfWayParams] = None,
             conflict: Optional[Conflict] = None) -> Priority:
    """Who of ``a`` and ``b`` had the right of way at the conflict their
    paths up to step ``at`` lead to (e.g. a collision, or their closest
    approach). A given ``conflict`` (its arc lengths along each agent's
    track from its first seen step) is judged instead of the one found
    from the paths up to ``at``."""
    p = params or RightOfWayParams()
    steps = _window(scene, a, b, at)
    if steps.size == 0:
        return Priority(None, "not-seen-together")
    if steps.size < 2:
        return Priority(None, "too-short")
    last = int(steps[-1])
    pa, pb = _Path(scene, a, last, p.extend), _Path(scene, b, last, p.extend)
    before = steps[-2] if last == at else last  # where they were before contact
    ref = (scene.position[a, before, :2] + scene.position[b, before, :2]).astype(float) / 2
    conflict = conflict or find_conflict(pa, pb, ref, p)
    if conflict is None:
        return Priority(None, "no-conflict")
    if _following(scene, a, b, int(steps[0]), pa, pb, p):
        return Priority(None, "following", conflict)
    approaches = (_approach(scene, a, pa, conflict.s[0], p), _approach(scene, b, pb, conflict.s[1], p))
    holder, case = _rules(*approaches, conflict.kind == "merge", p)
    return Priority({"a": a, "b": b, None: None}[holder], case, conflict, approaches)


def _signals(scene: Scene, x: int, other: int, path: _Path, s_star: float, angle: float, steps: np.ndarray,
             p: RightOfWayParams):
    """in_C, cleared and can_stop of agent x over ``steps`` (metres)."""
    length, width = scene.shape_at(x, int(steps[-1]))
    _, width_other = scene.shape_at(other, int(steps[-1]))
    sin = max(np.sin(angle), 0.5)
    h = length / 2 + (width_other + width * np.sqrt(1 - sin ** 2)) / (2 * sin)
    s = path.s_at(steps)
    v = np.linalg.norm(scene.velocity[x, steps], axis=-1).astype(float)
    in_zone = h - np.abs(s - s_star)
    cleared = s - s_star - h
    can_stop = (s_star - h - s) - (v * p.rho + v ** 2 / (2 * p.brake))
    return in_zone, cleared, can_stop


def _enters(in_zone):
    return stl.and_(in_zone, stl.not_(stl.prev(in_zone, first=stl.INF)))  # inside when first seen: not judged


def yield_duty(in_y, cleared_p, can_stop_p):
    """yield(y, p) = G( enters(y) -> cleared(p) or can_stop(p) ), per step."""
    return stl.always(stl.implies(_enters(in_y), stl.or_(cleared_p, can_stop_p)))


def avoid_duty(in_x, can_stop_x, in_z):
    """avoid(x, z) = G( in_C(z) and can_stop(x) and not in_C(x) -> (not in_C(x)) W (not in_C(z)) )."""
    occupied = stl.and_(in_z, can_stop_x, stl.not_(in_x))
    return stl.always(stl.implies(occupied, stl.weak_until(stl.not_(in_x), stl.not_(in_z))))


def _finite(x: Optional[float]) -> Optional[float]:
    """None for a vacuous robustness (+-inf: the duty never applied)."""
    return float(x) if x is not None and np.isfinite(x) else None


def right_of_way_blame(scene: Scene, ego: int, other: int, crash_step: int,
                       params: Optional[RightOfWayParams] = None) -> RightOfWay:
    """The right-of-way verdict on a collision of ``ego`` with ``other`` at
    ``crash_step`` (module docstring)."""
    p = params or RightOfWayParams()
    prio = priority(scene, ego, other, crash_step, p)
    holder = "none" if prio.holder is None else "ego" if prio.holder == ego else "other"
    if prio.approaches is None or (prio.case == "same-direction" and not p.same_direction_rules):  # or RSS's
        return RightOfWay("n/a", prio.case, holder)
    steps = _window(scene, ego, other, crash_step)
    c = prio.conflict
    pe, po = _Path(scene, ego, int(steps[-1]), p.extend), _Path(scene, other, int(steps[-1]), p.extend)
    in_e, cleared_e, stop_e = _signals(scene, ego, other, pe, c.s[0], c.angle, steps, p)
    in_o, cleared_o, stop_o = _signals(scene, other, ego, po, c.s[1], c.angle, steps, p)
    avoid_e = float(avoid_duty(in_e, stop_e, in_o)[0])
    avoid_o = float(avoid_duty(in_o, stop_o, in_e)[0])
    ok_e, ok_o = avoid_e >= 0, avoid_o >= 0
    rho_yield = gap = None
    if prio.holder is not None:
        in_y, cleared_h, stop_h = (in_e, cleared_o, stop_o) if holder == "other" else (in_o, cleared_e, stop_e)
        rho_yield = float(yield_duty(in_y, cleared_h, stop_h)[0])
        entries = _enters(in_y) > 0
        if np.any(entries):
            gap = float(stl.or_(cleared_h, stop_h)[entries].min())
        if holder == "other":
            ok_e &= rho_yield >= 0
        else:
            ok_o &= rho_yield >= 0
    if ok_e and ok_o:
        verdict, case = "n/a", prio.case + "/no-violation"
    else:
        verdict, case = ("shared" if not ok_e and not ok_o else "ego" if not ok_e else "other"), prio.case
    return RightOfWay(verdict, case, holder, _finite(rho_yield), _finite(avoid_e), _finite(avoid_o), gap)


def rollout_right_of_way(scene: Scene, rollout: Dict,
                         params: Optional[RightOfWayParams] = None) -> Optional[RightOfWay]:
    """right_of_way_blame for a rollout that ended in a vehicle collision,
    in the scene rebuilt from it, with the partner blame.rollout_blame uses;
    None after the end of the log or with nobody to have collided with."""
    from responsibility.blame import rollout_collision

    collision = rollout_collision(scene, rollout)
    if collision is None:
        return None
    step, other = collision
    return right_of_way_blame(scene, scene.sdc, other, step, params)
