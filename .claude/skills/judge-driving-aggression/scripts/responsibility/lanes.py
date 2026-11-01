"""The lane graph of a scene's map, for telling an alternative's intent:
which lanes an agent's logged route runs along, and which goals lie on them.

WOMD / ScenarioNet lanes are short segments (in scene 0: 68 lanes, median
51 m) linked by ``entry_lanes`` / ``exit_lanes`` and side by side through
``left_neighbor`` / ``right_neighbor``. An agent's route is therefore a set
of lanes, not one: braking alternatives end on a lane segment the agent
passed earlier, faster ones on a successor.

  route_lanes      the lanes the agent's logged motion from step k on runs
                   along, their left and right neighbours (a lane change
                   keeps the intent), and the chain of exit lanes after the
                   last of them (all branches: where the agent goes after
                   the log ends is unknown), each with its neighbours, up to
                   ``beyond`` metres further. Each logged position is
                   matched to one lane: the closest centreline within
                   MATCH_RADIUS running within MATCH_ALIGN of the agent's
                   heading (distance plus HEADING_WEIGHT m per rad). A lane
                   counts once the agent has driven MIN_ALONG metres matched
                   to it (half its length if shorter): where a turn lane
                   branches off, it overlaps the straight one for a metre or
                   two, and a car going straight passes it there. The lane
                   it is on at step k always counts.
  reachable_lanes  every lane an agent at step k can reach: the lanes it is
                   on, then the chain of exit lanes up to ``reach`` metres,
                   each with its neighbours -- the support of its valid goals.
  near             which points lie within ``radius`` of a lane's centreline.

Neighbours are leaves: following neighbours of neighbours (and their exits)
would spread across a whole intersection of short lane segments. Both
route_lanes and reachable_lanes give None -- no restriction -- for an agent
the map does not cover: on no lane at step k or at its last logged
position, or less than MIN_ON_LANES of its logged positions from step k on
a lane (a parking lot, or a driveway, which CAT's WOMD v1.1 maps lack).
"""

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from responsibility.scene import Scene

LANE_TYPES = ("LANE_FREEWAY", "LANE_SURFACE_STREET", "LANE_UNDEFINED")
MATCH_RADIUS = 2.5  # m: a logged position is on a lane within this of its centreline
MATCH_ALIGN = np.deg2rad(45.0)  # rad: and running the agent's way within this
SPACING = 1.0  # m between the centreline points distances are measured to
HEADING_WEIGHT = 2.0  # m per rad of heading difference, when picking a position's lane
MIN_ALONG = 5.0  # m driven along a lane before it counts as part of the route
MIN_ON_LANES = 0.5  # share of logged positions on a lane below which the map does not cover the motion


def _angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _densify(poly: np.ndarray, spacing: float = SPACING) -> np.ndarray:
    seg = poly[1:] - poly[:-1]
    n = np.maximum(1, np.ceil(np.sqrt((seg * seg).sum(axis=1)) / spacing).astype(int))
    k = np.arange(1, n.sum() + 1) - np.repeat(np.cumsum(n) - n, n)  # 1..n within each segment
    step = np.repeat(seg, n, axis=0) * (k / np.repeat(n, n))[:, None]
    return np.concatenate([poly[:1], np.repeat(poly[:-1], n, axis=0) + step])


class LaneGraph:
    """The vehicle lanes of one scene's map and how they connect."""

    def __init__(self, map_features: Dict):
        self.points: Dict[str, np.ndarray] = {}  # lane id -> centreline points [P, 2], ~1 m apart
        self.direction: Dict[str, np.ndarray] = {}  # lane id -> direction [P] of travel at each point
        self.length: Dict[str, float] = {}
        self.exits: Dict[str, List[str]] = {}
        self.sides: Dict[str, List[str]] = {}
        for key, f in map_features.items():
            poly = np.asarray(f.get("polyline", []), dtype=float)
            if f.get("type") not in LANE_TYPES or poly.ndim != 2 or len(poly) < 2:
                continue
            key = str(key)
            pts = _densify(poly[:, :2])
            d = np.diff(pts, axis=0)
            heading = np.arctan2(d[:, 1], d[:, 0])
            self.points[key] = pts
            self.direction[key] = np.append(heading, heading[-1])
            self.length[key] = float(np.linalg.norm(d, axis=1).sum())
            self.exits[key] = [str(x) for x in f.get("exit_lanes", [])]
            self.sides[key] = [str(n["feature_id"]) for side in ("left_neighbor", "right_neighbor")
                               for n in f.get(side, []) if isinstance(n, dict) and "feature_id" in n]
        for key in self.points:  # links to lanes that are not vehicle lanes (or not in the map) are dropped
            self.exits[key] = [x for x in self.exits[key] if x in self.points]
            self.sides[key] = [x for x in self.sides[key] if x in self.points]

    def __len__(self) -> int:
        return len(self.points)

    def _within(self, xy: np.ndarray) -> List[str]:
        """The lanes with a point within MATCH_RADIUS of ``xy``, in map order: the only ones whose
        closest point can pass the MATCH_RADIUS test (a k-d tree finds them; the test is unchanged)."""
        if not hasattr(self, "_tree"):
            from scipy.spatial import cKDTree

            keys = list(self.points)
            self._keys = keys
            self._tree = cKDTree(np.concatenate([self.points[k] for k in keys])) if keys else None
            self._owner = np.repeat(np.arange(len(keys)), [len(self.points[k]) for k in keys])
        if self._tree is None:
            return []
        hits = self._tree.query_ball_point(np.asarray(xy, dtype=float), MATCH_RADIUS + 1e-6)
        return [self._keys[i] for i in np.unique(self._owner[hits])]

    def matched(self, xy: np.ndarray, heading: float) -> Set[str]:
        """The lanes a vehicle at ``xy`` heading ``heading`` is on."""
        out = set()
        for key in self._within(xy):
            pts = self.points[key]
            d = np.linalg.norm(pts - xy, axis=1)
            i = int(np.argmin(d))
            if d[i] <= MATCH_RADIUS and abs(_angle(self.direction[key][i] - heading)) <= MATCH_ALIGN:
                out.add(key)
        return out

    def best(self, xy: np.ndarray, heading: float) -> Optional[str]:
        """The one lane a vehicle at ``xy`` heading ``heading`` is on (see
        route_lanes), or None."""
        best, score = None, np.inf
        for key in self._within(xy):
            pts = self.points[key]
            d = np.linalg.norm(pts - xy, axis=1)
            i = int(np.argmin(d))
            off = abs(_angle(self.direction[key][i] - heading))
            if d[i] <= MATCH_RADIUS and off <= MATCH_ALIGN and d[i] + HEADING_WEIGHT * off < score:
                best, score = key, d[i] + HEADING_WEIGHT * off
        return best

    def _with_sides(self, lanes: Iterable[str]) -> Set[str]:
        lanes = set(lanes)
        return lanes | {s for lane in lanes for s in self.sides.get(lane, [])}

    def onward(self, start: Iterable[str], distance: float) -> Set[str]:
        """``start`` and every lane reachable from it through exit lanes,
        until ``distance`` metres of lanes lie behind a branch (``start``
        itself not counted), each with its side neighbours (as leaves)."""
        chain = set(start)
        queue = deque((lane, 0.0) for lane in chain)
        while queue:
            lane, travelled = queue.popleft()
            if travelled >= distance:
                continue
            for nxt in self.exits.get(lane, []):
                if nxt not in chain:
                    chain.add(nxt)
                    queue.append((nxt, travelled + self.length[nxt]))
        return self._with_sides(chain)

    def near(self, points: np.ndarray, lanes: Iterable[str], radius: float) -> np.ndarray:
        """Whether each point [N, 2] lies within ``radius`` of one of the
        lanes' centrelines (to the ~1 m spaced points: up to 0.5 m more)."""
        lanes = [lane for lane in lanes if lane in self.points]
        if not lanes or len(points) == 0:
            return np.zeros(len(points), dtype=bool)
        ref = np.concatenate([self.points[lane] for lane in lanes]).astype(np.float32)
        q = np.asarray(points, dtype=float).reshape(-1, 2).astype(np.float32)
        d = np.concatenate([np.sqrt(((c[:, None, :] - ref[None]) ** 2).sum(-1)).min(-1)
                            for c in np.array_split(q, max(1, len(q) // 512))])
        return d <= radius


_GRAPHS: Dict[int, Tuple[Dict, LaneGraph]] = {}


def lane_graph(scene: Scene) -> LaneGraph:
    """The scene's lane graph, built once per map. The cache holds the map
    itself, so its id cannot be reused by another scene's map while cached."""
    key = id(scene.map_features)
    if key not in _GRAPHS or _GRAPHS[key][0] is not scene.map_features:
        if len(_GRAPHS) > 8:
            _GRAPHS.clear()
        _GRAPHS[key] = (scene.map_features, LaneGraph(scene.map_features))
    return _GRAPHS[key][1]


def route_lanes(scene: Scene, agent: int, step: int, beyond: float = 100.0) -> Optional[Set[str]]:
    """The lanes of the agent's logged route from ``step`` on (see the module
    docstring), or None when no lane matches it anywhere."""
    graph = lane_graph(scene)
    logged = _logged_lanes(scene, agent, step)
    if logged is None:
        return None
    along: Dict[str, float] = {}  # lane -> metres driven matched to it
    order: List[str] = []  # lanes in the order they were first matched
    prev = None
    for xy, lane in logged:
        if lane is not None:
            along[lane] = along.get(lane, 0.0) + (0.0 if prev is None else float(np.linalg.norm(xy - prev)))
            if lane not in order:
                order.append(lane)
        prev = xy
    now = logged[0][1]
    route = [lane for lane in order if lane == now or along[lane] >= min(MIN_ALONG, 0.5 * graph.length[lane])]
    return graph._with_sides(route) | graph.onward([route[-1]], beyond)


def _logged_lanes(scene: Scene, agent: int, step: int):
    """[(position, its lane or None)] for the agent's logged positions from
    ``step`` on, or None when the map does not cover them (see the module
    docstring)."""
    graph = lane_graph(scene)
    steps = [t for t in range(step, scene.n_steps) if scene.valid[agent, t]]
    out = [(scene.position[agent, t, :2], graph.best(scene.position[agent, t, :2], float(scene.heading[agent, t])))
           for t in steps]
    on = [lane is not None for _, lane in out]
    if not out or not on[0] or not on[-1] or np.mean(on) < MIN_ON_LANES:
        return None
    return out


def reachable_lanes(scene: Scene, agent: int, step: int, reach: float = 200.0) -> Optional[Set[str]]:
    """The lanes an agent at ``step`` can reach within ``reach`` metres, or
    None when it is on no lane."""
    if _logged_lanes(scene, agent, step) is None:
        return None
    graph = lane_graph(scene)
    start = graph.matched(scene.position[agent, step, :2], float(scene.heading[agent, step]))
    return graph.onward(start, reach) if start else None
