"""Scenes from MetaDrive scenario descriptions (CAT's ``raw_scenes_500/*.pkl``)
and their conversion to the WOMD tf.Example layout DenseTNT reads, at any
context step.

``AdvGenerator._parse`` builds that layout once per scene, at the log's
native current step (10) and with the self-driving car and the adversary as
the predicted pair. Responsibility needs it for any agent, at any step of
the clip, and with agents left out (the counterfactual scene without the
queried agent), so the conversion is re-implemented here with the step, the
predicted pair and the agent list as arguments. At step 10 with CAT's agent
order it reproduces ``_parse`` exactly (checked by
scripts/responsibility/verify_densetnt.py).
"""

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HISTORY_STEPS = 11  # 10 past steps + the current one, as DenseTNT was trained
FUTURE_STEPS = 80
MAX_AGENTS = 128
MAX_ROAD_POINTS = 20000
MAX_TRAFFIC_LIGHTS = 16

# MetaDrive -> WOMD type codes, as in advgen/adv_generator.py
AGENT_TYPES = dict(VEHICLE=1, PEDESTRIAN=2, CYCLIST=3, OTHERS=4)
MAP_TYPES = dict(
    LANE_FREEWAY=1, LANE_SURFACE_STREET=2, LANE_BIKE_LANE=3,
    ROAD_LINE_BROKEN_SINGLE_WHITE=6, ROAD_LINE_SOLID_SINGLE_WHITE=7, ROAD_LINE_SOLID_DOUBLE_WHITE=8,
    ROAD_LINE_BROKEN_SINGLE_YELLOW=9, ROAD_LINE_BROKEN_DOUBLE_YELLOW=10, ROAD_LINE_SOLID_SINGLE_YELLOW=11,
    ROAD_LINE_SOLID_DOUBLE_YELLOW=12, ROAD_LINE_PASSING_DOUBLE_YELLOW=13,
    ROAD_EDGE_BOUNDARY=15, ROAD_EDGE_MEDIAN=16, STOP_SIGN=17, CROSSWALK=18, SPEED_BUMP=19,
)
LIGHT_STATES = dict(
    LANE_STATE_UNKNOWN=0, LANE_STATE_ARROW_STOP=1, LANE_STATE_ARROW_CAUTION=2, LANE_STATE_ARROW_GO=3,
    LANE_STATE_STOP=4, LANE_STATE_CAUTION=5, LANE_STATE_GO=6, LANE_STATE_FLASHING_STOP=7,
    LANE_STATE_FLASHING_CAUTION=8,
)
VEHICLE = "VEHICLE"


@dataclass
class Scene:
    """One scenario as per-agent arrays over the clip's steps (10 Hz)."""

    scenario_id: str
    track_ids: List[str]
    types: List[str]
    position: np.ndarray  # [N, T, 3]
    heading: np.ndarray  # [N, T]
    velocity: np.ndarray  # [N, T, 2]
    length: np.ndarray  # [N, T]
    width: np.ndarray  # [N, T]
    height: np.ndarray  # [N, T]
    valid: np.ndarray  # [N, T] bool
    sdc: int  # index of the self-driving car
    objects_of_interest: List[int]
    map_features: Dict
    dynamic_map_states: Dict

    @property
    def n_agents(self) -> int:
        return len(self.track_ids)

    @property
    def n_steps(self) -> int:
        return self.position.shape[1]

    def index(self, track_id) -> int:
        return self.track_ids.index(str(track_id))

    def shape_at(self, agent: int, step: int) -> np.ndarray:
        """(length, width) of an agent, from ``step`` or its nearest valid step."""
        valid = np.nonzero(self.valid[agent])[0]
        if valid.size == 0:
            return np.array([4.5, 2.0])
        s = valid[np.argmin(np.abs(valid - step))]
        return np.array([self.length[agent, s], self.width[agent, s]])

    def with_track(self, agent: int, position: np.ndarray, heading: np.ndarray, velocity: np.ndarray,
                   valid: Optional[np.ndarray] = None) -> "Scene":
        """A copy with one agent's states replaced, e.g. by the trajectory a
        driving policy produced in simulation instead of the logged one.
        Arrays cover the whole clip ([T, 3] / [T, 2], [T], [T, 2])."""
        pos, head, vel, val = self.position.copy(), self.heading.copy(), self.velocity.copy(), self.valid.copy()
        position = np.asarray(position, dtype=pos.dtype)
        if position.shape[-1] == 2:
            position = np.concatenate([position, pos[agent, :, 2:3]], axis=-1)
        pos[agent], head[agent], vel[agent] = position, heading, velocity
        if valid is not None:
            val[agent] = valid
        return replace(self, position=pos, heading=head, velocity=vel, valid=val)

    @classmethod
    def from_description(cls, sd: Dict) -> "Scene":
        tracks = sd["tracks"]
        ids = [str(k) for k in tracks]
        states = [tracks[k]["state"] for k in tracks]

        def stack(key):
            return np.stack([np.asarray(s[key]) for s in states])

        meta = sd["metadata"]
        ooi = [ids.index(str(t)) for t in meta.get("objects_of_interest", []) if str(t) in ids]
        return cls(
            scenario_id=str(meta.get("scenario_id", sd.get("id", ""))),
            track_ids=ids,
            types=[tracks[k]["type"] for k in tracks],
            position=stack("position").astype(np.float32),
            heading=stack("heading").astype(np.float32),
            velocity=stack("velocity").astype(np.float32),
            length=stack("length").astype(np.float32),
            width=stack("width").astype(np.float32),
            height=stack("height").astype(np.float32),
            valid=stack("valid").astype(bool),
            sdc=ids.index(str(meta["sdc_id"])),
            objects_of_interest=ooi,
            map_features=sd["map_features"],
            dynamic_map_states=sd["dynamic_map_states"],
        )

    def drop(self, agents: Sequence[int]) -> "Scene":
        """A copy without ``agents`` (their tracks deleted; indices shift)."""
        drop = set(int(a) for a in agents)
        if self.sdc in drop:
            raise ValueError("the self-driving car cannot be dropped")
        keep = [i for i in range(self.n_agents) if i not in drop]
        new = {old: j for j, old in enumerate(keep)}
        arrays = {k: getattr(self, k)[keep] for k in
                  ("position", "heading", "velocity", "length", "width", "height", "valid")}
        return replace(self, track_ids=[self.track_ids[i] for i in keep], types=[self.types[i] for i in keep],
                       sdc=new[self.sdc], objects_of_interest=[new[i] for i in self.objects_of_interest if i in new],
                       **arrays)

    def to_description(self, dt: float = 0.1, dataset: str = "waymo") -> Dict:
        """The scene as a ScenarioNet / MetaDrive scenario description again
        (what ``from_description`` reads), e.g. for UniTraj; works for scenes
        rebuilt from rollouts too. Only what ``from_description`` keeps
        survives the round trip."""
        tracks = {}
        for i, tid in enumerate(self.track_ids):
            tracks[tid] = {
                "type": self.types[i],
                "state": {"position": self.position[i].copy(), "heading": self.heading[i].copy(),
                          "velocity": self.velocity[i].copy(), "length": self.length[i].copy(),
                          "width": self.width[i].copy(), "height": self.height[i].copy(),
                          "valid": self.valid[i].copy()},
                "metadata": {"track_length": self.n_steps, "type": self.types[i], "object_id": tid,
                             "dataset": dataset},
            }
        return {
            "id": self.scenario_id,
            "tracks": tracks,
            "map_features": self.map_features,
            "dynamic_map_states": self.dynamic_map_states,
            "metadata": {"scenario_id": self.scenario_id, "sdc_id": self.track_ids[self.sdc],
                         "objects_of_interest": [self.track_ids[i] for i in self.objects_of_interest],
                         "ts": np.arange(self.n_steps, dtype=np.float64) * dt, "dataset": dataset,
                         "track_length": self.n_steps},
        }

    @classmethod
    def load(cls, path) -> "Scene":
        with open(path, "rb") as f:
            return cls.from_description(pickle.load(f))


def scene_files(directory) -> List[Path]:
    """CAT's scene files (``0.pkl`` ... ``499.pkl``) in numeric order."""
    files = list(Path(directory).glob("*.pkl"))
    return sorted(files, key=lambda p: (int(p.stem) if p.stem.isdigit() else float("inf"), p.stem))


def sidecar_index(directory) -> Path:
    """Where an index of a scene folder is kept: <folder>.index.json next to
    it, since MetaDrive asserts that every file inside a scene folder is a
    scene (.pkl)."""
    directory = Path(directory)
    return directory.with_name(directory.name + ".index.json")


def cat_agent_order(scene: Scene) -> List[int]:
    """The agent order of ``AdvGenerator._parse``: the self-driving car, the
    other object of interest (the adversary), then the rest."""
    others = [i for i in scene.objects_of_interest if i != scene.sdc]
    first = [scene.sdc] + others[:1]
    return first + [i for i in range(scene.n_agents) if i not in first]


def _polyline_dir(polyline: np.ndarray) -> np.ndarray:
    if polyline.ndim == 1:
        return np.zeros(3)
    post = np.roll(polyline, shift=-1, axis=0)
    post[-1] = polyline[-1]
    diff = post - polyline
    return diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1e9)


def _map_arrays(scene: Scene) -> Dict[str, np.ndarray]:
    out = {
        "roadgraph_samples/dir": np.full([MAX_ROAD_POINTS, 3], -1, dtype=np.float32),
        "roadgraph_samples/id": np.full([MAX_ROAD_POINTS, 1], -1, dtype=np.int64),
        "roadgraph_samples/type": np.full([MAX_ROAD_POINTS, 1], -1, dtype=np.int64),
        "roadgraph_samples/valid": np.full([MAX_ROAD_POINTS, 1], 1, dtype=np.int64),
        "roadgraph_samples/xyz": np.full([MAX_ROAD_POINTS, 3], -1, dtype=np.float32),
    }
    count = 0
    for key, feature in scene.map_features.items():
        if feature["type"] not in MAP_TYPES:
            # WOMD v1.2's DRIVEWAY (type 20) and MetaDrive's UNKNOWN lines and edges: DenseTNT was
            # trained on v1.1 maps without them (and asserts types below 20), so it sees the same kinds
            continue
        kind = MAP_TYPES[feature["type"]]
        if kind == 17:
            poly = feature["position"]
        elif kind in (18, 19):
            poly = feature["polygon"]
        else:
            poly = feature["polyline"]
        poly = np.asarray(poly)
        direction = _polyline_dir(poly)
        # a stop sign is one [3] point; _parse writes it (broadcast) into
        # len(point) == 3 rows, which this keeps for identical inputs
        n = len(poly)
        take = min(n, MAX_ROAD_POINTS - count)
        out["roadgraph_samples/xyz"][count:count + take] = poly[:take] if poly.ndim > 1 else poly
        out["roadgraph_samples/dir"][count:count + take] = direction[:take] if direction.ndim > 1 else direction
        out["roadgraph_samples/id"][count:count + take] = int(key)
        out["roadgraph_samples/type"][count:count + take] = kind
        count += take
        if count >= MAX_ROAD_POINTS:
            break
    return out


def _traffic_light_arrays(scene: Scene, step: int) -> Dict[str, np.ndarray]:
    out = {
        "traffic_light_state/current/state": np.full([1, 16], -1, dtype=np.int64),
        "traffic_light_state/current/valid": np.full([1, 16], 0, dtype=np.int64),
        "traffic_light_state/current/id": np.full([1, 16], -1, dtype=np.int64),
        "traffic_light_state/current/x": np.full([1, 16], -1, dtype=np.float32),
        "traffic_light_state/current/y": np.full([1, 16], -1, dtype=np.float32),
        "traffic_light_state/current/z": np.full([1, 16], -1, dtype=np.float32),
        "traffic_light_state/past/state": np.full([10, 16], -1, dtype=np.int64),
        "traffic_light_state/past/valid": np.full([10, 16], 0, dtype=np.int64),
        "traffic_light_state/past/x": np.full([10, 16], -1, dtype=np.float32),
        "traffic_light_state/past/y": np.full([10, 16], -1, dtype=np.float32),
        "traffic_light_state/past/z": np.full([10, 16], -1, dtype=np.float32),
        "traffic_light_state/past/id": np.full([10, 16], -1, dtype=np.int64),
    }
    for i, light in enumerate(scene.dynamic_map_states.values()):
        if i == MAX_TRAFFIC_LIGHTS:
            break
        if light["type"] != "TRAFFIC_LIGHT":
            continue
        states = light["state"]["object_state"]
        for slot, t in [("past", j) for j in range(10)] + [("current", 10)]:
            source = step - 10 + t
            state = states[source] if 0 <= source < len(states) else None
            if not state:
                continue
            row = t if slot == "past" else 0
            prefix = f"traffic_light_state/{slot}/"
            out[prefix + "state"][row][i] = LIGHT_STATES[state]
            out[prefix + "valid"][row][i] = 1
            out[prefix + "id"][row][i] = int(light["lane"])
            out[prefix + "x"][row][i] = light["stop_point"][0]
            out[prefix + "y"][row][i] = light["stop_point"][1]
            out[prefix + "z"][row][i] = light["stop_point"][2]
    return out


def womd_features(scene: Scene, step: int, order: Sequence[int]) -> Dict[str, np.ndarray]:
    """The WOMD tf.Example fields DenseTNT reads, with ``step`` as the
    current step: the 10 steps before it are the past, the 80 after it the
    future (invalid beyond the clip). ``order`` lists the agents to include,
    the first two being the predicted pair; agents not listed are absent from
    the scene altogether."""
    order = list(order)[:MAX_AGENTS]
    if len(order) < 2:
        raise ValueError("DenseTNT's pair model needs at least two agents in the scene")
    n_t = scene.n_steps

    def window(first, count):
        idx = np.arange(first, first + count)
        inside = (idx >= 0) & (idx < n_t)
        return np.clip(idx, 0, n_t - 1), inside

    fields = {}
    for slot, (first, count) in {"past": (step - 10, 10), "current": (step, 1), "future": (step + 1, FUTURE_STEPS)}.items():
        idx, inside = window(first, count)
        agents = np.asarray(order)

        def pad(values, fill, dtype):
            full = np.full((MAX_AGENTS, count), fill, dtype=dtype)
            full[: len(agents)] = np.where(inside[None], values[agents][:, idx], fill)
            return full

        vel = scene.velocity[agents][:, idx]
        fields.update({
            f"state/{slot}/x": pad(scene.position[..., 0], -1, np.float32),
            f"state/{slot}/y": pad(scene.position[..., 1], -1, np.float32),
            f"state/{slot}/z": pad(scene.position[..., 2], -1, np.float32),
            f"state/{slot}/bbox_yaw": pad(scene.heading, -1, np.float32),
            f"state/{slot}/velocity_x": pad(scene.velocity[..., 0], -1, np.float32),
            f"state/{slot}/velocity_y": pad(scene.velocity[..., 1], -1, np.float32),
            f"state/{slot}/vel_yaw": _pad_rows(np.where(inside[None], np.arctan2(vel[..., 1], vel[..., 0]), -1),
                                               count, -1, np.float32),
            f"state/{slot}/width": pad(scene.width, -1, np.float32),
            f"state/{slot}/height": pad(scene.height, -1, np.float32),
            f"state/{slot}/length": pad(scene.length, -1, np.float32),
            f"state/{slot}/valid": pad(scene.valid.astype(np.int64), 0, np.int64),
        })

    ids = np.full([MAX_AGENTS], -1, dtype=np.int64)
    ids[: len(order)] = [int(scene.track_ids[i]) for i in order]
    types = np.full([MAX_AGENTS], 0, dtype=np.int64)
    types[: len(order)] = [AGENT_TYPES.get(scene.types[i], 4) for i in order]
    predict = np.zeros([MAX_AGENTS], dtype=np.int64)
    predict[:2] = 1
    is_sdc = np.zeros([MAX_AGENTS], dtype=np.int64)
    if scene.sdc in order:
        is_sdc[order.index(scene.sdc)] = 1
    fields.update({
        "state/id": ids,
        "state/type": types,
        "state/is_sdc": is_sdc,
        "state/tracks_to_predict": predict,
        "state/objects_of_interest": predict.copy(),
        "scenario/id": np.array(["template"]),
    })
    fields.update(_map_arrays(scene))
    fields.update(_traffic_light_arrays(scene, step))
    return fields


def _pad_rows(values: np.ndarray, count: int, fill, dtype) -> np.ndarray:
    full = np.full((MAX_AGENTS, count), fill, dtype=dtype)
    full[: len(values)] = values
    return full
