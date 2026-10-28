"""Who caused a collision, by Responsibility-Sensitive Safety (RSS;
Shalev-Shwartz, Shammah & Shashua, "On a Formal Model of Safe and Scalable
Self-driving Cars", arXiv:1708.06374): the rule-based baseline for the
counterfactual attribution of responsibility/blame.py.

RSS calls a time *dangerous* for two cars when they are closer than the safe
distance both longitudinally (Lemma 2: the rear car must be able to stop
behind a front car braking at up to brake_max, after a response time rho in
which it may still accelerate) and laterally (Lemma 4: after rho of lateral
acceleration toward each other and lateral braking, at least mu apart).
The *danger threshold* t_b is where the dangerous stretch that ends in the
collision began (Def. 9: the later of the longitudinal and the lateral
threshold). From t_b on both cars owe the *proper response* of the axis
that became dangerous last (Def. 10):

  longitudinal (Def. 4)  the rear car accelerates at most accel_max for rho,
                         then brakes at least brake_min until it stops; the
                         front car brakes at most brake_max
  lateral (Def. 8)       both accelerate laterally toward each other at most
                         lat_accel_max for rho, then brake laterally at least
                         lat_brake_min until their lateral velocity is 0

A car that did not comply is responsible: verdict "ego" / "other", "shared"
when neither complied, "n/a" when both did (the parameters did not
describe the collision) or RSS says nothing here. Both definitions constrain
speeds, and compliance is checked on them: from t_b on, a car's speed (on
the axis concerned, toward the other) may not exceed the envelope of its
proper response -- rising at most at the allowed acceleration for rho, then
falling at least at the owed braking to 0 -- nor the front car's fall below
braking at brake_max, up to a slack for the noise of estimated speeds.
Speeds rather than accelerations differenced from 10 Hz states, which are
far noisier.

Scope and approximations:
  * Same-direction pairs only (headings within 30 degrees before the
    collision): a multi-lane road with one geometry (paper Sec. 3.5), which
    covers rear-end, cut-in and side-swipe collisions. Oncoming and crossing
    collisions need the lane graph (which lane runs which way, which route
    has priority, Sec. 3.7) and get "n/a"; ``coverage`` in the comparisons
    says how many collisions that leaves.
  * The lane frame at every step is the direction of the nearest lane
    centreline (within 4 m of the two cars' midpoint, running their way
    within 45 degrees), so curves do not register as lateral motion and the
    frame does not turn with a car that leaves its lane; where no lane is
    near, the bisector of the two headings stands in. Longitudinal and
    lateral velocities are the logged (or simulated) velocities projected
    onto it.
  * When the cars were already dangerous at the first step both are seen,
    the true threshold lies before the clip; the first step stands in for
    it and both axes' responses apply (Def. 10 with t_long_b = t_lat_b).
    The case is then marked "/first-step".
  * One response time for both cars (the paper's; ad-rss-lib defaults to
    1 s for the ego and 2 s for others, which would judge the two sides by
    different standards). The other values are ad-rss-lib's defaults.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

DT = 0.1  # s per step (responsibility.interaction.DT in the CAT fork)
from responsibility.scene import Scene

SAME_DIRECTION = np.deg2rad(30.0)  # rad: headings closer than this share a lane frame
LANE_RADIUS = 4.0  # m: lane centreline points considered for the lane frame
LANE_ALIGN = np.deg2rad(45.0)  # rad: a lane runs the cars' way within this
LANE_TYPES = ("LANE_FREEWAY", "LANE_SURFACE_STREET")


@dataclass(frozen=True)
class RssParams:
    rho: float = 1.0  # s: response time (both cars)
    accel_max: float = 3.5  # m/s^2: longitudinal acceleration during the response time
    brake_min: float = 4.0  # m/s^2: the braking the rear car owes
    brake_max: float = 8.0  # m/s^2: the hardest braking the front car may do
    lat_accel_max: float = 0.2  # m/s^2
    lat_brake_min: float = 0.8  # m/s^2
    mu: float = 0.1  # m: lateral fluctuation margin
    lon_slack: float = 0.5  # m/s: longitudinal speed noise tolerated by the compliance check
    lat_slack: float = 0.2  # m/s: lateral one


@dataclass
class RssBlame:
    verdict: str  # "ego", "other", "shared" or "n/a"
    case: str  # "lon", "lat", "lon+lat" (+ "/first-step"), or why there is no verdict
    danger_step: Optional[int] = None  # t_b
    ego_proper: Optional[bool] = None
    other_proper: Optional[bool] = None

    def as_row(self) -> Dict:
        return {"rss": self.verdict, "rss_case": self.case}


def safe_longitudinal(v_rear, v_front, p: RssParams):
    """Lemma 2: the minimal safe gap from the rear car's front to the front
    car's rear (speeds along the lane, >= 0)."""
    v_rho = v_rear + p.rho * p.accel_max
    d = v_rear * p.rho + 0.5 * p.accel_max * p.rho ** 2 + v_rho ** 2 / (2 * p.brake_min) \
        - v_front ** 2 / (2 * p.brake_max)
    return np.maximum(d, 0.0)


def _lateral_reach(u, p: RssParams):
    """How far a car closes in laterally under Def. 6's worst case: rho at
    lat_accel_max toward the other, then lat_brake_min until its lateral
    velocity is 0. u: its lateral velocity toward the other. The braking
    distance is signed, so a car moving away keeps moving away."""
    u_rho = u + p.rho * p.lat_accel_max
    return (u + u_rho) / 2 * p.rho + u_rho * np.abs(u_rho) / (2 * p.lat_brake_min)


def safe_lateral(u1, u2, p: RssParams):
    """Lemma 4 in terms of the two cars' lateral velocities toward each
    other: the minimal safe gap between their facing sides."""
    return p.mu + np.maximum(_lateral_reach(u1, p) + _lateral_reach(u2, p), 0.0)


def max_speed(v0: float, tau: np.ndarray, accel: float, brake: float, rho: float) -> np.ndarray:
    """The highest speed a car at v0 may have tau after the danger
    threshold under a proper response: rising at most at ``accel`` for rho,
    then falling at least at ``brake`` to 0, and at most 0 from then on (a
    car already moving away after rho may keep doing so)."""
    v1 = v0 + accel * rho
    after = np.maximum(v1 - brake * (tau - rho), 0.0) if v1 > 0 else np.zeros_like(tau)
    return np.where(tau <= rho, v0 + accel * tau, after)


def min_speed(v0: float, tau: np.ndarray, brake: float) -> np.ndarray:
    """The lowest speed a front car at v0 >= 0 may have tau later, braking at
    most at ``brake``."""
    return np.maximum(v0 - brake * tau, 0.0)


def lane_segments(scene: Scene):
    """The segments of the scene's lane centrelines: starts [M, 2], vectors
    [M, 2] and directions [M]."""
    starts, vectors = [], []
    for feature in scene.map_features.values():
        poly = np.asarray(feature.get("polyline", []), dtype=float)
        if feature.get("type") not in LANE_TYPES or poly.ndim != 2 or len(poly) < 2:
            continue
        starts.append(poly[:-1, :2])
        vectors.append(np.diff(poly[:, :2], axis=0))
    if not starts:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    starts, vectors = np.concatenate(starts), np.concatenate(vectors)
    return starts, vectors, np.arctan2(vectors[:, 1], vectors[:, 0])


def lane_frame(segments, where: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Per step, the direction of the nearest lane segment within
    LANE_RADIUS of ``where`` [S, 2] running within LANE_ALIGN of
    ``heading`` [S]; the heading itself where there is none."""
    starts, vectors, angles = segments
    out = heading.copy()
    if len(starts) == 0:
        return out
    length2 = np.maximum(np.sum(vectors ** 2, -1), 1e-9)
    for i, (xy, h) in enumerate(zip(where, heading)):
        along = np.clip(np.sum((xy - starts) * vectors, -1) / length2, 0.0, 1.0)
        dist = np.linalg.norm(starts + along[:, None] * vectors - xy, axis=-1)
        ok = (dist <= LANE_RADIUS) & (np.abs(_angle(angles - h)) <= LANE_ALIGN)
        if np.any(ok):
            out[i] = angles[np.flatnonzero(ok)[np.argmin(dist[ok])]]
    return out


def _trailing_run(flags: np.ndarray) -> int:
    """Index where the run of True values that ends the array starts."""
    i = len(flags)
    while i > 0 and flags[i - 1]:
        i -= 1
    return i


def _angle(a):
    return np.angle(np.exp(1j * a))


def rss_blame(scene: Scene, ego: int, other: int, crash_step: int,
              params: Optional[RssParams] = None) -> RssBlame:
    """RSS's verdict on a collision of ``ego`` with ``other`` at
    ``crash_step`` (module docstring)."""
    p = params or RssParams()
    both = scene.valid[ego] & scene.valid[other]
    last = next((k for k in range(min(crash_step, scene.n_steps - 1), -1, -1) if both[k]), None)
    if last is None:
        return RssBlame("n/a", "not-seen-together")
    first = last
    while first > 0 and both[first - 1]:
        first -= 1
    steps = np.arange(first, last + 1)
    if steps.size < 2:
        return RssBlame("n/a", "too-short")
    before = steps[-2] if steps[-1] == crash_step else steps[-1]  # geometry before contact
    if abs(_angle(float(scene.heading[other, before]) - float(scene.heading[ego, before]))) > SAME_DIRECTION:
        return RssBlame("n/a", "not-same-direction")

    h_e, h_o = scene.heading[ego, steps].astype(float), scene.heading[other, steps].astype(float)
    middle = (scene.position[ego, steps, :2] + scene.position[other, steps, :2]).astype(float) / 2
    frame = lane_frame(lane_segments(scene), middle, h_e + _angle(h_o - h_e) / 2)
    t_hat = np.stack([np.cos(frame), np.sin(frame)], -1)
    n_hat = np.stack([-np.sin(frame), np.cos(frame)], -1)
    d = (scene.position[other, steps, :2] - scene.position[ego, steps, :2]).astype(float)
    lon = np.sum(d * t_hat, -1)  # where the other is, along and across the lane
    lat = np.sum(d * n_hat, -1)

    def extents(agent, heading):
        theta = heading - frame
        length, width = scene.length[agent, steps].astype(float), scene.width[agent, steps].astype(float)
        return (length / 2 * np.abs(np.cos(theta)) + width / 2 * np.abs(np.sin(theta)),
                length / 2 * np.abs(np.sin(theta)) + width / 2 * np.abs(np.cos(theta)))

    (lon_e, lat_e), (lon_o, lat_o) = extents(ego, h_e), extents(other, h_o)
    gap_lon = np.abs(lon) - (lon_e + lon_o)
    gap_lat = np.abs(lat) - (lat_e + lat_o)
    v_e, v_o = scene.velocity[ego, steps].astype(float), scene.velocity[other, steps].astype(float)
    vlon_e, vlon_o = np.sum(v_e * t_hat, -1), np.sum(v_o * t_hat, -1)
    vlat_e, vlat_o = np.sum(v_e * n_hat, -1), np.sum(v_o * n_hat, -1)

    ego_behind = lon > 0
    v_rear = np.maximum(np.where(ego_behind, vlon_e, vlon_o), 0.0)
    v_front = np.maximum(np.where(ego_behind, vlon_o, vlon_e), 0.0)
    side = np.where(lat >= 0, 1.0, -1.0)  # +1: the other lies on the ego's left (+n side)
    toward_e, toward_o = vlat_e * side, -vlat_o * side
    danger_lon = gap_lon < safe_longitudinal(v_rear, v_front, p)
    danger_lat = gap_lat < safe_lateral(toward_e, toward_o, p)
    danger_lon[-1] = danger_lat[-1] = True  # the collision itself is dangerous by definition

    i_lon, i_lat = _trailing_run(danger_lon), _trailing_run(danger_lat)
    i_b = max(i_lon, i_lat)
    lon_case, lat_case = i_b == i_lon, i_b == i_lat
    case = "+".join(name for name, on in (("lon", lon_case), ("lat", lat_case)) if on)
    if i_b == 0:  # dangerous since the cars were first seen together
        lon_case = lat_case = True
        case = "lon+lat/first-step"

    tau = np.arange(len(steps) - i_b) * DT
    ego_ok = other_ok = True
    if lon_case:
        rear_is_ego = bool(ego_behind[i_b])
        v_r = np.where(rear_is_ego, vlon_e, vlon_o)[i_b:]
        v_f = np.where(rear_is_ego, vlon_o, vlon_e)[i_b:]
        rear_ok = bool(np.all(v_r <= max_speed(max(v_r[0], 0.0), tau, p.accel_max, p.brake_min, p.rho) + p.lon_slack))
        front_ok = bool(np.all(v_f >= min_speed(max(v_f[0], 0.0), tau, p.brake_max) - p.lon_slack))
        ego_ok &= rear_ok if rear_is_ego else front_ok
        other_ok &= front_ok if rear_is_ego else rear_ok
    if lat_case:
        s = side[i_b]  # who is left of whom at the threshold
        for name, toward in (("ego", vlat_e[i_b:] * s), ("other", -vlat_o[i_b:] * s)):
            ok = bool(np.all(toward <= max_speed(toward[0], tau, p.lat_accel_max, p.lat_brake_min, p.rho)
                             + p.lat_slack))
            if name == "ego":
                ego_ok &= ok
            else:
                other_ok &= ok

    if ego_ok and other_ok:
        verdict = "n/a"
        case += "/no-violation"
    else:
        verdict = "shared" if not ego_ok and not other_ok else "ego" if not ego_ok else "other"
    return RssBlame(verdict, case, int(steps[i_b]), ego_ok, other_ok)


def rollout_rss(scene: Scene, rollout: Dict, params: Optional[RssParams] = None) -> Optional[RssBlame]:
    """rss_blame for a rollout that ended in a vehicle collision, in the
    scene rebuilt from it, with the partner blame.rollout_blame uses; None
    after the end of the log or with nobody to have collided with."""
    from responsibility.blame import rollout_collision

    collision = rollout_collision(scene, rollout)
    if collision is None:
        return None
    step, other = collision
    return rss_blame(scene, scene.sdc, other, step, params)


def rss_weight(rss: Optional[RssBlame]) -> float:
    """The share of a collision penalty the ego keeps under RSS: none when
    RSS puts the collision on the other alone, all of it otherwise (doubt
    keeps the penalty, as in blame_reward.penalty_weight)."""
    return 0.0 if rss is not None and rss.verdict == "other" else 1.0
