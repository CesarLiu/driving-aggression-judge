"""Screen WOMD-style scenes for right-of-way violations at intersections without a working signal.

For every scene it takes the self-driving car (SDC) and the other object of interest (OOI), or a
pair you name, finds where their paths cross, decides who had the right of way under a
jurisdiction profile (``jurisdictions/<id>.json``) and writes one row per scene to a CSV: who went
first, who held priority, and so whether the SDC or the OOI violated the other's right of way.

A violation is a *candidate*: the reviewer still looks at the scene (see SKILL.md). Cut-ins,
tailgating and close passes are not screened here (no right of way is involved); judge them with
rubric.md.

Input is a folder of scene pickles, or one file: ScenarioNet/MetaDrive (CAT's ``raw_scenes_500/*.pkl``)
or catk's cached WOMD (``catk/data/womd_cache/<split>/*.pkl``; see scenes.py for what that format lacks).
It reads them with the copy of the ``responsibility`` package next to this script (the scene
arrays, the lane graph, the conflict finder). Convert a WOMD tfrecord to ScenarioNet scenes first.

    python screen_crossing.py --scenes raw_scenes_500 --profile us-california --out screen.csv
"""

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jurisdiction  # noqa: E402
from scenes import PAIR_MODES, load_scene, pick_pair  # noqa: E402
from responsibility import lanes as lanes_mod  # noqa: E402
from responsibility import right_of_way as R  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

DT = 0.1
SIGNAL_CASES_PASSED_THROUGH = ("following", "same-direction", "driveway", "oncoming", "signal-unknown",
                               "no-conflict", "too-short", "not-seen-together")


def scene_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    files = list(path.glob("*.pkl"))
    return sorted(files, key=lambda p: (int(p.stem) if p.stem.isdigit() else float("inf"), p.stem))


def arc_and_speed(scene, i):
    idx = np.flatnonzero(scene.valid[i])
    pos = scene.position[i, idx, :2].astype(float)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    return idx, arc, np.linalg.norm(scene.velocity[i, idx], axis=1)


def time_at(idx, arc, s):
    """Time (s) the agent has covered s metres of its path; NaN if never within the clip."""
    return float(idx[np.argmax(arc >= s)]) * DT if (arc >= s).any() else float("nan")


def stop_line_arc(R, scene, graph, stop_lanes, agent, path):
    """Arc length at which the agent's path meets the stop line of the stop-controlled lane it uses."""
    best = None
    for lane in stop_lanes:
        try:
            s, dist = path.project(graph.points[lane][0])
        except KeyError:
            continue
        if dist < 2.5 and (best is None or dist < best[1]):
            best = (s, dist)
    return None if best is None else best[0]


def smooth(v, steps=5):
    """Speed averaged over ``steps`` samples (0.5 s): WOMD speeds jitter enough to fake 10+ m/s² spikes."""
    return np.convolve(np.pad(v, (steps // 2, steps // 2), mode="edge"), np.ones(steps) / steps, mode="valid")


def sustained_decel(v, window=10):
    """Per step, the speed lost over the next ``window`` steps (1 s) per second: braking that lasts,
    not a one-frame speed glitch (WOMD speeds jump by 1-2 m/s between frames)."""
    v = np.asarray(v, dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) > window:
        out[:-window] = (v[:-window] - v[window:]) / (window * DT)
    return out


def victim_impact(idx, arc, v, s_star, t_x, pet, victim_stops, hard_brake, close_pet):
    """Whether the vehicle that was gone ahead of (the victim) paid for it when the other passed at t_x.

    Returns (speed at t_x, hardest braking in m/s², affected, why), or None if the victim never
    reaches the conflict point within the clip. Waiting is not a cost for a victim that faces a stop
    sign: it had to stop anyway. The victim paid if
      * it was made to stop again: it had stopped, got going (>= 2 m/s) and then had to stop (< 1 m/s);
      * it braked hard (>= ``hard_brake``, sustained over 1 s) without a stop sign of its own; or
      * it was cut close: it reached the conflict within ``close_pet`` s of the other while moving.
    """
    if not (arc >= s_star).any():
        return None
    t, vs = idx * DT, smooth(v)
    t_y = t[np.argmax(arc >= s_star)]
    speed = float(np.interp(t_x, t, vs))
    w = (t >= t_x - 3.0) & (t <= t_y)
    decel = sustained_decel(vs)
    max_dec = float(np.nanmax(decel[w])) if w.any() and np.isfinite(decel[w]).any() else 0.0
    vw = vs[w]
    restop = False
    if vw.size:
        k = int(np.argmax(vw))
        restop = vw[k] >= 2.0 and vw[:k + 1].min() < 1.0 and vw[k:].min() < 1.0
    why = [name for name, hit in (("made_to_stop_again", restop),
                                  ("hard_braking", not victim_stops and max_dec >= hard_brake),
                                  ("cut_close", pet == pet and pet <= close_pet and speed >= 2.0)) if hit]
    return speed, max_dec, bool(why), "+".join(why)


def track_turn(scene, i, idx, arc, s_star, R):
    """The turn the agent actually made, from its own heading 20 m before and after the conflict point
    (the lane-based turn of right_of_way.py is sometimes wrong where lanes fan out)."""
    if arc[-1] < s_star + 10.0:  # the turn is not finished within the clip: keep the lane-based one
        return None
    kb = int(np.argmax(arc >= max(s_star - 20.0, 0.0)))
    ka = int(np.argmax(arc >= s_star + 20.0)) if arc[-1] >= s_star + 20.0 else len(arc) - 1
    if ka <= kb:
        return None
    d = (scene.heading[i, idx[ka]] - scene.heading[i, idx[kb]] + math.pi) % (2 * math.pi) - math.pi
    return ("u-turn" if abs(d) > R.U_TURN else "left" if d > R.TURN else "right" if d < -R.TURN else "straight")


def screen_pair(R, lanes_mod, scene, a, o, profile, tie_steps, hard_brake=3.0, close_pet=1.5):
    params = R.RightOfWayParams()
    pr = R.priority(scene, a, o, scene.n_steps - 1, params)
    row = {"case": pr.case, "holder": "none", "went_first": "", "pet_s": "", "sdc_violation": 0,
           "ooi_violation": 0, "victim_speed_at_pass": "", "victim_max_decel": "", "affected": "",
           "affected_by": "", "sdc_aggressive_candidate": 0, "ooi_aggressive_candidate": 0, "confidence": "",
           "flags": ""}
    if pr.conflict is None or pr.approaches is None:
        return row
    A, B = pr.approaches
    ia, arc_a, v_a = arc_and_speed(scene, a)
    io, arc_o, v_o = arc_and_speed(scene, o)
    flags = []
    for role, i, idx, arc, s_star in (("sdc", a, ia, arc_a, pr.conflict.s[0]), ("ooi", o, io, arc_o, pr.conflict.s[1])):
        lane_turn = A.turn if role == "sdc" else B.turn
        actual = track_turn(scene, i, idx, arc, s_star, R)
        if actual is not None and actual != lane_turn:  # trust the track over the lane match
            flags.append("%s_turn_from_track=%s(lanes:%s)" % (role, actual, lane_turn))
            if role == "sdc":
                A = dataclasses.replace(A, turn=actual)
            else:
                B = dataclasses.replace(B, turn=actual)
    row.update(conflict_x=round(float(pr.conflict.point[0]), 1), conflict_y=round(float(pr.conflict.point[1]), 1),
               conflict=pr.conflict.kind, angle_deg=round(math.degrees(pr.conflict.angle), 1),
               sdc_control=A.control, ooi_control=B.control, sdc_turn=A.turn, ooi_turn=B.turn)
    holder = {a: "sdc", o: "ooi", None: "none"}[pr.holder]
    case = pr.case
    if pr.case not in SIGNAL_CASES_PASSED_THROUGH and jurisdiction.handles(A, B):
        h, case = jurisdiction.decide(profile, A, B, tie_steps)
        holder = {"a": "sdc", "b": "ooi", None: "none"}[h]
    row["case"], row["holder"] = case, holder

    t_a, t_o = time_at(ia, arc_a, pr.conflict.s[0]), time_at(io, arc_o, pr.conflict.s[1])
    if t_a == t_a and t_o == t_o:
        row["went_first"] = "sdc" if t_a < t_o else "ooi" if t_o < t_a else "tie"
        row["pet_s"] = round(abs(t_o - t_a), 1)
    elif t_a == t_a or t_o == t_o:
        row["went_first"] = "sdc" if t_a == t_a else "ooi"
        flags.append("other_never_crossed_in_clip")
    sdc_viol = holder == "ooi" and row["went_first"] == "sdc"
    ooi_viol = holder == "sdc" and row["went_first"] == "ooi"
    row["sdc_violation"], row["ooi_violation"] = int(sdc_viol), int(ooi_viol)
    if sdc_viol or ooi_viol:  # a violation only matters if the vehicle gone ahead of paid for it
        pet = abs(t_o - t_a) if t_a == t_a and t_o == t_o else float("nan")
        if sdc_viol:
            victim = (io, arc_o, v_o, pr.conflict.s[1], t_a, pet, B.control in ("stop", "red"))
        else:
            victim = (ia, arc_a, v_a, pr.conflict.s[0], t_o, pet, A.control in ("stop", "red"))
        impact = victim_impact(*victim, hard_brake, close_pet)
        if impact is None:
            flags.append("victim_never_reached_conflict")
        else:
            row["victim_speed_at_pass"], row["victim_max_decel"] = round(impact[0], 1), round(impact[1], 1)
            row["affected"], row["affected_by"] = int(impact[2]), impact[3]
            row["sdc_aggressive_candidate"] = int(sdc_viol and impact[2])
            row["ooi_aggressive_candidate"] = int(ooi_viol and impact[2])

    if "order-unknown" in case:
        flags.append("both_waiting_at_clip_start")
    if A.control == "stop" and B.control == "stop" and (A.arrived == -1 or B.arrived == -1):
        flags.append("arrival_censored")
    graph = lanes_mod.lane_graph(scene)
    stop_lanes, _ = R._stop_signs(scene)
    for role, agent, arc, v in (("sdc", a, arc_a, v_a), ("ooi", o, arc_o, v_o)):
        s_line = stop_line_arc(R, scene, graph, stop_lanes, agent, R._Path(scene, agent, scene.n_steps - 1, params.extend))
        if s_line is None:
            continue
        k = int(np.argmax(arc >= s_line)) if (arc >= s_line).any() else None
        if k is None:
            continue
        flags.append("%s_v_at_step0=%.1f" % (role, v[0]))
        if s_line > 3.0 and v[:k + 1].min() >= 1.5:  # an observed approach that never slowed
            flags.append("%s_rolled_stop_observed" % role)
    row["flags"] = ";".join(flags)
    if sdc_viol or ooi_viol:
        row["confidence"] = ("low" if "other_never_crossed_in_clip" in flags
                             else "medium" if "both_waiting_at_clip_start" in flags else "high")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="folder of scene pickles, or one file")
    ap.add_argument("--profile", default="us-california", help="profile id or path: %s" % ", ".join(jurisdiction.list_profiles()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", help="track id of the first agent (default: the SDC); needs --other")
    ap.add_argument("--other", help="track id of the second agent (default: the other object of interest)")
    ap.add_argument("--pairs", choices=PAIR_MODES, default="auto",
                    help="which pair to judge when --agent is not given (see scenes.pick_pair)")
    ap.add_argument("--tie-seconds", type=float, default=1.0, help="arrivals this close count as simultaneous")
    ap.add_argument("--hard-brake", type=float, default=3.0,
                    help="m/s²: braking this hard by a vehicle without a stop sign of its own is a cost forced on it")
    ap.add_argument("--close-pet", type=float, default=1.5,
                    help="s: reaching the conflict this soon after the other, while moving, is being cut close")
    args = ap.parse_args()

    profile = jurisdiction.load_profile(args.profile)
    if profile["status"].startswith("draft"):
        print("WARNING: profile %s is a draft: %s" % (profile["id"], profile["status"]), file=sys.stderr)
    tie_steps = int(round(args.tie_seconds / DT))
    rows, skipped = [], 0
    for path in scene_files(args.scenes):
        try:
            scene, fmt = load_scene(path, Scene)
            if args.agent:
                a, o = scene.index(args.agent), scene.index(args.other)
                how = "named"
            else:
                pair = pick_pair(scene, args.pairs)
                if pair is None:
                    skipped += 1
                    continue
                a, o, how = pair
            row = screen_pair(R, lanes_mod, scene, a, o, profile, tie_steps, args.hard_brake, args.close_pet)
        except Exception as exc:  # one bad scene must not stop a batch
            print("skip %s: %s: %s" % (path.name, type(exc).__name__, exc), file=sys.stderr)
            skipped += 1
            continue
        if fmt == "catk_cache" and row["case"] == "driveway":  # off-lane is unreliable on a rebuilt map
            row["flags"] = (row["flags"] + ";" if row["flags"] else "") + "driveway_on_rebuilt_map"
            if row["confidence"]:
                row["confidence"] = "low"
        row.update(scene=path.stem, scenario_id=scene.scenario_id, format=fmt, pair=how,
                   first_id=scene.track_ids[a], second_id=scene.track_ids[o], jurisdiction=profile["id"])
        rows.append(row)

    import csv
    cols = ["scene", "scenario_id", "format", "pair", "first_id", "second_id", "jurisdiction", "case", "holder",
            "went_first", "pet_s", "sdc_violation",
            "ooi_violation", "victim_speed_at_pass", "victim_max_decel", "affected", "affected_by",
            "sdc_aggressive_candidate",
            "ooi_aggressive_candidate", "confidence", "conflict", "angle_deg", "conflict_x", "conflict_y",
            "sdc_control", "ooi_control", "sdc_turn", "ooi_turn", "flags"]
    # the same recording cut at different times gives the same conflict point and angle
    seen = {}
    for r in rows:
        if "conflict_x" in r:
            seen.setdefault((r["conflict_x"], r["conflict_y"], r["angle_deg"]), []).append(r["scene"])
    for r in rows:
        if "conflict_x" in r:
            same = [s for s in seen[(r["conflict_x"], r["conflict_y"], r["angle_deg"])] if s != r["scene"]]
            if same:
                r["flags"] = (r["flags"] + ";" if r["flags"] else "") + "same_conflict_as=" + "/".join(same)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    v_s, v_o = sum(r["sdc_violation"] for r in rows), sum(r["ooi_violation"] for r in rows)
    c_s, c_o = sum(r["sdc_aggressive_candidate"] for r in rows), sum(r["ooi_aggressive_candidate"] for r in rows)
    print("%d scenes screened (%d skipped); right-of-way violations SDC %d, OOI %d; of those the other vehicle "
          "was affected (aggressive candidates) SDC %d, OOI %d -> %s" % (len(rows), skipped, v_s, v_o, c_s, c_o, args.out))


if __name__ == "__main__":
    main()
