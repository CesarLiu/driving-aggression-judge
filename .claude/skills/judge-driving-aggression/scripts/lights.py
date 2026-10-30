"""Filling in the traffic-light states WOMD leaves unknown, so that fewer signalised conflicts end
as signal-unknown (no holder).

WOMD records a light per controlled lane, but not at every step and not for every lane. On CAT's
500 scenes most missing states sit before a light's first or after its last recorded step, and
some intersection lanes carry no light at all. Two ways to fill them, each checked against
recorded states by ``python lights.py --validate <scenes>`` (leave one out):

  hold    a step without a state takes the nearest recorded state of the same light, if one lies
          within ``hold`` steps. Lights change rarely within a few seconds.
  borrow  a lane without a state at a step (no light, or none recorded or held) takes the state of
          its siblings: lanes with a light that start within SAME_LINE metres of it (the same stop
          line), head within SAME_HEADING of it and make the same movement (straight, left, right
          or U-turn). Turn lanes are often on their own phase (arrows), hence the last condition.
          Their states must agree in what they allow (go, arrow, red, stop); otherwise nothing is
          borrowed.

``infer_lights`` returns a copy of the scene whose dynamic_map_states hold the filled states. An
added light carries ``metadata.inferred = "borrowed"``. A held step is not marked, and neither is
a borrowed step on a light the map already has. Compare against the original scene to see what
changed (screen_crossing.py --infer-lights does, and flags it).
"""

import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from responsibility.lanes import lane_graph  # noqa: E402
from responsibility.right_of_way import LIGHTS, TURN, U_TURN  # noqa: E402

UNKNOWN_STATES = (None, "", "LANE_STATE_UNKNOWN")
HOLD = 10  # steps (1 s): a recorded state agrees with the same light 1 s away 97.4% of the time on CAT 500
SAME_LINE = 6.0  # m between the starts of sibling lanes
SAME_HEADING = np.deg2rad(20.0)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def control(state):
    """What a WOMD lane state allows (right_of_way.LIGHTS), None when unknown."""
    if state in UNKNOWN_STATES:
        return None
    return LIGHTS.get(state)


def movement(graph, lane):
    turn = float(_wrap(graph.direction[lane][-1] - graph.direction[lane][0]))
    return "u-turn" if abs(turn) > U_TURN else "left" if turn > TURN else "right" if turn < -TURN else "straight"


def hold(states, steps=HOLD):
    """``states`` with each unknown step given the nearest recorded state within ``steps`` (the
    earlier one on a tie)."""
    known = [t for t, s in enumerate(states) if s not in UNKNOWN_STATES]
    out = list(states)
    if not known:
        return out
    known = np.array(known)
    for t, s in enumerate(states):
        if s in UNKNOWN_STATES:
            k = known[np.argmin(np.abs(known - t))]
            if abs(int(k) - t) <= steps:
                out[t] = states[int(k)]
    return out


def siblings(graph, lane, lit):
    """The lanes of ``lit`` that start at the same stop line as ``lane``, head the same way and make
    the same movement."""
    if lane not in graph.points:
        return []
    start, head, move = graph.points[lane][0], graph.direction[lane][0], movement(graph, lane)
    return [o for o in lit if o != lane and o in graph.points
            and np.linalg.norm(graph.points[o][0] - start) <= SAME_LINE
            and abs(_wrap(graph.direction[o][0] - head)) <= SAME_HEADING and movement(graph, o) == move]


def consensus(states_by_lane, lanes, t):
    """The state at step t that the lanes agree on (by what it allows), None if none or disagreeing."""
    seen = [states_by_lane[o][t] for o in lanes if t < len(states_by_lane[o]) and control(states_by_lane[o][t])]
    if not seen or len({control(s) for s in seen}) > 1:
        return None
    return seen[0]


def infer_lights(scene, steps=HOLD, borrow=True):
    """(a copy of ``scene`` with light states filled in, a Counter of what was filled)."""
    graph = lane_graph(scene)
    lights = copy.deepcopy(scene.dynamic_map_states)
    by_lane = {str(l["lane"]): l for l in lights.values() if l.get("type") == "TRAFFIC_LIGHT" and "lane" in l}
    report = Counter()
    states = {}
    for lane, light in by_lane.items():
        own = list(light["state"]["object_state"])
        states[lane] = hold(own, steps)
        report["held"] += sum(a in UNKNOWN_STATES and b not in UNKNOWN_STATES for a, b in zip(own, states[lane]))
    if borrow:
        lit = list(states)
        starts = np.array([graph.points[o][0] for o in lit if o in graph.points]) if lit else np.zeros((0, 2))
        candidates = set(lit)
        if len(starts):
            for lane, pts in graph.points.items():  # unlit lanes that start at a lit stop line
                if lane not in by_lane and np.linalg.norm(starts - pts[0], axis=1).min() <= SAME_LINE:
                    candidates.add(lane)
        filled = {}
        for lane in candidates:
            sib = siblings(graph, lane, lit)
            if not sib:
                continue
            own = states.get(lane, [None] * scene.n_steps)
            new = [s if s not in UNKNOWN_STATES else consensus(states, sib, t) for t, s in enumerate(own)]
            n = sum(a in UNKNOWN_STATES and b not in UNKNOWN_STATES for a, b in zip(own, new))
            if n:
                filled[lane] = new
                report["borrowed"] += n
        for lane, new in filled.items():
            states[lane] = new
            if lane not in by_lane:
                lights["inferred_" + lane] = {"type": "TRAFFIC_LIGHT", "lane": lane,
                                              "stop_point": np.array([*graph.points[lane][0], 0.0]),
                                              "state": {"object_state": new},
                                              "metadata": {"inferred": "borrowed"}}
                report["lanes_added"] += 1
    for lane, light in by_lane.items():
        light["state"] = {"object_state": states[lane]}
    out = copy.copy(scene)
    out.dynamic_map_states = lights
    return out, report


# --- validation: leave one out against recorded states ------------------------------------------

def validate(scene_paths, Scene, load_scene, max_hold=50):
    """Accuracy of hold at each distance and of borrow, judged by what the state allows."""
    hold_hits, hold_n = np.zeros(max_hold + 1), np.zeros(max_hold + 1)
    borrow = Counter()
    for path in scene_paths:
        scene, _ = load_scene(path, Scene)
        graph = lane_graph(scene)
        raw = {str(l["lane"]): list(l["state"]["object_state"]) for l in scene.dynamic_map_states.values()
               if l.get("type") == "TRAFFIC_LIGHT" and "lane" in l}
        for st in raw.values():
            c = [control(s) for s in st]
            known = [t for t, x in enumerate(c) if x]
            ks = set(known)
            for t in known:  # predict c[t] from the same light d steps earlier or later
                for d in range(1, max_hold + 1):
                    for u in (t - d, t + d):
                        if u in ks:
                            hold_n[d] += 1
                            hold_hits[d] += c[u] == c[t]
        lit = list(raw)
        for lane in lit:  # hide this lane's light, borrow from its siblings' recorded states
            sib = siblings(graph, lane, lit)
            for t, s in enumerate(raw[lane]):
                if not control(s):
                    continue
                borrow["recorded"] += 1
                if not sib:
                    continue
                guess = consensus(raw, sib, t)
                if guess is None:
                    borrow["no_consensus"] += 1
                    continue
                borrow["predicted"] += 1
                borrow["right"] += control(guess) == control(s)
                borrow["right_" + movement(graph, lane)] += control(guess) == control(s)
                borrow["predicted_" + movement(graph, lane)] += 1
    acc = {d: hold_hits[d] / hold_n[d] for d in range(1, max_hold + 1) if hold_n[d]}
    return acc, hold_n, borrow


def main(argv=None):
    import argparse
    from responsibility.scene import Scene
    from scenes import load_scene
    from screen_crossing import scene_files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", required=True, help="folder of scenes whose recorded states are the truth")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args(argv)
    acc, n, b = validate(scene_files(args.validate)[: args.n], Scene, load_scene)
    print("hold: agreement of a recorded state with the same light d steps away")
    for d in (1, 5, 10, 15, 20, 30, 40, 50):
        if d in acc:
            print("  d = %4.1f s: %.3f (n %d)" % (d / 10, acc[d], n[d]))
    p = b["predicted"]
    print("borrow: %d recorded states; %d have siblings with a consensus, %d siblings disagreed; right %.3f"
          % (b["recorded"], p, b["no_consensus"], b["right"] / p if p else float("nan")))
    for m in ("straight", "left", "right", "u-turn"):
        if b["predicted_" + m]:
            print("  %s lanes: right %.3f of %d" % (m, b["right_" + m] / b["predicted_" + m], b["predicted_" + m]))


if __name__ == "__main__":
    main()
