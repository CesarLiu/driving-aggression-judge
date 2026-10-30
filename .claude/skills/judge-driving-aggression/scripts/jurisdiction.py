"""Jurisdiction profiles and the right-of-way decision they drive.

A profile (``jurisdictions/<id>.json``) states which traffic rules decide who goes first at an
intersection without a working signal, and optionally (``green_rules``) between two vehicles that
both have a circular green. ``decide`` applies one to two approaches (anything with
``.control``, ``.turn``, ``.heading``, ``.arrived``, ``.entered``, ``.through``, as
``responsibility.right_of_way.Approach`` has). It has no dependency on the scene code, so it can
be tested on its own (``python jurisdiction.py`` runs the self-test).

Conventions, all relative to each vehicle's own driving direction so that left- and right-hand
traffic need no special case:

* ``rel`` is the heading of b minus the heading of a, in (-180, 180] degrees. rel in (45, 135)
  means b comes from a's right (it heads towards a's left); rel in (-135, -45) from a's left.
* a rule returns "a", "b" or None for who has the right of way; None means it does not apply.
"""

import json
import math
import sys
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent.parent / "jurisdictions"
SAME_DIRECTION = math.radians(45.0)  # the same value as right_of_way.SAME_DIRECTION
ONCOMING = math.radians(135.0)  # and right_of_way.ONCOMING
STOP, NONE, GO, RED = "stop", "none", "go", "red"
RED_ARROW = "LANE_STATE_ARROW_STOP"  # the WOMD lane state of a red arrow


def load_profile(name_or_path):
    path = Path(name_or_path)
    if not path.suffix:
        path = PROFILE_DIR / (str(name_or_path) + ".json")
    if not path.exists():
        sys.exit("no profile %r; available: %s" % (str(name_or_path), ", ".join(list_profiles())))
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    named = profile["all_way_stop"]["tie_rules"] + profile["uncontrolled_rules"] + (profile.get("green_rules") or [])
    unknown = [r for r in named if r not in RULES]
    if unknown:
        raise ValueError("profile %s names unknown rules %s; known: %s" % (path.name, unknown, sorted(RULES)))
    return profile


def list_profiles():
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _rel(a, b):
    return _wrap(b.heading - a.heading)


def _left(turn):
    return turn in ("left", "u-turn")


# Each rule: (a, b, rel) -> "a" | "b" | None, who holds the right of way.

def oncoming_left_yields(a, b, rel):
    """A left turn (or U-turn) yields to oncoming traffic, straight on or turning right (CVC 21801(a);
    StVO 9(3), (4)). Two oncoming left turners: no holder (StVO 9(4): they turn in front of each other)."""
    if abs(rel) > ONCOMING and _left(a.turn) != _left(b.turn):
        return "b" if _left(a.turn) else "a"


def oncoming_right_yields(a, b, rel):
    """Left-hand traffic: a right turn crosses the oncoming lane and yields to it."""
    if abs(rel) > ONCOMING and (a.turn == "right") != (b.turn == "right"):
        return "b" if a.turn == "right" else "a"


def oncoming_right_yields_left(a, b, rel):
    """A right turner yields to an oncoming left turner (China, implementing regulations 51(7) and 52(4);
    the opposite of StVO 9(4))."""
    if abs(rel) > ONCOMING and {a.turn, b.turn} == {"right", "left"}:
        return "b" if a.turn == "right" else "a"


def turner_yields_straight(a, b, rel):
    """A vehicle that turns yields to one that goes straight on (China, implementing regulations 51(7)
    and 52(3))."""
    if (a.turn == "straight") != (b.turn == "straight"):
        return "a" if a.turn == "straight" else "b"


def yield_right(a, b, rel):
    """The vehicle on the left yields to the one on the right (CVC 21800(b)(1), (c); StVO 8(1); China,
    implementing regulations 52(2))."""
    if SAME_DIRECTION < rel < ONCOMING:
        return "b"
    if -ONCOMING < rel < -SAME_DIRECTION:
        return "a"


def yield_left(a, b, rel):
    """Left-hand traffic: the vehicle on the right yields to the one on the left."""
    if SAME_DIRECTION < rel < ONCOMING:
        return "a"
    if -ONCOMING < rel < -SAME_DIRECTION:
        return "b"


def ending_road_yields(a, b, rel):
    """At a T junction the road that ends there yields to the one that continues (CVC 21800(b)(1),
    as an exception to yield_right on a simultaneous entry)."""
    if {a.through, b.through} == {True, False}:
        return "a" if a.through else "b"


RULES = {f.__name__: f for f in (oncoming_left_yields, oncoming_right_yields, oncoming_right_yields_left,
                                 turner_yields_straight, yield_right, yield_left, ending_road_yields)}
RULES["first_entered"] = None  # handled in decide: whoever entered the intersection first


def _first(ta, tb, tie_steps):
    """"a" / "b" for the earlier of two step indices (None: unknown or within the tie window)."""
    if ta is None and tb is None:
        return None
    if ta is None or tb is None:
        return "b" if ta is None else "a"
    return None if abs(ta - tb) <= tie_steps else "a" if ta < tb else "b"


def _apply(names, a, b, rel, tie_steps):
    for name in names:
        if name == "first_entered":
            holder = _first(a.entered, b.entered, tie_steps)
        else:
            holder = RULES[name](a, b, rel)
        if holder is not None:
            return holder, name
    return None, None


def handles(a, b, profile=None):
    """Whether the profile decides this pair: neither faces a light (both face a stop sign or
    nothing), or both entered on a circular green and the profile has ``green_rules``. Every other
    signalised pair (a red light, a green arrow, an unknown light) is left to the CAT right-of-way
    code."""
    if a.control in (STOP, NONE) and b.control in (STOP, NONE):
        return True
    return a.control == b.control == GO and bool((profile or {}).get("green_rules"))


def decide(profile, a, b, tie_steps=10):
    """(holder, case) for approaches ``a`` and ``b`` that ``handles`` accepts; holder is "a", "b" or None.

    ``tie_steps``: arrivals (or entries) this many 0.1 s steps apart count as simultaneous.
    """
    rel = _rel(a, b)
    if a.control == b.control == GO:
        holder, name = _apply(profile.get("green_rules") or [], a, b, rel, tie_steps)
        return holder, "green/" + name if name else "green/simultaneous"
    stop_a, stop_b = a.control == STOP, b.control == STOP
    if stop_a and stop_b:
        awy = profile["all_way_stop"]
        if awy["first_arrival"]:
            if a.arrived == b.arrived == -1:  # both already waiting when the clip began
                if awy["unknown_order"] != "tie":
                    return None, "all-way-stop/order-unknown"
                holder, name = _apply(awy["tie_rules"], a, b, rel, tie_steps)
                return holder, "all-way-stop/order-unknown-" + name if name else "all-way-stop/order-unknown"
            # -1 means "already there when the clip began", so it is earlier than any observed arrival,
            # however close in steps: comparing it as step -1 would turn it into a tie
            if (a.arrived == -1) != (b.arrived == -1) and None not in (a.arrived, b.arrived):
                return ("a" if a.arrived == -1 else "b"), "all-way-stop/first-in"
            holder = _first(a.arrived, b.arrived, tie_steps)
            if holder is not None:
                return holder, "all-way-stop/first-in"
        holder, name = _apply(awy["tie_rules"], a, b, rel, tie_steps)
        return holder, "all-way-stop/" + name if name else "all-way-stop/simultaneous"
    if stop_a != stop_b:  # two-way stop: the vehicle at the sign yields to cross traffic
        return ("b" if stop_a else "a"), "stop-sign"
    if profile["t_junction"] == "ending_road_yields" and {a.through, b.through} == {True, False}:
        return ("a" if a.through else "b"), "through-road"
    holder, name = _apply(profile["uncontrolled_rules"], a, b, rel, tie_steps)
    return holder, "uncontrolled/" + name if name else "uncontrolled"


def red_entry(profile, approach, stopped):
    """How the profile's law sees ``approach`` entering the intersection, "" if it did not enter on red.

    ``approach.control`` is the light at the stop line when the vehicle crossed it and
    ``approach.signal`` the WOMD lane state behind it. ``stopped`` says whether the vehicle was seen
    to stop before entering (None: its approach was not observed). The profile's ``red_turn``
    (``turns``: the turns allowed on a circular red; ``stop_first``; ``sign_required``: only where
    a sign allows it, which the maps do not show) decides:

      ran_red_arrow              a red arrow forbids its movement (CVC 21453(c); China reg. 40)
      ran_red                    a turn the profile does not allow on red, or straight on
      turn_on_red_needs_sign     allowed only under a sign the map cannot show (StVO 37(2) Grünpfeil)
      turn_on_red_without_stop   allowed after a stop, and the vehicle was seen not to stop
      turn_on_red_stop_unseen    allowed after a stop, and the approach was not observed
      lawful_turn_on_red         allowed (CVC 21453(b) after a stop; China reg. 38 without one)

    Entering on red is a signal violation, not a right-of-way question: who had to yield is still
    decided by the red-light rule (the vehicle on red yields), and it is aggressive only if it cost
    someone something."""
    if approach.control != RED or approach.entered is None:
        return ""
    if approach.signal == RED_ARROW:
        return "ran_red_arrow"
    rule = profile.get("red_turn") or {}
    if approach.turn not in rule.get("turns", []):
        return "ran_red"
    if rule.get("sign_required"):
        return "turn_on_red_needs_sign"
    if rule.get("stop_first"):
        if stopped is None:
            return "turn_on_red_stop_unseen"
        if not stopped:
            return "turn_on_red_without_stop"
    return "lawful_turn_on_red"


def _self_test():
    from types import SimpleNamespace as N

    def ap(heading_deg, turn, control=STOP, arrived=-1, entered=0, through=True):
        return N(heading=math.radians(heading_deg), turn=turn, control=control, arrived=arrived, entered=entered,
                 through=through)

    def mirror(x):  # reflect the road: headings negate, left and right swap
        swap = {"left": "right", "right": "left"}
        return N(heading=-x.heading, turn=swap.get(x.turn, x.turn), control=x.control, arrived=x.arrived,
                 entered=x.entered, through=x.through)

    us, lh = load_profile("us-california"), load_profile("left-hand-draft")
    # both waiting at an all-way stop when the clip begins: order unknown, no verdict ...
    a, b = ap(0, "straight"), ap(90, "straight")
    assert decide(us, a, b) == (None, "all-way-stop/order-unknown"), decide(us, a, b)
    # ... unless a profile asks to treat it as a tie: b comes from a's right (rel +90), a yields
    tie = dict(us, all_way_stop=dict(us["all_way_stop"], unknown_order="tie"))
    assert decide(tie, a, b) == ("b", "all-way-stop/order-unknown-yield_right"), decide(tie, a, b)
    # arriving together, a turns left across oncoming b: a yields
    a, b = ap(0, "left", arrived=20), ap(180, "straight", arrived=20)
    assert decide(us, a, b)[0] == "b"
    # the first to arrive goes first, whatever the turn
    a, b = ap(0, "left", arrived=-1), ap(180, "straight", arrived=30)
    assert decide(us, a, b) == ("a", "all-way-stop/first-in")
    # already waiting at the start beats an arrival 0.6 s into the clip (not a tie)
    a, b = ap(0, "left", arrived=-1), ap(180, "straight", arrived=6)
    assert decide(us, a, b) == ("a", "all-way-stop/first-in"), decide(us, a, b)
    # arrivals 0.5 s apart are a tie: left turn yields
    a, b = ap(0, "left", arrived=10), ap(180, "straight", arrived=15)
    assert decide(us, a, b)[0] == "b"
    # two-way stop: the sign yields
    a, b = ap(0, "straight", control=STOP), ap(90, "straight", control=NONE)
    assert decide(us, a, b) == ("b", "stop-sign")
    # a T junction: the road that ends yields
    a, b = ap(0, "left", control=NONE, through=False), ap(90, "straight", control=NONE, through=True)
    assert decide(us, a, b) == ("b", "through-road")
    # mirror symmetry: the left-hand profile is the US one reflected, so on the reflected road it
    # must name the same vehicle (a reflection swaps left and right and negates the headings)
    for a, b in [(ap(0, "left"), ap(180, "straight")), (ap(0, "right"), ap(180, "straight")),
                 (ap(0, "straight"), ap(90, "left")), (ap(0, "straight", arrived=5), ap(180, "left", arrived=50)),
                 (ap(0, "straight", control=NONE), ap(90, "left", control=NONE)),
                 (ap(0, "left", control=NONE, through=False), ap(90, "straight", control=NONE))]:
        assert decide(us, a, b)[0] == decide(lh, mirror(a), mirror(b))[0], (a.turn, b.turn, decide(us, a, b), decide(lh, mirror(a), mirror(b)))

    st, de, cn = load_profile("us-california-statute"), load_profile("germany-draft"), load_profile("china-draft")
    # the CVC as written: whoever entered first (21800(a)), then on a simultaneous entry the
    # terminating highway yields (21800(b)(1)), then yield to the right
    a, b = ap(0, "straight", control=NONE, entered=10), ap(90, "straight", control=NONE, entered=40)
    assert decide(st, a, b) == ("a", "uncontrolled/first_entered"), decide(st, a, b)
    a, b = ap(0, "straight", control=NONE, entered=10), ap(90, "straight", control=NONE, entered=15)
    assert decide(st, a, b) == ("b", "uncontrolled/yield_right"), decide(st, a, b)
    a, b = ap(0, "straight", control=NONE, entered=10), ap(90, "left", control=NONE, entered=15, through=False)
    assert decide(st, a, b) == ("a", "uncontrolled/ending_road_yields"), decide(st, a, b)
    a, b = ap(0, "straight", control=NONE, entered=40), ap(90, "left", control=NONE, entered=10, through=False)
    assert decide(st, a, b) == ("b", "uncontrolled/first_entered"), decide(st, a, b)
    # Germany: rechts vor links holds at a T junction too (StVO 8(1)); the US profile lets the road end yield
    a, b = ap(0, "straight", control=NONE), ap(90, "left", control=NONE, through=False)
    assert decide(de, a, b) == ("b", "uncontrolled/yield_right"), decide(de, a, b)
    assert decide(us, a, b) == ("a", "through-road")
    # a left turner and an oncoming right turner: Germany has the left turner yield (StVO 9(4)),
    # China the right turner (reg. 52(4))
    a, b = ap(0, "left", control=NONE), ap(180, "right", control=NONE)
    assert decide(de, a, b)[0] == "b" and decide(cn, a, b) == ("a", "uncontrolled/oncoming_right_yields_left")
    # China: the car from the right (reg. 52(2)); a turner yields to an oncoming car going straight (52(3))
    a, b = ap(0, "straight", control=NONE), ap(90, "straight", control=NONE)
    assert decide(cn, a, b) == ("b", "uncontrolled/yield_right")
    a, b = ap(0, "right", control=NONE), ap(180, "straight", control=NONE)
    assert decide(cn, a, b) == ("b", "uncontrolled/turner_yields_straight")
    # on a circular green the profile decides only if it has green_rules (China 51(7), StVO 9(3)/(4));
    # California's signalised rules stay with the CAT code
    a, b = ap(0, "left", control=GO), ap(180, "straight", control=GO)
    assert handles(a, b, cn) and handles(a, b, de) and not handles(a, b, us) and not handles(a, b)
    assert decide(cn, a, b) == ("b", "green/turner_yields_straight") and decide(de, a, b)[0] == "b"
    a, b = ap(0, "left", control=GO), ap(180, "left", control=GO)
    assert decide(de, a, b) == (None, "green/simultaneous")  # two oncoming left turners: no holder
    # entering on red: a right turn is lawful in China without a stop, in California after one, in
    # Germany only under a Grünpfeil sign; a red arrow and going straight on are never lawful
    red = lambda turn, signal="LANE_STATE_STOP": N(control=RED, signal=signal, turn=turn, entered=30)
    assert red_entry(cn, red("right"), None) == "lawful_turn_on_red"
    assert red_entry(us, red("right"), True) == "lawful_turn_on_red"
    assert red_entry(us, red("right"), False) == "turn_on_red_without_stop"
    assert red_entry(us, red("right"), None) == "turn_on_red_stop_unseen"
    assert red_entry(de, red("right"), True) == "turn_on_red_needs_sign"
    assert red_entry(us, red("straight"), True) == "ran_red"
    assert red_entry(cn, red("right", RED_ARROW), True) == "ran_red_arrow"
    assert red_entry(lh, red("left"), True) == "lawful_turn_on_red"
    assert red_entry(us, N(control=GO, signal="LANE_STATE_GO", turn="left", entered=30), None) == ""
    print("jurisdiction self-test passed; profiles:", ", ".join(list_profiles()))


if __name__ == "__main__":
    sys.exit(_self_test())
