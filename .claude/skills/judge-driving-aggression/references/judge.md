# The duty x cost x attribution judge (`scripts/judge.py`)

A second opinion beside the two screens. It was built after the screens and the reviewing agents,
from what they got wrong (aggression_generalisation_plan.md in the research workspace). The code is
in `scripts/responsibility/`: `attribution.py`, `forced_merge.py`, `forced_yield.py` and
`aggression_judge.py`. All four are copied from the CAT fork (see that folder's `__init__.py`).

## The decision

X is aggressive toward Y when all three hold:

1. **Duty.** X owed Y something:
   - at a crossing, to yield (right_of_way.py's table, CVC 21800–21802 and 21804, or the
     left-hand mirror image);
   - when moving over, to do so only with reasonable safety (21658(a), 22107);
   - when following, not to follow too closely (21703).

   The CVC sections are cited from memory; check them before quoting.
2. **Attribution.** X caused the conflict:
   - **crossing:** the yielder entered the conflict zone first;
   - **join:** the car that moved sideways against the local road direction did. The road
     direction is the median direction of the lane and edge segments within 6 m, so a lane that
     branches across the road does not hide the move. This is the screens' lane-join blind spot.
   - **following:** the follower. The 3 s after the other moved in ahead are excused.
3. **Cost.** The braking X's move asks of Y is above the threshold.
   - **Crossing:** forced_yield's a_req.
   - **Cut-in and tailgating:**
     `a_req = v_r² / (2 (gap + v_f²/(2·4) − v_r·1 s))`. It is the deceleration the rear car needs
     if the front car brakes at 4 m/s² and the rear car responds 1 s later.
     - A cut-in is scored by the worst value in the 1 s after the join.
     - Tailgating is scored by a value held for 1 s, at 5 m/s or more.
   - **Threshold:** the 0.9 quantile of the same quantity over logged drivers in the same situation
     (`judge.py calibrate`). It is one in ten of the logged drivers, not a number written by hand.

**Abstain** is its own verdict. The judge abstains when there was a conflict but its cause cannot
be read: a crossing with no holder, or a simultaneous entry, or a join where the map gives no road
direction.

## How it differs from the screens

| | screens | judge |
|---|---|---|
| Thresholds | absolute, from the rubric (0.8 s / 0.5 s headway, 2 m/s² held 1 s) | relative: q0.9 of logged driving in the same dataset |
| Who moved (joins) | change of offset from each car's own lane | sideways motion against the local road direction |
| Crossing law | any jurisdiction profile | right_of_way.py's California table or its left-hand mirror only |
| Output | candidates for review | a verdict, or abstain, with the demand and its quantile |

**Relative thresholds were chosen on 2026-10-29.** Each dataset and region sets its own
threshold, so a cut-in counts as aggressive only when it is unusual for the traffic it happens in.
The cost of that choice is a lower count: on CAT's scenes the rubric's absolute headways flag about
three times as many cut-ins as q0.9 does (34 against 11). Calibrate on the data you judge, or on a
larger set from the same place. A calibration holds a few hundred references at best, so report
`n` with every threshold.

## What it was measured to do (CAT's 500 scenes, CPU)

All numbers are from the CAT fork's `responsibility/README.md`, "Duty, cost and attribution". They
were measured with that repository's ground-truth builders (make_go_first.py, make_cut_in.py),
which are not copied here.

**Calibration**

| kind | references | q0.9 |
|---|---|---|
| crossing | 17 (marked pair), 27 (all pairs) | 2.54 m/s² |
| cut-in | 225 joins | 5.32 m/s² |
| tailgate | 2549 following episodes | 4.68 m/s² |

**Attribution (A1)**
- 9 of 9 scenes with a clear actor are right. They include 3 lane-branch joins the screen missed.
- It agrees with the reviewing agents on the actor in 37 of 40 cut-in and tailgate verdicts.

**Crossings (rewrites where the yielder is made to go first)**
- Flagged: 50% at PET 1 s and 20% at PET 2 s.
- Logged lawful pairs: 1% flagged, against 8% on cost alone.
- None of the crossing screen's 16 violations that cost the other nothing is flagged.

**Cut-ins (the car behind advanced to a target headway)**
- Flagged: 40% at 0.6 s, 60% at 0.4 s and 67% at 0.25 s.
- Logged lane changes with 1–3 s headway: 5% flagged.

**Against the reviewing agents' labels** (1000 judgements):
- both call 16 aggressive; the agents alone 24; the judge alone 12;
- about a quarter of the judgements abstain.

**Left-hand traffic**
- The 500 scenes were mirrored and judged with the left-hand table.
- The holders and all 1000 verdicts are identical.
- This tests the code's symmetry, not any real country's law.

## Known limits

- **Few crossing references.** A reference needs a holder and a yielder that went first. CAT's
  500 scenes give 17–27 and catk's 306 give 1–17, so the crossing threshold is weak. Use more
  scenes.
- **Low speed.** At an all-way stop, a holder creeping at 2–3 m/s can get an infinite a_req: the
  1 s response covers its distance to the zone.
- **The ground-truth rewrites are open loop.** The car behind does not react.
- **No real accuracy yet.** None of the comparisons is against ground truth; the agents' labels
  are not it either. A person who knows the traffic code has to adjudicate a sample (E5).
