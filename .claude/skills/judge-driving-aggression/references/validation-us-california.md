# How the California profile was checked

Data: CAT's 500 WOMD scenes (`raw_scenes_500`), the SDC and the other object of interest in each.
The check is a review: the screen's verdict on every flagged crossing was compared with the
trajectories and figures. A person's labels (13 scenes marked "SDC aggressive") were looked at as
one input; the annotator did not know US traffic rules well, so they are not ground truth.

## What went wrong on the way

1. **A first labeller that judged by braking and PET** missed the right-of-way logic altogether:
   it gave the SDC priority whenever the SDC was already at its line at step 0 (45 of 48 all-way
   stops) and only looked for hard braking.
2. **A law-only screen** (went first against the other's right of way = aggressive) flagged 19
   cases. Reviewed, none was aggressive:
   - 8 were technical violations the other vehicle never noticed: it drove on at 3.8 to 11.7 m/s,
     15 to 36 m from the conflict;
   - 7 came from a rule added to reproduce the human labels: "both already waiting at step 0 is a
     tie, so the vehicle on the left must yield". On review every one was ordinary all-way-stop
     turn-taking (one went, the other waited at its line and never started). Removed: unknown
     order now gives no verdict;
   - 391: the lane match swapped the turns (the SDC went straight, the OOI was the waiting left
     turner). Fixed: the turn now comes from the vehicle's own heading;
   - 493: an arrival before the clip (-1) was compared as step -1 and made a tie with an arrival
     at 0.6 s. Fixed: before the clip is earlier than any arrival seen in it;
   - 154: a car leaving a parking lot was read as cross traffic with priority over the SDC.
     Not fixed in code; flagged in the rubric.
3. **"Affected" first meant "was slow or slowed"**, which counts waiting at a stop sign as a cost.
   Now it means made to stop again, hard braking without a stop sign of its own, or cut close.

## Result

The current screen finds 16 law violations in the 500 scenes and none that cost the other vehicle
anything. The review agrees: no crossing in these scenes shows aggression by the SDC or the OOI.
The only borderline case is scene 480's OOI, which never went below 3.5 m/s at its all-way stop and
turned in 1.9 m behind the SDC without forcing it to react (listed for a person's look).

Of the 13 human SDC labels only the cut-in (342/482, one event: the SDC changed into the OOI's lane
7 to 8 m ahead at 17 to 18 m/s, leaving it about 0.45 s) holds up. The crossing ones (76, 137, 197,
207, 282/400, 318, 423, 431, 436, 480) are ordinary first arrivals or turn-taking.

## Limits

- No real crossing aggression was found, so the effect thresholds (3 m/s², 1.5 s) are judgement,
  not fitted; check them when new data has real cases.
- Same-direction labels now have their own screen (below); only a sample of its disagreements
  with the first labeller was checked in the figures.
- Scenes where the other vehicle never reached the conflict within the clip have no verdict.

## Same-direction screen (scripts/screen_same_direction.py)

Compared, on CAT's 500 scenes, with the first labeller's same-direction labels (not ground truth),
and a sample of the disagreements (scenes 1, 200, 237, 250, 293, 359) looked at in the figures:

| | first labeller | screen | both |
|---|---|---|---|
| SDC cut-in | 8 | 6 | 5 |
| OOI cut-in | 50 | 32 | 27 |
| SDC tailgate | 0 | 2 | 0 |
| OOI tailgate | 11 | 8 | 6 |
| close pass | 1 | 0 | 0 |

Where they differ:

- **The labeller missed real lane changes**, judged by the figures: 237 (the SDC moved into the
  OOI's lane ahead of it; the labeller had it the other way round), 250 and 359 (the OOI merged in
  ahead of the SDC late in the clip; the labeller wrote "no lane change").
- **Deliberate differences of judgement** (OOI cut-ins the labeller had): 13 where the merging car
  was 2 m/s or more faster, left 0.5 to 0.8 s and the follower did not brake, and 6 at 0.8 to
  0.92 s of headway. The screen reports both in `notes`.
- **Screen blind spot** (checked for 293, assumed for the rest from the same note): 4 OOI and 3
  SDC cut-ins (15, 69, 156, 337; 293, 381, 483) where the map draws the move as a lane branch or merge, so the car follows a lane link while crossing the road.
- **Tailgating**: the labeller's 50 and 290 are cars half a lane over and falling back slowly,
  134 and 231 start right after the other moved in ahead (excused), 437 lasts 0.1 s. 342 and 411
  are screen hits the labeller had overruled, by its notes a turn with the gap opening and a
  lane-match artefact (not checked by eye).

## catk's cached WOMD (validation split, 306 scenes)

Both screens run on the cache directly (scripts/scenes.py rebuilds the map). Pairs: the SDC with
its object-of-interest partner in 32 scenes, two other objects of interest in 73, the SDC and its
nearest vehicle in 201. Results, not reviewed by eye (catk has no figures yet):

- crossing: 14 law violations, none that cost the other vehicle anything; 10 of them are
  "driveway" verdicts, 12 times CAT's rate, because the rebuilt map misses lanes (now low
  confidence);
- same-direction: SDC cut-in 2, OOI cut-in 4, tailgating 2; 4 scenes dropped by the sanity
  checks (sideways moves of 15 to 47 m, boxes touching at 22 m/s), which point at lane
  references or tracks gone wrong on the rebuilt map.

Treat catk results as candidates for review; the lights (only at 1.0 s), the inferred lane links
and the missing driveway polygons all make them weaker than on ScenarioNet scenes.
