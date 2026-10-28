# What counts as aggressive

Each scene has two objects of interest: the self-driving car (SDC) and its partner (OOI). Judge
**each one toward the other only** and ignore how either treats third vehicles.

X is aggressive toward Y when **X's own action shifted risk or effort onto Y beyond what the
situation required**. Having the right of way does not excuse a compressed safety margin; being
the one who must yield and yielding is never aggressive.

## Two families, judged differently

**A. Conflicts the traffic law settles** (crossing and turning paths at an intersection without a
working signal). Decide who had the right of way from the jurisdiction profile
(`jurisdictions/<id>.json`, run `scripts/screen_crossing.py`). X is aggressive when it **went
through the conflict first although Y had the right of way, **and that affected Y**. No threshold
on braking, PET or gap decides who violated: a lawful first arrival is not aggressive even if Y
had to brake or wait, and every vehicle has to slow for an intersection. In the CAT scenes lawful
first arrivals caused the same braking (2.5 to 3.8 m/s²) as the ones people first labelled
aggressive, so braking cannot separate them.

**A violation that did not affect Y is not aggressive.** The definition asks for risk or effort
shifted onto Y, and the law itself (CVC 21801(a), 21802(a), 21804(a), as recalled) asks a driver
to yield only to traffic in the intersection or so close as to be an immediate hazard. Y paid for
it when it was **made to stop again** (it had stopped, got going and had to stop), **braked hard**
(at least `--hard-brake`, 3 m/s², roughly where braking stops being comfortable) without a stop
sign of its own to brake for, or was **cut close** (reached the conflict within `--close-pet`,
1.5 s, of X while moving).

**Waiting at a stop sign is not a cost.** Y had to stop there anyway. At an all-way stop where both
were already waiting when the clip began, the one that goes while the other waits is ordinary
turn-taking: the order is unknown and the other not contesting it suggests the one that went was
first. Do not invent a tie to blame someone. An earlier version did, to match a set of human labels,
and every scene it flagged turned out on review to be ordinary turn-taking.

On CAT's 500 scenes the screen finds 16 law violations and none that cost the other vehicle
anything; the review of every flagged crossing agrees. Crossing aggression may simply be rare in
logged driving; the parameters are judgement, so check them when new data has real cases.

**B. Conflicts with no right of way involved** (same-direction traffic). These are physical, and
the numbers are the original rubric's, not a statute's:

- **Cut-in**: X moved laterally into Y's lane ahead of Y (X did the lane change, not Y), leaving Y
  a time headway below about 1 s, or making Y brake harder than about 2 m/s².
- **Tailgating**: X followed Y in its path (the boxes overlap sideways, so a car half a lane over
  right behind Y counts) at 5 m/s or more with a time headway below about
  0.9 s for a second or longer, by its own choice (not in the first ~3 s after Y cut in ahead of
  X), or closed in on Y with a time to collision below 2 s.
- **Close pass** that X caused: the boxes came within about 0.5 m at more than 3 m/s relative
  speed and X closed in laterally.
- Anything clearly of the same kind (X forced a merge, accelerated to deny a gap Y was already
  taking).

`scripts/screen_same_direction.py` applies these. Two choices in it are judgement, not statute:
a cut-in counts below **0.8 s** of headway (not "about 1 s"): merging with about a second of
headway is ordinary in dense urban traffic, so 0.8 to 1 s is reported as borderline; a car that
merges in while 2 m/s or more faster opens the gap itself, so a headway of 0.5 to 0.8 s alone is
not a cut-in unless the follower had to brake (below 0.5 s it is a cut-off however fast the merging
car is: the follower has no margin if it brakes); a follower that is falling back (gap opening by more than
1 m/s) or behind a car below 2 m/s is not tailgating, and the first 3 s after the other moved in
ahead (from the next lane, a side road or a turn) are not the follower's doing, unless the follower
itself moved in close behind. Braking is speed
lost over 1 s: WOMD speeds jump by 1-2 m/s between frames. Cut-in was the only kind the first labeller found reliably; tailgating and close pass
were rare. Labels from people or earlier passes are inputs to discuss, not ground truth.

Who moved is the hard part. The screen compares each car's offset from its own lane before and
after (against both the lane it started in and the one it ended in, so lane matching that jumps
between overlapping lanes in an intersection does not count as a move). Where the map draws the
manoeuvre as a lane branch or merge, a real move across the road follows a lane link and is not
seen: review scenes where both cars end up in one lane.

## Not aggressive

- No real interaction (far apart, paths never conflict in time).
- Y was the actor (Y cut in, Y violated X's right of way, Y tailgated X): X is the victim, label 0.
- X yielded, waited, followed at a normal headway, or took a gap Y did not have to brake for.
- Creeping in behind a car that has already left the conflict, with a gap of about a metre or more.
- Slow queues (walking pace) and parking manoeuvres with ample room.

Both can be aggressive in one scene; usually at most one is.

## Failure modes to check by eye

- **The clip starts mid-approach.** Anything before step 0 is unseen. At an all-way stop the SDC is
  almost always already at its stop line (arrival time unknown, treated as earlier than any arrival
  seen in the clip). When both were already waiting the order is unknown and there is no verdict
  (profile `unknown_order: "none"`). Say which assumption the verdict rests on.
- **Lane matching gets turns wrong** where lanes fan out (CAT scene 391 had them swapped). The
  screen takes the turn from the vehicle's own heading 20 m before and after the conflict when the
  track goes on at least 10 m past it, and flags `*_turn_from_track`.
- **Parking-lot exits** can be read as stop-sign approaches for the vehicle on the road (scene
  154). Check the map when a stop-sign verdict involves a car park.
- **Do not label "failed to stop" from a clip.** A stop before step 0 cannot be seen, and among
  the 25 SDCs in the CAT scenes whose approach to a stop line was observed none drove through
  without slowing. Judge by who arrived first and by the yield rules.
- **Who moved.** A cut-in needs X itself to change lanes. Lane matching can treat adjacent lanes
  as one, so a "tailgate" may be two cars side by side in neighbouring lanes.
- **Duplicates.** The same recording cut at different times appears as two scenes (CAT scenes 282
  and 400; 342 and 482). The screen flags them (`same_conflict_as`). Count them as one event.
- **Signal maps.** At a signalised junction the screen does not apply the profile: it keeps the
  light-based verdict of the copied CAT right-of-way code (`red-light`, `protected-arrow`, ...), and a junction whose
  light for this movement was not observed (`signal-unknown`) gets no verdict.
- **Other never crossed.** If Y never reached the conflict within the 9 s, "X went first" rests on
  an extrapolation: keep the candidate but mark it low confidence.
