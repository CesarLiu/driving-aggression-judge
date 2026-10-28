---
name: judge-driving-aggression
description: Decide whether an agent's driving toward another agent is aggressive in WOMD-style scenes (Waymo Open Motion Dataset as ScenarioNet scene files or catk/SMART womd_cache pickles), judging crossing and turning conflicts at intersections by the traffic law of a chosen country or region, and cut-ins, tailgating and close passes by physical rules. Use this whenever the user wants to label, audit or score aggressive driving of the SDC or objects of interest in motion-prediction or driving-log scenes, check or explain human aggression labels, find who violated the right of way, or adapt the aggression criteria to another country (left-hand traffic, China, Germany, other traffic codes), even if they only say "is the SDC aggressive here", "who had the right of way", or "label these scenes".
---

# Judging aggressive driving in WOMD-style scenes

Aggressive means: X's own action shifted risk or effort onto Y beyond what the situation required.
For conflicts at intersections the traffic law decides what X owed Y, so the criteria depend on
the country. For same-direction traffic (cut-in, tailgating, close pass) they are physical.

The method came out of reviewing CAT's 500 WOMD scenes. Its main lessons: who had the right of
way must come from the law, not from who braked; a violation counts only if it cost the other
vehicle something; and labels made by people who do not know the traffic rules can be wrong.
`references/validation-us-california.md` has the details; read it before trusting a result.

## Workflow

1. **Check the input.** Both screens read a folder of scene pickles, one per scene, in either of
   two formats (`scripts/scenes.py` tells them apart):
   - ScenarioNet / MetaDrive (CAT's `raw_scenes_500/*.pkl`): full map with lane links, stop signs
     and per-step lights;
   - catk's cached WOMD (`catk/data/womd_cache/<split>/<scenario_id>.pkl`): the map is rebuilt from
     SMART's tokens, so lane links are inferred from geometry, stop signs sit at the start of their
     lanes, and lights are known only at 1.0 s. Say so when a verdict rests on a light.
   They need only NumPy: the scene, lane-graph and right-of-way code they use is copied into
   `scripts/responsibility/` from the CAT fork's `responsibility` package. A raw WOMD tfrecord must
   be converted to ScenarioNet scenes first (ScenarioNet's Waymo converter, or the CAT fork's
   `scripts/responsibility/convert_womd_split.py`).
   The pair judged (`--pairs auto`, column `pair`): the SDC and its object-of-interest partner;
   else the two objects of interest (WOMD marks pairs without the SDC too); else the SDC and the
   vehicle that comes closest to it. `--agent ID --other ID` names a pair. In the output `sdc_*`
   columns are the first of the pair and `ooi_*` the second (`first_id`, `second_id`).

2. **Pick the jurisdiction.** Ask the user if they have not said. WOMD is entirely US cities, so
   `us-california` is the default there; other datasets need the profile of where they were
   recorded. `python scripts/jurisdiction.py` lists the profiles. Only `us-california` has been
   validated; the others are drafts (the script warns). If the user needs a country with no
   profile, follow "Adding a jurisdiction" in `references/jurisdictions.md` and say plainly that
   the result is unvalidated until a local expert has checked it.

3. **Screen the crossing conflicts.**

   ```bash
   python scripts/screen_crossing.py --scenes <folder-or-file> --profile us-california --out screen.csv
   ```

   One row per scene: who held priority (`holder`), who `went_first` through the conflict, `pet_s`,
   `sdc_violation` and `ooi_violation` (the other vehicle held priority and this one went first),
   `affected` and `affected_by` (the vehicle gone ahead of paid for it: `made_to_stop_again`,
   `hard_braking` without a stop sign of its own, or `cut_close`; with `victim_speed_at_pass` and
   `victim_max_decel`), `sdc_aggressive_candidate` and `ooi_aggressive_candidate` (violation and
   affected), a `confidence`, and `flags` that say what the verdict rests on
   (`both_waiting_at_clip_start`, `victim_never_reached_conflict`, `*_turn_from_track`,
   `same_conflict_as=<scene>` for duplicates, ...). A violation with `affected` = 0 is not
   aggressive; with `affected` empty its impact is unknown. These are candidates, not labels.

   **Screen the same-direction conflicts.**

   ```bash
   python scripts/screen_same_direction.py --scenes <folder-or-file> --out same_direction.csv
   ```

   For each direction (`sdc_*`: the first toward the second): `cut_in` (moved sideways into the
   other's lane ahead of it and forced 2 m/s² of braking held for 1 s, or left under 0.5 s of
   headway, or under 0.8 s while not pulling away at 2 m/s or more; with `cut_in_t`, `cut_in_lateral_move`,
   `cut_in_min_thw`, `cut_in_other_brake`, `cut_in_speed_difference`), `tailgate` (under 0.9 s
   behind a moving car, in its path (boxes overlapping sideways), for 1 s, at 5 m/s or more, not falling back; or TTC under 2 s; the 3 s after
   the other moved in ahead, by any route, are excused), `close_pass` (0.5 m box to box at 3 m/s
   relative speed, closing in sideways itself), a `reason` sentence and `notes`
   (`borderline_cut_in_thw_*`, `cut_in_pulling_away_*`, `lane_join_*_not_by_this_agent` when the
   other one moved, and two sanity checks that drop a candidate for a look by eye:
   `implausible_lateral_move_*` over 6 m, `boxes_touch_while_following_check_tracks`). Position is measured against the map
   lane the other car is in at that moment; who moved, by each car's change of offset from its own
   lane.

4. **Review every candidate and every scene the screen cannot see.** Open the tracks
   (`responsibility.scene.Scene.load(path)`: `position`, `velocity`, `heading`, `valid`, 10 Hz,
   91 steps; `sdc`, `objects_of_interest`) and check against `references/rubric.md`:
   - who arrived at the stop line first and whether the vehicle that went first really had to
     yield (the clip's first second is the SDC already at its line; a flagged tie rests on that);
   - whether the other vehicle actually crossed in the clip (`other_never_crossed_in_clip` rows
     are low confidence);
   - for same-direction scenes, which vehicle changed lanes and what the headway and braking were.
     The same-direction screen misses lane changes the map draws as a lane branch or merge (the car
     follows a lane link while moving across the road): look at scenes where two cars end up in one
     lane, and at `borderline_cut_in_thw_*` notes. It also does not see close passes between
     opposite-direction or turning cars; those belong to the crossing review.

5. **Write the labels.** One row per scene: `scene, sdc_aggressive, sdc_reason, ooi_aggressive,
   ooi_reason, confidence`. Each reason is one English sentence with the deciding rule and
   numbers ("SDC left-turned across the oncoming OOI at 3.6 s while both had been waiting since
   before the clip; CVC 21801(a) gives the OOI priority"). Name the jurisdiction. Merge duplicate
   scenes into one event when you count.

## Rules of judgement, and why

- **Judge for yourself.** Human or earlier labels are inputs, not ground truth; annotators often do
  not know the local traffic rules. Never tune a rule or threshold just to reproduce them; say where
  you disagree and why.
- **A violation that did not cost the other vehicle anything is not aggressive.** Going first
  against the law matters only if the other was made to stop again, braked hard or was cut close;
  otherwise it is a technical violation. Waiting at a stop sign is not a cost: it had to stop anyway.
- **A lawful first arrival is not aggressive**, even if the other vehicle braked or waited.
  Every vehicle has to slow for an intersection, and in the CAT scenes lawful first arrivals
  caused the same braking as ones first labelled aggressive, so braking cannot tell them apart.
- **Do not label "failed to stop" from a clip.** A stop before step 0 cannot be seen, so a vehicle
  creeping at its line at step 0 proves nothing; among 25 SDCs whose approach was observed none
  rolled through.
- **Unknown order is no verdict.** When two vehicles were already waiting at step 0, the one that
  goes while the other waits is ordinary turn-taking. Do not treat it as a tie to assign blame.
- **Both directions.** Judge the SDC toward the OOI and the OOI toward the SDC; usually at most
  one is aggressive. If the other vehicle was the actor, the victim is not aggressive.
- **Never present an unvalidated profile's result as a finding.** A draft profile is a starting
  point for a conversation with someone who knows the local code.

## Files

- `scripts/screen_crossing.py` and `scripts/screen_same_direction.py`: the two screens.
  `scripts/scenes.py`: scene loading (ScenarioNet and catk cache) and pair choice.
  `scripts/jurisdiction.py`: the profile engine (`python scripts/jurisdiction.py` self-tests it).
  `scripts/responsibility/`: the scene, lane-graph and right-of-way code from the CAT fork (see its
  `__init__.py` for the source commit and the two changes).
- `jurisdictions/*.json`: the profiles. `references/jurisdictions.md`: schema, rule catalogue, how
  to add a country.
- `references/rubric.md`: the definition, the physical rules for same-direction traffic, failure
  modes. `references/validation-us-california.md`: what was checked and its limits.
- `assets/human_sdc_aggressive_cat500.csv`: a person's SDC labels on CAT's 500 scenes with an
  earlier decision for each. Not ground truth: on review only the cut-in (342/482) holds up.
- For a whole dataset, `aggression_agent/` at the repository root runs this workflow
  unattended: both screens, triage (including the lane-join blind spot), a blind reviewer agent per
  flagged scene with the tracks, numbers and a rendered figure, a critic where the reviewer overrules
  the screens, and a rule auditor that proposes (never applies) screen changes. It reads its rules
  from this skill's rubric, "Rules of judgement" and profile. Its agents run on Claude (Claude
  Agent SDK) or on a local model behind an OpenAI-compatible server (vLLM, Ollama). See its README.
