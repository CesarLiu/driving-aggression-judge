# Jurisdiction profiles

A profile is a JSON file in `jurisdictions/` that says which rules settle who goes first at an
intersection without a working signal, and optionally between two vehicles that both have a
circular green. `scripts/jurisdiction.py` applies it; `python
scripts/jurisdiction.py` self-tests the engine, including that the left-hand profile is the exact
mirror of the California one.

| id | status |
|---|---|
| `us-california` | CVC 21800–21804 as applied to WOMD maps; checked by reviewing every flagged crossing in CAT's 500 WOMD scenes (see `validation-us-california.md`) |
| `us-california-statute` | CVC 21800–21802 in the statute's order, for maps whose junction control is complete; text checked, **unvalidated** |
| `china-draft` | Road Traffic Safety Law 44, Implementing Regulations 38, 51, 52; text checked, **unvalidated** |
| `germany-draft` | StVO §§ 8, 9, 10, 37; text checked, **unvalidated** |
| `left-hand-draft` | the California structure reflected for left-hand traffic; **not any country's law** |

"Text checked" means that each rule was compared on 2026-10-30 with the text of the law:
- California: Vehicle Code 21800, 21801, 21802 and 21804, from a published copy (california.public.law);
- China: the Implementing Regulations as revised on 2017-10-07, from a published copy (not the
  official gazette);
- Germany: the StVO on gesetze-im-internet.de.

The `vru` blocks (pedestrians and cyclists) were checked on 2026-10-31 against CVC 21451(a), 21453(b),
21950, 21952, 21954 and 21760 (california.public.law); StVO §§ 5(4), 9(3), 10, 25(3) and 26
(gesetze-im-internet.de); and the Road Traffic Safety Law as amended in 2021, arts. 43, 47 and 62,
its Implementing Regulations arts. 38 and 47 (zh.wikisource.org), and the Supreme People's Court's
Guiding Case No. 90 (court.gov.cn).

What the law leaves open is marked as an interpretation in the profile's `notes`.

Draft profiles print a warning and their results must not be reported as findings until someone
who knows the local traffic code has read the profile and a few labelled scenes agree with it.

## Schema

```json
{
  "id": "us-california",
  "driving_side": "right",
  "all_way_stop": {
    "first_arrival": true,
    "unknown_order": "none",
    "tie_rules": ["oncoming_left_yields", "yield_right"]
  },
  "uncontrolled_rules": ["oncoming_left_yields"],
  "t_junction": "ending_road_yields",
  "green_rules": ["oncoming_left_yields"],
  "status": "...",
  "notes": ["..."]
}
```

- `first_arrival`: the first vehicle to arrive and stop goes first. When false, the tie rules
  decide every all-way-stop pair.
- `unknown_order`: when both vehicles were already waiting at the start of the clip, `"none"` (the
  default in every profile) gives no verdict; `"tie"` applies `tie_rules`. Leave it at `"none"`: on
  review, every case `"tie"` blamed was ordinary turn-taking.
- An arrival before the clip (`arrived == -1`) counts as earlier than any arrival seen in the clip.
- `uncontrolled_rules`: no sign and no light; rules applied in order.
- `t_junction`: `"ending_road_yields"` (applied before `uncontrolled_rules`) or `"none"`. The rule
  `ending_road_yields` can also be placed inside `uncontrolled_rules`, where the law ranks it after
  another rule (California: only on a simultaneous entry, after `first_entered`).
- `green_rules` (optional): both vehicles entered on a circular green (no arrow), as in China's
  reg. 51(7) or StVO § 9(3), (4). Without it, the copied CAT right-of-way code decides those
  pairs by California's rules: left turn yields, then whoever entered first.
- A two-way stop always makes the vehicle at the sign yield to cross traffic. Red lights, arrows,
  unknown lights and driveways are decided by the copied CAT code (`scripts/responsibility/`),
  not by the profile. In all three countries the vehicle on red, or entering from private ground,
  yields.

## Pedestrians and cyclists (`vru`, read by `scripts/screen_vru.py`)

```json
"vru": {
  "crosswalk": "on",
  "unmarked_crosswalk_at_intersection": true,
  "unsignalised_road": "care",
  "sidewalk_crossing": "yield",
  "turning_yields_to_pedestrians": true,
  "turning_yields_to_cyclists": true,
  "passing_m": {"cyclist": [0.91, 0.91], "pedestrian": [null, null]},
  "notes": ["..."]
}
```

`jurisdiction.vru_duty` turns these into the vehicle's duty toward a VRU whose path it crossed:
`yield` (the VRU had priority), `care` (no priority, but due care) or `vehicle_rules` (a cyclist
crossing like a vehicle, judged by the vehicle rules above). In order:

1. A cyclist: `yield` if the vehicle had turned and the cyclist went straight beside it or
   oncoming (`turning_yields_to_cyclists`), else `vehicle_rules`.
2. A pedestrian who stepped out when the vehicle could no longer stop: `care`.
3. A pedestrian not yet crossing when the vehicle reached the conflict: `care`. Crossing means on
   the crosswalk, or on the carriageway; with `crosswalk` `"on_or_waiting"` also waiting at the
   crosswalk's edge.
4. The vehicle's light. Unknown: `care`. Red: `yield`. A green arrow: `care` (a protected turn: the
   pedestrians it crosses face a red). Circular green: `yield` if the vehicle turned into the road
   the pedestrian crosses (`turning_yields_to_pedestrians`), else `care`.
5. Without a light:
   - off the road (a driveway): `sidewalk_crossing`;
   - on a marked crosswalk: `yield`;
   - the vehicle turned into the road crossed (`turning_yields_to_pedestrians`): `yield`;
   - at a junction without markings: `yield` if `unmarked_crosswalk_at_intersection`;
   - elsewhere: `unsignalised_road`.

`passing_m` gives the lateral distance for overtaking a cyclist or pedestrian, `[in town, outside
town]`; the screen takes a lane speed limit above 50 km/h as outside town. `null` means the law
names no distance: the screen then applies a 1 m physical fallback and flags it
`no_statutory_distance`.

| situation | California | Germany | China |
|---|---|---|---|
| pedestrian on a crosswalk without a light | yield (CVC 21950(a)) | yield (§ 26(1)) | stop and yield (RTSL 47(1)) |
| pedestrian waiting at that crosswalk's edge | care | yield ("erkennbar benutzen wollen", § 26(1)) | care; staying on the crosswalk counts as on it (Guiding Case 90) |
| pedestrian crossing at a junction without markings | yield (unmarked crosswalk, 21950(a)) | care, yield if the vehicle turns (§ 9(3)) | yield where there is no signal (RTSL 47(2)) |
| pedestrian crossing mid-block, no signal | care (21954) | care (§ 25(3)) | yield (RTSL 47(2)) |
| turning on green across a crossing pedestrian | yield (21451(a)) | yield (§ 9(3)) | yield (reg. 38(1)(1)) |
| turning across a cyclist going straight | yield (21451(a), 21801(a); cyclists are drivers, 21200) | yield (§ 9(3)) | yield (reg. 38(1)(1)) |
| leaving a driveway across the sidewalk | yield (21952) | yield (§ 10) | no specific article: care |
| overtaking a cyclist | 3 ft = 0.91 m; change lanes if another is free (21760(c), not checked) | 1.5 m in town, 2 m outside (§ 5(4)) | no distance: fallback |
| overtaking a pedestrian | no distance: fallback | 1.5 m / 2 m (§ 5(4)) | no distance: fallback |

What the maps cannot show: whether a crosswalk is a zebra (Germany: one without a light is taken to
be), pedestrian signals (inferred from the vehicle's light and whether it turned), kerbs (the
carriageway is within 2 m of a lane centreline), and whether a pedestrian waved the car on (China's
exception in Guiding Case 90).

## Rules (ordered lists; the first that names a holder wins)

| rule | meaning | sources |
|---|---|---|
| `oncoming_left_yields` | a left or U-turner yields to oncoming traffic, straight on or turning right; two oncoming left turners: no holder | CVC 21801(a), 21800(e)(3); StVO § 9(3), (4) |
| `oncoming_right_yields` | right-turn counterpart for left-hand traffic | – |
| `oncoming_right_yields_left` | of two oncoming turners, the right turner yields to the left turner | China reg. 51(7), 52(4) |
| `turner_yields_straight` | a vehicle that turns yields to one going straight on | China reg. 51(7), 52(3) |
| `yield_right` | the vehicle on the left yields to the one on the right | CVC 21800(b)(1), (c); StVO § 8(1); China reg. 52(2) |
| `yield_left` | the mirror of `yield_right` | – |
| `ending_road_yields` | at a T junction the road that ends yields to the one that continues | CVC 21800(b)(1), (b)(2) |
| `first_entered` | whoever entered the intersection first (entries more than 1 s apart) | CVC 21800(a) |

The same situation is settled differently in each place:

| situation | California | Germany | China |
|---|---|---|---|
| T junction, no sign | the road that ends yields (on a simultaneous entry) | rechts vor links (§ 8(1) covers Einmündungen) | the car from the right (reg. 52(2)) |
| left turner vs oncoming right turner | left turner yields | left turner yields (§ 9(4)) | right turner yields (reg. 52(4)) |
| one already in the intersection | it goes first (21800(a)) | no such rule | no such rule |
| all-way stop | first to enter, then yield right | no all-way stop in the StVO | no arrival-order rule |
| both on green, one turns | left turner yields to oncoming (CAT code) | the turner yields to oncoming (§ 9(3), (4)) | the turner yields to straight-on; right yields to left (reg. 51(7)) |

"Left", "right" and "oncoming" are all relative to each vehicle's own heading, so a left-hand
profile needs no coordinate flip.

## Adding a jurisdiction

1. Copy the closest profile, give it a new `id` and set `status` to a string starting with
   `draft`.
2. Write the statute's rules into `all_way_stop`, `uncontrolled_rules` and `t_junction`, and the
   pedestrian and cyclist rules into `vru`, citing the article in `notes`. If you need a rule that is not in the table, add a function to
   `scripts/jurisdiction.py` (signature `(a, b, rel) -> "a" | "b" | None`) and register it in
   `RULES`.
3. Run `python scripts/jurisdiction.py`. Then screen a few dozen scenes from that country, look at
   every flagged scene and a sample of unflagged ones with a person who knows the local rules, and
   only then change `status` from draft and record the check in a `validation-<id>.md`.

Datasets differ in region: every WOMD scene is from US cities (right-hand traffic; California-like
rules are a reasonable default), nuPlan's Singapore scenes are left-hand, and datasets from China
or Germany need the matching profile.

## The profiles on CAT's 500 scenes (2026-10-30)

These scenes are all from US cities, so the other profiles are judged on the wrong country's
traffic. The run shows what each profile changes; it says nothing about their accuracy.

| profile | pairs with a holder | violations | aggressive candidates | holder differs from `us-california` |
|---|---|---|---|---|
| `us-california` | 156 | 16 | 0 | – |
| `us-california-statute` | 189 | 14 | 1 | 40 |
| `germany-draft` | 210 | 55 | 2 | 80 |
| `china-draft` | 211 | 67 | 2 | 86 |

The differences come from the following rules:
- **California as written:** the differences come from first-entered at junctions the map shows
  without control (32 of 40).
- **Germany and China:** they come from yield-to-the-right:
  - at those junctions (33);
  - at all-way stops where `us-california` gives no verdict or goes by arrival (25);
  - at T junctions (9).
- **China only:** 8 pairs of a left turner and an oncoming right turner, where the right turner
  is now the one to yield.

Many of these extra violations are exactly what `us-california` was tuned to avoid: on WOMD maps
an "uncontrolled" junction usually has signs the map lacks.

No pair in these scenes had both vehicles on a circular green at a crossing, so `green_rules`
are tested only by the self-test.
