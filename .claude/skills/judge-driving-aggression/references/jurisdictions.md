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

## Rules (ordered lists; the first that names a holder wins)

| rule | meaning |
|---|---|
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
2. Write the statute's rules into `all_way_stop`, `uncontrolled_rules` and `t_junction`, citing the
   article in `notes`. If you need a rule that is not in the table, add a function to
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
