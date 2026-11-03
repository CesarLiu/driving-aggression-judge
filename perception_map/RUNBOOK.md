# Runbook: right of way on perception-only maps

How far right-of-way verdicts change when the map has only lane lines (no lights, stop signs, links
or junction lanes), and how much of that each fix wins back. The reference throughout is the same
rules on the full WOMD map (`screen_crossing.csv`), not human labels.

| Experiment | Question | Code |
|---|---|---|
| A topology | Rebuilding junction lanes from tracks: how much is won back with true controls? | `perception_map/topology.py` |
| B controls | Inferring controls on the full map: how much is won back? Does the second stage fix all-way stops? | `control_inference/infer_controls.py` |
| C combined | Rebuilt topology plus inferred controls (the real perception-only setting) | `infer_controls.py --map rebuilt` |
| D pooling | Do the tracks of other scenes at the same place help A and C? | `perception_map/pool.py`, `--pool` |

## Results so far (6,010 junction scenes, all checked)

| Setting | Case unchanged | Holder unchanged | SDC violation precision / recall |
|---|---|---|---|
| Degraded map, no links, no controls | 24.0% | 50.0% | 18% / 32% |
| Full map, no controls | 38.1% | 64.7% | 23% / 64% |
| Full map, inferred controls (models v1) | 69.8% | 80.5% | 26% / 66% |
| Full map, inferred, unsure lanes unknown (gated) | 64.4% | 76.5% | 33% / 54% |
| Rebuilt topology, true controls moved by id | 60.2% | 70.8% | 31% / 55% |
| Rebuilt topology, true controls moved by geometry | 67.7% | 76.6% | 38% / 60% |
| Rebuilt topology, no controls | 25.7% | 50.8% | 16% / 43% |
| **Rebuilt topology + inferred controls (v1 models, trained on the full map)** | **34.2%** | **54.3%** | 20% / 44% |

The combination collapses because the v1 models were trained on full-map lanes. On a rebuilt map
they see 0 to 4 connectors per scene (only movements driven in the 9 s clip) and predict "none".
Hence steps 4 to 6: train on the domain the models are applied to.

What we know about pooling (step 2):

- WOMD gives each drive its own coordinate frame. Checked on 40 random scenes: against the nearest
  other drives, at most 11% of nearby lane points coincide within 0.3 m, which is chance level.
  Repeat visits to one junction therefore **cannot** be found by coordinates.
- What can be pooled are the overlapping windows of the same drive (median 2 per scene). They are
  usually shifted by only 1 to 2 s and together cover about 11 s instead of 9.1 s (median), so
  expect a small effect from D.

## 0. Setup

Requires Python with numpy, scipy, scikit-learn and pandas. Run from `driving-aggression-judge`:

```bash
cd driving-aggression-judge
export SCENES=E:/womd_val_test/validation_interactive_sdc_scenes   # ScenarioNet scene pickles
export REAL=E:/womd_val_test/judge_sdc/screen_crossing.csv          # full-map screen of the SDC pairs
export OUT=E:/womd_val_test                                         # everything below is written here
export P=8                                                          # parallel parts = CPU cores
export OMP_NUM_THREADS=1      # one thread per worker: sklearn threads otherwise stall each other
A="--scenes $SCENES --pairs $REAL"
mkdir -p $OUT/topology $OUT/controls $OUT/pool
```

If `$REAL` does not exist yet, create it with the full-map screen (SDC against the other object of
interest):

```bash
python .claude/skills/judge-driving-aggression/scripts/screen_crossing.py --scenes $SCENES \
    --profile us-california --out $REAL
```

The list of junction scenes (those whose full-map verdict depends on controls or junction lanes):

```bash
python - <<'EOF'
import os, pandas as pd
r = pd.read_csv(os.environ["REAL"])
NJ = ("same-direction", "following", "no-conflict", "paths-do-not-cross", "too-short", "driveway")
open(os.environ["OUT"] + "/topology/junction_scenes.txt", "w").write("\n".join(r[~r.case.isin(NJ)].scene))
EOF
wc -l $OUT/topology/junction_scenes.txt    # 6010 on the WOMD validation interactive set
```

A helper that runs one command in `$P` parts in parallel and waits; `{k}` becomes the part number:

```bash
par() { for k in $(seq 0 $((P-1))); do eval "${1//\{k\}/$k} > ${2//\{k\}/$k} 2>&1" & done; wait; }
```

## 1. Smoke test (~5 min; run it first)

Not yet run end to end: evaluation with `--map rebuilt` on models with a second stage, and topology
with `--pool`, because the data drive was offline. Each command uses about 29 scenes (`--part 0/400`),
and each must finish without a traceback:

```bash
S=$OUT/smoke; mkdir -p $S
python perception_map/pool.py extract --scenes $SCENES --out $S/pool --part 0/400
python perception_map/pool.py index --pool $S/pool --pairs $REAL
python perception_map/topology.py run $A --score-out $S/score --screen-out $S/screen --part 0/400 \
    --modes rebuilt_geo,rebuilt_none,no_links_none --pool $S/pool
python control_inference/infer_controls.py extract $A --out $S/fx_rb --map rebuilt --pool $S/pool --part 0/400
OMP_NUM_THREADS=4 python control_inference/infer_controls.py train --features $S/fx_rb --out $S/m.pkl --folds 3
python control_inference/infer_controls.py evaluate $A --models $S/m.pkl --out $S/eval --part 0/400 \
    --map rebuilt --pool $S/pool --modes inferred,gated,no_controls --junction-only
```

The smoke pool holds only these 29 scenes, so they have no place-mates and `--pool` changes
nothing here. That is expected; it only checks that the code paths run.

## 2. Track pool (D): ~15 to 25 min, 2.1 GB

One file of vehicle tracks per scene, plus an index of the other scenes at each scene's place.
Extraction is limited by disk reads, so 2 to 4 workers are enough:

```bash
for k in 0 1 2 3; do python perception_map/pool.py extract --scenes $SCENES --out $OUT/pool --part $k/4 \
    > $OUT/pool/extract_$k.log 2>&1 & done; wait
python perception_map/pool.py index --pool $OUT/pool --pairs $REAL
```

Expected: `11452 scenes, ~3800 places; other scenes pooled per scene: median 2`.

## 3. A: topology with true controls, plain and pooled (~6 min each with 8 parts)

```bash
J="--only $OUT/topology/junction_scenes.txt --modes rebuilt,rebuilt_geo,no_links,rebuilt_none,no_links_none"
par "python perception_map/topology.py run $A --score-out $OUT/topology/score_a --screen-out $OUT/topology/screen_a --part {k}/$P $J" "$OUT/topology/a_{k}.log"
par "python perception_map/topology.py run $A --score-out $OUT/topology/score_ap --screen-out $OUT/topology/screen_ap --part {k}/$P $J --pool $OUT/pool" "$OUT/topology/ap_{k}.log"
python perception_map/topology.py report --real $REAL --score $OUT/topology/score_a --screen $OUT/topology/screen_a --out $OUT/topology/report_a.md
python perception_map/topology.py report --real $REAL --score $OUT/topology/score_ap --screen $OUT/topology/screen_ap --out $OUT/topology/report_ap.md
```

Sanity check: in `report_a.md` the `rebuilt map, true controls moved by geometry` section must
show case 67.7% and holder 76.6% (the table above). The `run` command resumes a stopped part where
it left off; to start a part fresh, delete its CSVs.

## 4. Control features (B, C, D): ~10 min each with 8 parts

```bash
for f in full rb rbp; do mkdir -p $OUT/controls/feats_$f; done
par "python control_inference/infer_controls.py extract $A --out $OUT/controls/feats_full --part {k}/$P" "$OUT/controls/x_full_{k}.log"
par "python control_inference/infer_controls.py extract $A --out $OUT/controls/feats_rb --map rebuilt --part {k}/$P" "$OUT/controls/x_rb_{k}.log"
par "python control_inference/infer_controls.py extract $A --out $OUT/controls/feats_rbp --map rebuilt --pool $OUT/pool --part {k}/$P" "$OUT/controls/x_rbp_{k}.log"
tail -qn 1 $OUT/controls/x_*_*.log
```

The last line of each log lists the candidate lanes per class. On the rebuilt map expect about a
quarter of the full map's candidate lanes, with labels moved over from the true controls by geometry.
`feats_full` is the same as the existing `controls/feats` (with lane starts); if that folder is
complete, it can be used instead.

## 5. Train (~20 to 60 min per bundle; one at a time, all threads)

Each bundle holds per fold the lane model, the light-state model, the junction model and (unless
`--no-context`) the second-stage lane model, which is trained with nested cross-fitting (25 lane-model
fits). Run with the machine otherwise idle, so all threads are free:

```bash
export OMP_NUM_THREADS=$P
C=$OUT/controls
python control_inference/infer_controls.py train --features $C/feats_full --out $C/models_full2.pkl > $C/train_full2.log 2>&1
python control_inference/infer_controls.py train --features $C/feats_rb --out $C/models_rb0.pkl --no-context > $C/train_rb0.log 2>&1
python control_inference/infer_controls.py train --features $C/feats_rb --out $C/models_rb.pkl > $C/train_rb.log 2>&1
python control_inference/infer_controls.py train --features $C/feats_rbp --out $C/models_rbp.pkl > $C/train_rbp.log 2>&1
export OMP_NUM_THREADS=1
```

Each run writes its report next to the bundle (`models_*.txt`). For the all-way question, read
`junction, from the lane model's predictions` and `junction, from the second stage's predictions`:
the `all-way` row (recall 0.29 for the old junction model) and the confusion between `two-way` and
`all-way`.

## 6. Evaluate on the junction scenes (~6 min each with 8 parts)

```bash
C=$OUT/controls
E="--modes inferred,gated,no_controls --junction-only"
ev() {  # ev <models> <out> [extra options]
  par "python control_inference/infer_controls.py evaluate $A --models $C/$1 --out $C/$2 --part {k}/$P $E ${3:-}" "$C/$2_{k}.log"
  python control_inference/infer_controls.py compare --real $REAL --eval $C/$2 --out $C/compare_$2.md > /dev/null
}
ev models.pkl      eval_full_v1                           # B, before: must reproduce 69.8% / 80.5%
ev models_full2.pkl eval_full2                            # B, second stage
ev models.pkl      eval_rb_v1 "--map rebuilt"             # C, before: must reproduce 34.2% / 54.3%
ev models_rb0.pkl  eval_rb0   "--map rebuilt"             # C, trained on rebuilt maps
ev models_rb.pkl   eval_rb    "--map rebuilt"             # C, plus second stage
ev models_rbp.pkl  eval_rbp   "--map rebuilt --pool $OUT/pool"   # C + D
```

`compare` merges the folders it is given by mode name, so each folder gets its own report
(`compare_<name>.md`). In each, read the `junction scenes` sections.

## 7. Write down

Fill in from `compare_*.md` (mode `inferred`, and `gated` in brackets) and `report_*.md`:

| Setting | Case | Holder | Holder reversed | SDC viol. P / R | OOI viol. P / R |
|---|---|---|---|---|---|
| A rebuilt_geo, plain / pooled | | | | | |
| B full map v1 / second stage | | | | | |
| C rebuilt, v1 / rb0 / rb / rbp | | | | | |

How to read it:

- **Training on the domain (C rb0 vs rb_v1)**: how much of the collapse from 67.7% to 34.2% comes
  from applying full-map models to rebuilt maps.
- **Second stage (B full2 vs v1, C rb vs rb0)**: whether all-way stops get better, both in
  the junction confusion table in `models_*.txt` and in fewer `stop-sign <-> all-way-stop` changes
  in the `commonest case changes` line.
- **Pooling (A plain vs pooled, C rbp vs rb)**: expect a small effect, since the windows add only
  about 2 s.
- Violation precision is the number that decides whether verdicts on perception-only maps can be
  used at all; up to now it is 20 to 55%.

## Caveats

- The folds are 500 m cells of each drive's own frame. Cells therefore group unrelated drives at
  random, which is harmless. Overlapping windows of one drive share a frame and almost always fall
  in one cell, but a window near a cell border can land in another fold than its place-mates; a
  strict split would fold by `pool.py` place (not implemented).
- `extract` and `evaluate` rewrite a whole part when rerun; only `topology.py run` resumes.
- RAM: the pool is read file by file, so 8 workers fit in 16 GB.
- The previous runs' outputs (`topology/score`, `screen`, `screen_geo`, `controls/eval`,
  `eval_gated`, `models.pkl` = v1) stay as they are. Nothing above overwrites them.
