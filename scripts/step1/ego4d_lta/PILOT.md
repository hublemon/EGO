# PILOT.md — Ego4D LTA Z=1 pilot-first validation procedure

This is the recommended order for standing up the Ego4D LTA Z=1 pipeline:
validate cheaply on a subset + restricted taxonomy first, then scale up.
It reuses the EK100 architecture unchanged (frozen V-JEPA2 backbone,
attentive probe, 3-head classifier, focal loss, class-mean Recall@5) --
see `docs/step1_action_anticipation.md` and
`src/ego/step1_action_anticipation/models/` for that shared code, and
`src/ego/datasets/ego4d.py` for what's new here.

## Prerequisites

- `fho_lta_train.json`, `fho_lta_val.json`, `fho_lta_taxonomy.json` (Ego4D LTA annotations)
- `ego4d.json` (optional, for scenario tags -- without it every sample's
  `scenario` is `"unknown"` and scenario-stratified sampling/breakdowns are
  meaningless)
- Ego4D LTA clips (or full_scale videos) downloaded, matching `dataset.video_source`
  in the config
- V-JEPA2 checkpoint at `checkpoints/vjepa2/vitl.pt` (already vendored, shared with EK100)

All of the above require Ego4D License Agreement access (AWS credentials
issued after approval) -- see `develop_report/` for this repo's current
status on that.

## Step 1: Pilot index + pilot taxonomy

Subsample ~10-20% of train clips and restrict to the top ~80 verbs / ~150
nouns, so the whole pipeline can be smoke-tested in minutes instead of hours:

```bash
python scripts/step1/ego4d_lta/build_lta_z1_index.py \
    --taxonomy <path>/fho_lta_taxonomy.json \
    --train-json <path>/fho_lta_train.json \
    --val-json <path>/fho_lta_val.json \
    --ego4d-json <path>/ego4d.json \
    --train-clip-fraction 0.15 \
    --top-verb 80 --top-noun 150 --pilot-mode exclude \
    --output-dir outputs/ego4d_lta/index_pilot
```

Console output reports (per the Task 1 spec): taxonomy size (`N_verb`,
`N_noun`), registered `(verb, noun)` action-combination count, boundary
policy and how many samples were truncated/excluded under it, and final
train/dev/heldout sample counts.

Then look at the class distribution before committing to a training run:

```bash
python scripts/step1/ego4d_lta/analyze_lta_stats.py \
    --index outputs/ego4d_lta/index_pilot/train.parquet \
    --output-dir outputs/ego4d_lta/stats_pilot
```

## Step 2: Feature extraction (pilot)

```bash
python scripts/step1/ego4d_lta/extract_features.py \
    --config configs/step1/ego4d_lta/pilot.yaml --split train
python scripts/step1/ego4d_lta/extract_features.py \
    --config configs/step1/ego4d_lta/pilot.yaml --split dev
```

Record wall-clock time / clip here -- this is the number that decides
whether the full-taxonomy extraction run (all train clips) is feasible on
the available compute before starting it.

## Step 3: Pilot training smoke test

```bash
python scripts/step1/ego4d_lta/train_lta_z1.py --config configs/step1/ego4d_lta/pilot.yaml
```

Check, in this order:
1. It completes all epochs without a shape/key error (confirms the
   AnticipationHead output-dimension swap and feature cache plumbing are
   correct).
2. Train loss decreases (confirms the focal loss / optimizer / label
   encoding aren't silently broken -- e.g. verb/noun ids swapped).
3. Dev class-mean Recall@5 (`outputs/ego4d_lta/runs/pilot/metrics.json`) is
   above chance level for the pilot's ~80/150-class taxonomy.
4. `outputs/ego4d_lta/runs/pilot/likelihood_entropy.jsonl` has one row per
   dev sample with finite, non-NaN `*_likelihood`/`*_entropy` values.

If any of these fail, fix it here -- at pilot scale -- before spending
compute on the full run.

## Step 4: Full taxonomy

Once the pilot is clean, rebuild the index without `--top-verb`/`--top-noun`/
`--train-clip-fraction`:

```bash
python scripts/step1/ego4d_lta/build_lta_z1_index.py \
    --taxonomy <path>/fho_lta_taxonomy.json \
    --train-json <path>/fho_lta_train.json \
    --val-json <path>/fho_lta_val.json \
    --ego4d-json <path>/ego4d.json \
    --output-dir outputs/ego4d_lta/index_full

python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/ego4d_lta/full.yaml --split train
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/ego4d_lta/full.yaml --split dev
python scripts/step1/ego4d_lta/train_lta_z1.py --config configs/step1/ego4d_lta/full.yaml
```

`configs/step1/ego4d_lta/full.yaml` sets `training.focal_gamma: 2.0` (the
EK100 baseline value) as a starting point -- Ego4D LTA's long tail is worse
than EK100's, so re-run with a higher gamma (3-4) as an A/B if head classes
still dominate the band breakdown in `metrics.json`.

Watch, per epoch, in the console and `metrics.json`:
- overall class-mean Recall@5 (verb / noun / action)
- head/mid/tail band Recall@5 breakdown (is the model only learning head classes?)
- per-scenario Recall@5 breakdown (is performance uneven across domains? --
  this is the diagnostic for multi-domain variance the spec calls out)

## Result interpretation -- read this before comparing numbers

- **Do not directly compare Ego4D LTA Recall@5 to EK100 Recall@5.** The
  action taxonomy here is several times larger than EK100's (EK100: 97
  verbs / ~300 nouns / a few thousand action combos actually observed in
  train; Ego4D LTA's full taxonomy is larger across the board -- see
  `outputs/ego4d_lta/index_full/action_registry.json` for the actual counts
  once built). A lower Recall@5 does not necessarily mean a worse model.
- **Pilot-taxonomy metrics are not comparable to full-taxonomy metrics**,
  for the same reason (different class count and difficulty) -- pilot is for
  pipeline validation and fast iteration only, never for reporting final
  numbers.
- **Report the actual measured values, not the config defaults**, in any
  write-up: exact registered action-combination count, exact train/dev/
  heldout sample counts, and the scenario distribution -- all written to
  `outputs/ego4d_lta/index_full/build_stats.json` and
  `outputs/ego4d_lta/stats_full/lta_stats.json` by the scripts above.

## Known open item

This pipeline has been implemented and unit-tested against synthetic
fixtures matching the documented Ego4D LTA JSON schema, but **not yet run
against the real annotation files** (this environment has no Ego4D access
credentials yet -- see `develop_report/`). `ego.datasets.ego4d.
parse_lta_annotations` resolves several candidate field names per logical
field and raises a clear, catalogued error (listing the record's actual
keys) if none match, specifically so that the first real run fails loudly
and locally-fixably instead of silently mis-parsing. If it does, the fix is
almost always adding the real key name to `_FIELD_CANDIDATES` in
`src/ego/datasets/ego4d.py`.
