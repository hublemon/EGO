# Step 1 Action Anticipation

Step 1 estimates immediate future action distributions from egocentric video
observations and exports Top-K verb, noun, and action-pair candidates with
raw logits and probabilities for Step 2.

Implementation status: **EK100 and Assembly101 baselines implemented**
(`ego step1 prepare|train|infer|evaluate`). Ego4D remains a scaffold
(`configs/step1/ego4d_vjepa2.yaml`, `src/ego/datasets/ego4d.py`) until a
first EK100 baseline is validated.

Verified end-to-end against real data and the real V-JEPA2 checkpoint on this
machine: `EK100Dataset`/`Assembly101Dataset` construction, video decode +
transform pipeline, backbone checkpoint loading (0 missing/mismatched keys),
a full `ego step1 train` -> `infer` -> `evaluate` CLI run producing
schema-valid `action_candidates.jsonl`, and `Assembly101Dataset` manifest
building against the full 293GB local Assembly101 download (102 train / 36
val videos, 24,743 / 9,148 samples). **Caveat:** `data/EPIC-KITCHENS/` on
this machine only has the ~44-video demo/validation-subset actually
downloaded (matching the old prototype's hardcoded `TARGET_VIDEOS`), not the
full EK100 train split -- `ego step1 prepare --config
configs/step1/ek100_vjepa2.yaml` will report zero train videos until the
remaining EK100 participant videos are downloaded into that directory.

This was refactored from a working prototype in the `EvE/V-JEPA2` repo
(`evals/action_anticipation_frozen/`, `scripts/*.py`), which is left in
place untouched as reference. See `third_party/versions.yaml` for the exact
upstream V-JEPA2 commit the vendored backbone code was copied from.

## Design decisions vs. the prototype

- **Map-style datasets, not WebDataset streaming.** `EK100Dataset` /
  `Assembly101Dataset` are plain `torch.utils.data.Dataset`s
  (`__len__`/`__getitem__`), matching `ego.datasets.base`. The prototype used
  a WebDataset streaming pipeline built for multi-node SLURM training; this
  repo targets single-machine runs, so the simpler map-style form was chosen.
- **Corrected observation/target invariant.** The prototype's training-time
  sampler (`anticipation_point`) could let the observation window extend
  into the target action segment. `ego.datasets.video_sampling.build_clip_window`
  always ends the observation `anticipation_time_sec` seconds before the
  target action's start frame, with a hard clamp -- see
  `tests/unit/test_video_sampling.py`.
- **Deterministic label mapping.** EK100's prototype assigned verb ids via
  `enumerate(set(...))` (Python set iteration order, not guaranteed stable).
  `ego.datasets.label_mapping.build_label_mapping` always sorts first.
- **One classifier per run, not a parallel hyperparameter sweep.** The
  prototype trained ~20 classifier instances per process for a learning-rate/
  weight-decay grid search. `configs/step1/*.yaml` config a single
  `training.learning_rate`/`weight_decay`; sweep externally over configs if
  needed.
- **Independent best checkpoints.** `train.py` saves `best_verb.pt`,
  `best_noun.pt`, `best_action.pt` (by validation class-mean recall@5) plus
  `latest.pt`, since the three heads' best epochs typically differ.
- **Feature cache for Assembly101.** Assembly101 is ~300GB; re-decoding video
  every epoch is impractical. `training.use_feature_cache: true` (see
  `configs/step1/assembly101_vjepa2.yaml`) runs the frozen backbone once,
  caches per-sample tokens under `dataset.feature_cache_dir`, and trains the
  classifier heads from cache afterward -- mirroring the prototype's
  `extract_features_a101.py` + `train_probe_a101.py` two-stage split, but
  driven by one config instead of two scripts.

## V-JEPA2 backbone

The encoder + predictor source is vendored under `third_party/vjepa2/`
(copied, not referenced in place, per the EGO data/model policy) and wrapped
by `src/ego/step1_action_anticipation/models/vjepa2_backbone.py`. The
checkpoint lives at `checkpoints/vjepa2/vitl.pt` (gitignored).

## Commands

```bash
ego step1 prepare  --config configs/step1/ek100_vjepa2.yaml
ego step1 train    --config configs/step1/ek100_vjepa2.yaml
ego step1 infer    --config configs/step1/inference.yaml
ego step1 evaluate --config configs/step1/inference.yaml
```

`prepare` loads annotations, resolves video paths, and fits the label
mapping without touching the model -- see `docs/datasets.md` for what it
reports.
