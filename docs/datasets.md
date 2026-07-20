# Datasets

Dataset adapters: `src/ego/datasets/{ek100,assembly101,ego4d}.py`, all
implementing the common interface in `src/ego/datasets/base.py`. Ego4D is
still a scaffold.

Raw videos, frames, processed features, and local path manifests must not be
committed. `data/*` is gitignored except `data/README.md`, `data/demo/`, and
`data/manifests/` (see `.gitignore`).

## Layout

```text
data/
├── EPIC-KITCHENS/<participant>/videos/<video_id>.MP4   # EK100 videos
├── annotations/EPIC_100_{train,validation,validation_subset}.csv
├── annotations/EPIC_100_{verb,noun}_classes.csv         # canonical text labels
└── Assembly101/
    ├── annotations/fine-grained/{train,validation,test}.csv
    ├── videos/recordings/<sequence>/HMC_21176875_mono10bit.mp4
    └── feature_cache/                                    # written by `ego step1 train` (use_feature_cache)
```

Both datasets were moved (not copied) here from the earlier `EvE/V-JEPA2`
prototype's `data/` directory.

## Label mapping and split filtering

`ego.datasets.label_mapping.build_label_mapping` fits verb/noun/action ids
on the train split only (sorted, deterministic). Validation rows whose
`(verb, noun)` pair never appears in train are dropped by
`build_ek100_manifest` / `build_assembly101_manifest` before the dataset is
constructed, so `LabelMapping.encode_*` never encounters an unseen class in
practice -- `check_mapping_covers_split` exists as an explicit assertion for
callers that skip that filtering step.

## Observation sampling

`ego.datasets.video_sampling.build_clip_window` samples the
`frames_per_clip`-length observation window ending strictly before the
target action's start frame -- see `docs/step1_action_anticipation.md` for
why this differs from the prototype.

## `ego step1 prepare` output

Running `ego step1 prepare --config configs/step1/<dataset>_vjepa2.yaml`
reports train/val sample and video counts, verb/noun/action class counts,
and any videos referenced in annotations but missing on disk, and writes
`outputs/step1/<experiment>/dataset_summary.json`.
