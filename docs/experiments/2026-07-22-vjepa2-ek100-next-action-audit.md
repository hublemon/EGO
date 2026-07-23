# V-JEPA 2 EK100 next-action audit

- Date: 2026-07-22
- Official repository: `facebookresearch/vjepa2`
- Audited commit: `204698b45b3712590f06245fbfba32d3be539812`
- Local clone: `third_party/vjepa2_official`

## Verdict

The temporal-semantics risk report is correct for the released EK100 code. The
validation config sets `anticipation_point=[0, 0]` and
`anticipation_time_sec=[1, 1]`. The decoder therefore ends the observation at
`current_action.end - 1s`, while the released decoder assigns the verb and noun
from that same current-action row. This is late action recognition, not a
strict next-action target.

The released training protocol is also inconsistent with the validation
protocol: it randomly samples `anticipation_point` in `[0, .25]` and the time
offset in `[.25, 1.75]`, but still supervises the same annotation row.

## Minimal correction implemented

The corrected protocol is:

```text
observation anchor = action_1.end - 1s
target             = immediate next annotation action_2 in the same video
retain pair only if action_2.start >= observation anchor
```

Changes are limited to the EK100 evaluation data path and its configs:

1. Sort annotations by video and start time, then shift verb/noun targets by
   one row within each video.
2. Remove each video's final action because it has no next target.
3. Remove overlapping pairs for which action 2 has already started by the
   `action_1.end - 1s` observation point.
4. Build train/validation class mappings from the shifted targets.
5. Fix training to the same deterministic `end - 1s` endpoint as validation.

On the local official EK100 annotations this produces:

| split | source rows | adjacent pairs | future-at-end−1s pairs | excluded overlap/no-next |
|---|---:|---:|---:|---:|
| train | 67,217 | 66,722 | 62,896 | 4,321 |
| validation | 9,668 | 9,530 | 8,915 | 753 |

Validation subsequently drops 351 pairs whose shifted action combination is
not present in the corrected training target set, leaving about 8,564 samples
before missing-video filtering.

## Is changing labels alone sufficient?

Only under a narrow contract. If the observation clips/features remain exactly
the same, a frozen representation cache can be reused and a label overlay is
mathematically sufficient to train a probe for “predict the next annotation
from this observation.” However, the dataset construction must still handle:

- grouping and ordering within the same video;
- the last action in each video;
- overlapping annotations that expose the next action early;
- train/validation class mappings based on shifted targets;
- validation labels and metrics, not training labels alone.

This corrected `action_1.end - 1s -> action_2` task also has a variable horizon
to the start of action 2. It is not the standard fixed-horizon EK100 protocol.
For the stronger claim “predict action 2 exactly one second before it starts,”
the observation must instead end at `action_2.start - 1s`; those clips and
features differ and must be regenerated.

## Feature reuse and public artifacts

The official repository does not implement an EK100 feature cache. It decodes
raw videos and executes the frozen backbone again in every epoch. Meta publishes
V-JEPA 2 backbone checkpoints and trained EK100 attentive-probe checkpoints,
but not the per-sample EK100 features used to train those probes.

For the corrected end−1s/next-label protocol, an existing cache made from the
same end−1s clips can be reused after applying shifted labels and the safe-pair
filter. A cache made at `action_2.start−1s` cannot be substituted, and moving to
that strict fixed-horizon protocol requires extraction again.

A third-party Hugging Face dataset, `sjmathy/epic_kitchen_100_resume`, advertises
ViT-g/384 cached tokens with the released fixed 4-second/end−1s setting. It is
not an official Meta artifact, is about 2.61 TB, and retains the original label
semantics, so it would still require provenance checks, next-label remapping,
and overlap filtering.

### Follow-up inspection of the third-party cache

The repository actually mixes two different feature packages:

1. `vjepa2_vitg384_4s_1sbefore/` is a complete 64-train/16-validation-shard
   cache. Its own manifest reports 1.823 TB train and 255 GB validation (about
   2.078 TB total). A remotely range-read sample was successfully loaded and
   contained:

   ```text
   tokens             [2, 9792, 1408], float16
   verb/noun/action   [2], int64
   anticipation_time [2] = 1.0
   metadata           video_id, narration_id, start_frame, stop_frame,
                      anchor_frame, source_video_path
   ```

   For the inspected sample, `stop_frame=202` and `anchor_frame=143`, confirming
   that this cache is anchored approximately one second before the **end of the
   same annotated action**. The 9,792 tokens are consistent with ViT-g/384:
   32 context frames / tubelet 2 × 24 × 24 = 9,216 encoder tokens, plus 576
   predicted-future tokens. Because `narration_id` and frame anchors are stored,
   the official EK100 CSV is sufficient to map each cached observation to its
   immediate next label and to remove next-actions already visible at the
   anchor. Raw videos are not required for that relabelled experiment.

2. `data/full_tokens/` is a separate package. Its downloadable manifests define
   observations ending at `target_action.start - 1s`, i.e. a strict standard
   anticipation endpoint, with 66,178 train and 9,595 validation manifest rows.
   However, only train tar parts 000--006 and 022--023 of an expected 000--023
   are present; validation has all 24 parts and reports eight missing feature
   files. Thus the raw strict-anticipation feature upload is not complete enough
   for full training without recovering the missing train parts or re-extracting.

The small `ek100_stage1_action_embeddings` ZIP is complete but is not a
substitute for either token cache. Each sample contains only a 768-dimensional
embedding of a custom model's predicted top-1 action plus `pred_action_id`; it
cannot retrain or faithfully evaluate the official attentive probe.

Consequently, “no video download” is feasible for the corrected
`current.end-1s -> next action` experiment, but it is not immediately runnable:
about 2.08 TB of complete tokens must be downloaded (or streamed), a cached-token
loader must be added, labels must be shifted using `narration_id`, and a new
probe must be trained. Streaming repeatedly is impractical for multi-epoch
training because it would transfer the multi-terabyte cache every epoch.

## One-H200 time estimate

The official config was designed for 64 GPUs and evaluates 20 probe
hyperparameter combinations concurrently. On one H200, the dominant cost is
re-running the frozen V-JEPA encoder/predictor over roughly 62.9k train and
8.6k validation clips in every epoch.

The closest measured local H200 run extracted 30,374 same-shape ViT-L clips in
about 1 hour 47 minutes (about 3.5 clips/s). Scaling that observation gives
about 5.5–6 hours for one full corrected EK100 train+validation feature pass.
Therefore:

- released on-the-fly loop, 20 epochs: about 4.5–6 days;
- extract/cache once, then train the probe: about 8–10 hours total;
- label-only retraining with an already valid same-endpoint cache: about
  3–4 hours for 20 epochs, depending on cache I/O and probe batch size.

These are planning estimates, not a fresh benchmark: the single H200 was at
100% utilization during this audit. A 200–500 sample timing run after it is free
should be used to tighten the estimate.

## Verification

- Both changed Python files pass `py_compile`.
- The official clone passes `git diff --check`.
- The pairing/class-map path was executed against the local official EK100 CSVs
  in the project's CUDA environment.
