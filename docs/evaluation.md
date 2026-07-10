# Evaluation

Planned evaluation areas:

- Step 1 action anticipation accuracy and calibration.
- Step 2 policy decision quality and reward-component behavior.
- Step 3 trigger efficiency, replanning quality, and long-horizon trajectory metrics.

## Step 1 (implemented)

`src/ego/step1_action_anticipation/metrics.py` has the pure metric
functions; `evaluate.py` reads an exported `action_candidates.jsonl` and
reports, per run:

- Verb / Noun / Action Recall@K (instance-level Top-K hit rate)
- Verb / Noun / Action Class-Mean Recall@K
- Verb+Noun joint Recall@K (both heads independently correct within Top-K)
- Head/tail class-mean recall split (most-frequent 20% of classes vs. the rest)
- Prediction class distribution (Top-1 prediction counts per verb/noun)

Output: `outputs/step1/<experiment>/metrics.json` and `class_distribution.csv`.

`train.py`'s per-epoch validation instead calls `class_mean_recall` directly
on batched logits (see `metrics.py`), independent of the JSONL export path.

Step 2 and Step 3 evaluation are pending implementation.
