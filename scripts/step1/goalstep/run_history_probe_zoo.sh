#!/usr/bin/env bash
set -Eeuo pipefail

# Phase-2 is intentionally serial after the completed Phase-1 default arm.
# Run this script inside tmux. Passing --resume is safe on a fresh directory
# and enables provenance-checked recovery from the last committed zoo epoch.

REPO=/root/nvme/migration/jihun/EGO_jihun2
PYTHON=/root/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
CONFIG=configs/step1/goalstep/z1_history_context_probe_zoo_ep10.yaml
PHASE1_RUN=outputs/goalstep/runs/z1_history_context_k8_vna_ep10
PHASE1_CROSSFIT_JSON="$PHASE1_RUN/history_context_vs_p0a_results.json"
PHASE1_CROSSFIT_SCORES="$PHASE1_RUN/history_context_vs_p0a_oof_scores.pt"
RUN_DIR=outputs/goalstep/runs/z1_history_context_probe_zoo_ep10
LOG_PATH="$RUN_DIR/logs/train.log"
EVAL_LOG_PATH="$RUN_DIR/logs/champion_eval.log"

cd "$REPO"
mkdir -p "$RUN_DIR/logs"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

phase1_ready() {
  "$PYTHON" - "$PHASE1_RUN/final_metrics.json" "$PHASE1_RUN/val_predictions/epoch_10.pt" \
    "$PHASE1_CROSSFIT_JSON" "$PHASE1_CROSSFIT_SCORES" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

import torch

final_path, epoch10_path, result_path, score_path = map(Path, sys.argv[1:])
if not all(path.is_file() and path.stat().st_size > 0 for path in (
    final_path, epoch10_path, result_path, score_path
)):
    raise SystemExit(1)
result = json.loads(result_path.read_text(encoding="utf-8"))
try:
    scores = torch.load(score_path, map_location="cpu", weights_only=True)
except TypeError:
    scores = torch.load(score_path, map_location="cpu")
valid = (
    result.get("phase") == "Phase-1 crossfit selection and P0-a-aware final ensemble"
    and result.get("sample_count") == 6960
    and scores.get("kind") == "goalstep_history_context_crossfit_oof_scores"
    and len(scores.get("sample_ids", [])) == 6960
)
raise SystemExit(0 if valid else 1)
PY
}

printf '[%s] waiting for Phase-1 training and P0-a-aware crossfit evaluation\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_PATH"
while ! phase1_ready; do
  sleep 15
done

printf '[%s] Phase-1 crossfit artifact verified; starting/resuming Phase-2 11-arm shared-loader zoo\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_PATH"
set +e
"$PYTHON" src/ego/step1_action_anticipation/goalstep/train_goalstep_history_probe_zoo.py \
  --config "$CONFIG" \
  --resume \
  2>&1 | tee -a "$LOG_PATH"
status=${PIPESTATUS[0]}
set -e
printf '[%s] Phase-2 trainer exit=%s\n' "$(date -u +%FT%TZ)" "$status" | tee -a "$LOG_PATH"
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

printf '[%s] Phase-2 training complete; starting leakage-safe outer-fold evaluator\n' \
  "$(date -u +%FT%TZ)" | tee -a "$EVAL_LOG_PATH"
set +e
"$PYTHON" scripts/step1/goalstep/evaluate_history_probe_zoo_vs_p0a.py \
  --phase1-oof "$PHASE1_CROSSFIT_SCORES" \
  --expected-epochs 10 \
  --alpha-step 0.05 \
  --bootstrap-samples 10000 \
  --seed 42 \
  2>&1 | tee -a "$EVAL_LOG_PATH"
eval_status=${PIPESTATUS[0]}
set -e
printf '[%s] Phase-2 outer-fold evaluator exit=%s\n' \
  "$(date -u +%FT%TZ)" "$eval_status" | tee -a "$EVAL_LOG_PATH"
exit "$eval_status"
