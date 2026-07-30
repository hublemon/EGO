#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_PYTHON="/root/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python"
PYTHON="${PYTHON_BIN:-$DEFAULT_PYTHON}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python)"
fi

export PYTHONPATH="$REPO/src"
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

ADAPTIVE_INDEX="$REPO/src/ego/step1_action_anticipation/goalstep/index_adaptive_transition_mr24x8"
HISTORY_INDEX="$REPO/src/ego/step1_action_anticipation/goalstep/index_adaptive_transition_mr24x8_history_k8"
CACHE="$REPO/../datasets/Ego4D/goalstep_feature_cache_adaptive_transition_mr24x8_vna"
STORE="$REPO/../datasets/Ego4D/goalstep_history_context_store_adaptive_transition_mr24x8"
VISUAL_CONFIG="$REPO/configs/step1/goalstep/z1_adaptive_transition_mr24x8_vna_ep10.yaml"
VISUAL_CHECKPOINT="$REPO/outputs/goalstep/runs/z1_adaptive_transition_mr24x8_vna_ep10/best_action_top5.pt"
CONFIG="$REPO/configs/step1/goalstep/z1_adaptive_transition_history_context_k8_vna_ep10.yaml"
RUN="$REPO/outputs/goalstep/runs/z1_adaptive_transition_history_context_k8_vna_ep10"
LOGS="$RUN/logs"

mkdir -p "$LOGS"
cd "$REPO"

trap 'code=$?; echo "[$(date -u +%FT%TZ)] ERROR exit=$code" | tee -a "$LOGS/pipeline.log"; exit "$code"' ERR

if [[ -f "$RUN/final_metrics.json" ]]; then
  echo "[$(date -u +%FT%TZ)] already complete: $RUN/final_metrics.json" \
    | tee -a "$LOGS/pipeline.log"
  exit 0
fi

echo "[$(date -u +%FT%TZ)] build adaptive K=8 history index" \
  | tee -a "$LOGS/pipeline.log"
"$PYTHON" \
  "$REPO/src/ego/step1_action_anticipation/goalstep/build_goalstep_adaptive_history_index.py" \
  --adaptive-index-dir "$ADAPTIVE_INDEX" \
  --output-dir "$HISTORY_INDEX" \
  --history-length 8 \
  >"$LOGS/index.log" 2>&1

echo "[$(date -u +%FT%TZ)] derive compact adaptive history store; no V-JEPA extraction" \
  | tee -a "$LOGS/pipeline.log"
"$PYTHON" "$REPO/scripts/step1/goalstep/prepare_history_context_store.py" \
  --source-index-dir "$ADAPTIVE_INDEX" \
  --cache-dir "$CACHE" \
  --output-dir "$STORE" \
  --visual-config "$VISUAL_CONFIG" \
  --visual-checkpoint "$VISUAL_CHECKPOINT" \
  --omit-recognition-logits \
  --split all \
  --batch-size 32 \
  --num-workers 8 \
  --shard-size 1024 \
  --device cuda \
  >"$LOGS/store.log" 2>&1

echo "[$(date -u +%FT%TZ)] train adaptive-history Phase 1 for 10 epochs" \
  | tee -a "$LOGS/pipeline.log"
"$PYTHON" \
  "$REPO/src/ego/step1_action_anticipation/goalstep/train_goalstep_history_context.py" \
  --config "$CONFIG" \
  >"$LOGS/train.log" 2>&1

echo "[$(date -u +%FT%TZ)] adaptive-history Phase 1 complete" \
  | tee -a "$LOGS/pipeline.log"
