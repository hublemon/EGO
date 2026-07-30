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

AUX_INDEX="$REPO/src/ego/step1_action_anticipation/goalstep/index_lta_aux_end_m1_lobs8"
AUX_CONFIG="$REPO/configs/step1/goalstep/z1_lta_aux_end_m1_lobs8.yaml"
AUX_CACHE="$REPO/../datasets/Ego4D/lta_aux_feature_cache_end_m1_lobs8"
DIRECT_CONFIG="$REPO/configs/step1/goalstep/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux.yaml"
DIRECT_RUN="$REPO/outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux"
P0_RUN="$REPO/outputs/goalstep/runs/history_context_phase0_ltaaux"
STORE="$REPO/../datasets/Ego4D/goalstep_history_context_store_ltaaux"
PHASE1_CONFIG="$REPO/configs/step1/goalstep/z1_history_context_k8_vna_ep10_ltaaux.yaml"
PHASE1_RUN="$REPO/outputs/goalstep/runs/z1_history_context_k8_vna_ep10_ltaaux"
LOGS="$DIRECT_RUN/logs"

mkdir -p "$LOGS"
cd "$REPO"

trap 'code=$?; echo "[$(date -u +%FT%TZ)] ERROR exit=$code" | tee -a "$LOGS/pipeline.log"; exit "$code"' ERR

milestone() {
  echo "[$(date -u +%FT%TZ)] $1" | tee -a "$LOGS/pipeline.log"
}

if [[ -f "$PHASE1_RUN/history_context_vs_p0a_results.json" ]]; then
  milestone "already complete: Phase-1 paired OOF result exists"
  exit 0
fi

milestone "build strict LTA auxiliary A1 index (fixed end-1s / 8s / uniform; no adaptive)"
"$PYTHON" \
  "$REPO/src/ego/step1_action_anticipation/goalstep/build_lta_aux_index.py" \
  --output-dir "$AUX_INDEX" \
  --match-policy both \
  >"$LOGS/index.log" 2>&1

"$PYTHON" - "$AUX_INDEX/build_stats.json" <<'PY'
import json
import sys
from pathlib import Path

stats = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if stats.get("output_rows") != 14926:
    raise SystemExit(f"unexpected LTA A1 rows: {stats.get('output_rows')} != 14926")
if stats.get("match_policy") != "both":
    raise SystemExit("LTA A1 must use both-match supervision")
if stats.get("observation_contract", {}).get("adaptive_transition_window") is not False:
    raise SystemExit("adaptive-transition-window must be disabled")
if stats.get("excluded_goalstep_val_rows") != 771:
    raise SystemExit("GoalStep validation leakage exclusion count drifted")
PY

milestone "extract new LTA auxiliary V-JEPA features (fixed uniform 32f)"
"$PYTHON" "$REPO/scripts/step1/ego4d_lta/extract_features.py" \
  --config "$AUX_CONFIG" \
  --split train \
  --cache-dir "$AUX_CACHE" \
  --device cuda \
  >"$LOGS/extract.log" 2>&1

if [[ ! -f "$DIRECT_RUN/final_metrics.json" ]]; then
  if [[ -e "$DIRECT_RUN/training_history.csv" || -e "$DIRECT_RUN/latest.pt" ]]; then
    milestone "ERROR: partial direct run exists and v1 has no resume contract"
    exit 1
  fi
  milestone "train joint GoalStep + LTA-aux direct probe for 10 epochs"
  "$PYTHON" \
    "$REPO/src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py" \
    --config "$DIRECT_CONFIG" \
    >"$LOGS/train.log" 2>&1
else
  milestone "reuse completed joint direct run"
fi

milestone "build new video-disjoint P0-a from LTA-aux direct epochs 1-8"
"$PYTHON" "$REPO/scripts/step1/goalstep/run_history_phase0.py" \
  --stage primary \
  --output-dir "$P0_RUN" \
  --next-config "$DIRECT_CONFIG" \
  --next-checkpoint-dir "$DIRECT_RUN/checkpoints" \
  --next-best-checkpoint "$DIRECT_RUN/checkpoints/epoch_03.pt" \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda \
  >"$LOGS/phase0.log" 2>&1

milestone "derive GoalStep-only Phase-1 store with the new frozen visual foundation"
"$PYTHON" "$REPO/scripts/step1/goalstep/prepare_history_context_store.py" \
  --source-index-dir "$REPO/src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8" \
  --cache-dir "$REPO/../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna" \
  --output-dir "$STORE" \
  --visual-config "$DIRECT_CONFIG" \
  --visual-checkpoint "$DIRECT_RUN/checkpoints/epoch_03.pt" \
  --recognition-config "$REPO/configs/step1/goalstep/z1_end_m1_lobs8_vna.yaml" \
  --recognition-checkpoint "$REPO/outputs/goalstep/runs/z1_end_m1_lobs8_vna/best.pt" \
  --split all \
  --batch-size 32 \
  --num-workers 8 \
  --shard-size 1024 \
  --device cuda \
  >"$LOGS/store.log" 2>&1

if [[ ! -f "$PHASE1_RUN/final_metrics.json" ]]; then
  if [[ -e "$PHASE1_RUN/training_history.csv" || -e "$PHASE1_RUN/latest.pt" ]]; then
    milestone "ERROR: partial Phase-1 run exists and v1 has no resume contract"
    exit 1
  fi
  milestone "train GoalStep-only visual-history Phase 1 for 10 epochs"
  "$PYTHON" \
    "$REPO/src/ego/step1_action_anticipation/goalstep/train_goalstep_history_context.py" \
    --config "$PHASE1_CONFIG" \
    >"$LOGS/phase1.log" 2>&1
else
  milestone "reuse completed LTA-aux Phase-1 run"
fi

milestone "evaluate Phase 1 against the new P0-a with paired video bootstrap"
"$PYTHON" "$REPO/scripts/step1/goalstep/evaluate_history_context_vs_p0a.py" \
  --endpoint-logits "$P0_RUN/endpoint_logits.pt" \
  --p0a-oof "$P0_RUN/p0a_primary_same_decision_oof_scores.pt" \
  --predictions-dir "$PHASE1_RUN/val_predictions" \
  --output-json "$PHASE1_RUN/history_context_vs_p0a_results.json" \
  --output-scores "$PHASE1_RUN/history_context_vs_p0a_oof_scores.pt" \
  --expected-last-epoch 10 \
  --alpha-step 0.05 \
  --bootstrap-samples 10000 \
  --seed 42 \
  >"$LOGS/evaluate.log" 2>&1

milestone "LTA-aux Phase-1 pipeline complete (Phase 2 intentionally out of scope)"
