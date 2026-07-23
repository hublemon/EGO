#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/root/nvme/migration/jihun/EGO_jihun2
PYTHON=/root/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
STORE="$REPO/../datasets/Ego4D/goalstep_history_context_store/manifest.json"
RUN_DIR="$REPO/outputs/goalstep/runs/z1_history_context_k8_vna_ep10"
LOG="$RUN_DIR/logs/phase1_queue.log"

mkdir -p "$RUN_DIR/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log_milestone() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1" | tee -a "$LOG"
}

store_complete() {
  "$PYTHON" - "$STORE" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
splits = manifest.get("splits", {})
valid = (
    splits.get("train", {}).get("rows") == 30374
    and splits.get("val", {}).get("rows") == 7214
)
raise SystemExit(0 if valid else 1)
PY
}

log_milestone "waiting for provenance-complete train/val derived store"
while ! store_complete; do
  if ! pgrep -f '[p]repare_history_context_store.py --split all' >/dev/null; then
    log_milestone "ERROR: store process ended before a complete manifest was published"
    exit 1
  fi
  sleep 10
done

log_milestone "derived store complete; starting revised Phase-1 (P0-b diagnostic only)"
"$PYTHON" src/ego/step1_action_anticipation/goalstep/train_goalstep_history_context.py \
  --config configs/step1/goalstep/z1_history_context_k8_vna_ep10.yaml \
  2>&1 | tee "$RUN_DIR/logs/train.log"
log_milestone "Phase-1 training complete; cross-fitted champion evaluation is ready"
"$PYTHON" scripts/step1/goalstep/evaluate_history_context_vs_p0a.py \
  --expected-last-epoch 10 \
  --alpha-step 0.05 \
  --bootstrap-samples 10000 \
  --seed 42 \
  2>&1 | tee "$RUN_DIR/logs/champion_eval.log"
log_milestone "Phase-1 P0-a-aware cross-fitted evaluation complete"
