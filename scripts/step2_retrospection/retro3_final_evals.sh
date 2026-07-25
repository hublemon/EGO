#!/usr/bin/env bash
# retro3 마무리 평가 — DPO D1은 G3 abort(문체-학습 가드)로 종료됐으므로
# base / r1_sft 두 arm의 배터리(acc|cov)·개입③만 돌리고,
# RETRO3_CHAIN_DONE(partial)을 써서 retro4가 GPU를 이어받게 한다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH="$REPO/src" HF_HOME=/mnt/nvme/cache TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO3_RUNS=runs/retro3
LOG=runs/retro3/logs/chain.log
ADAPT=outputs/step2_retrospection

run() { local mk="$1"; shift; [[ -f "runs/retro3/markers/$mk" ]] && return 0
  "$@" >> "$LOG" 2>&1 || true; }

run S7_EVAL_BASE_DONE          "$PY" -m ego.step2_retrospection.eval.battery --arm base --eval_n 1000
run S7_EVAL_R1_SFT_DONE        "$PY" -m ego.step2_retrospection.eval.battery --arm r1_sft --adapter "$ADAPT/r1_sft/adapter" --eval_n 1000
run S7_INTERVENTION_BASE_DONE  "$PY" -m ego.step2_retrospection.eval.intervention --arm base --n 300
run S7_INTERVENTION_R1_SFT_DONE "$PY" -m ego.step2_retrospection.eval.intervention --arm r1_sft --adapter "$ADAPT/r1_sft/adapter" --n 300

echo "{\"ts\": $(date +%s), \"partial\": \"dpo_d1 G3_STYLE_ABORT — base/r1_sft만 평가\"}" \
  > runs/retro3/markers/RETRO3_CHAIN_DONE
echo "[retro3-final] 완료 — retro4로 GPU 인계"
