#!/usr/bin/env bash
# supervisor 가 실행하는 병렬 오케스트레이터 래퍼 (직렬 cesft_v2_chain.sh 대체).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export PYTHONPATH="$REPO/src"
export HF_HOME=/mnt/nvme/cache
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RETRO3_RUNS="${RETRO3_RUNS:-runs/cesft_v2}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export MIN_FREE_MB="${MIN_FREE_MB:-60000}"
export RAM_FLOOR_GB="${RAM_FLOOR_GB:-100}"
exec "$PY" tools/parallel_orchestrator.py
