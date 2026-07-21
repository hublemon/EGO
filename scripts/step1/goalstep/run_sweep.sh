#!/usr/bin/env bash
# 정규화 스윕: depth / weight decay / sampler 를 한 번에 하나씩만 바꾼다.
# 공통 고정 — action-only, seed 42, val_subset 2000/seed 42, bf16, lr 3e-4, 15 epoch, 동일 feature cache.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
gpu="${1:?usage: run_sweep.sh <gpu_id> <name...>}"; shift
for name in "$@"; do
  out="outputs/goalstep/sweep/$name"; mkdir -p "$out/logs"
  echo "[$(date -Is)] START $name (gpu $gpu)" >> outputs/goalstep/sweep/sweep.log
  CUDA_VISIBLE_DEVICES="$gpu" python src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py \
    --config "configs/step1/goalstep/sweep/$name.yaml" > "$out/logs/train.log" 2>&1
  echo "[$(date -Is)] DONE $name rc=$?" >> outputs/goalstep/sweep/sweep.log
done
