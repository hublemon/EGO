#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"

run_dir="outputs/goalstep/runs/z1_end_m1_lobs8_vna"
log_dir="$run_dir/logs"
config="configs/step1/goalstep/z1_end_m1_lobs8_vna.yaml"
mkdir -p "$log_dir"

echo "[$(date -Is)] action_end-1s / 8s / VNA pipeline started" > "$log_dir/pipeline.log"
echo "[$(date -Is)] full feature extraction started: train=GPU0 val=GPU1" >> "$log_dir/pipeline.log"

CUDA_VISIBLE_DEVICES=0 python scripts/step1/ego4d_lta/extract_features.py \
  --config "$config" --split train > "$log_dir/extract_train.log" 2>&1 &
train_extract_pid=$!
CUDA_VISIBLE_DEVICES=1 python scripts/step1/ego4d_lta/extract_features.py \
  --config "$config" --split val > "$log_dir/extract_val.log" 2>&1 &
val_extract_pid=$!
echo "$train_extract_pid" > "$log_dir/extract_train.pid"
echo "$val_extract_pid" > "$log_dir/extract_val.pid"

wait "$train_extract_pid"
train_extract_rc=$?
wait "$val_extract_pid"
val_extract_rc=$?
echo "[$(date -Is)] extraction finished train_rc=$train_extract_rc val_rc=$val_extract_rc" >> "$log_dir/pipeline.log"

if [[ "$train_extract_rc" -ne 0 || "$val_extract_rc" -ne 0 ]]; then
  echo "[$(date -Is)] training not started because extraction failed" >> "$log_dir/pipeline.log"
  exit 1
fi

echo "[$(date -Is)] full 15-epoch V/N/A training started on GPU0" >> "$log_dir/pipeline.log"
CUDA_VISIBLE_DEVICES=0 python src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py \
  --config "$config" > "$log_dir/train.log" 2>&1
train_rc=$?
echo "[$(date -Is)] training finished rc=$train_rc" >> "$log_dir/pipeline.log"
exit "$train_rc"
