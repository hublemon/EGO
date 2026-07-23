#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_dir="outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10"
log_dir="$run_dir/logs"
config="configs/step1/goalstep/z1_end_m1_lobs8_next_action_vna_ep10.yaml"
mkdir -p "$log_dir"

echo "[$(date -Is)] action_end-1s / 8s observation -> next strict-future action training started" > "$log_dir/pipeline.log"
echo "[$(date -Is)] reusing frozen features from goalstep_feature_cache_end_m1_lobs8_vna; applying labels from index" >> "$log_dir/pipeline.log"
CUDA_VISIBLE_DEVICES=0 "$python_bin" src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py \
  --config "$config" > "$log_dir/train.log" 2>&1
train_rc=$?
echo "[$(date -Is)] training finished rc=$train_rc" >> "$log_dir/pipeline.log"
exit "$train_rc"
