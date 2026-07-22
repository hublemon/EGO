#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found or not executable: $python_bin" >&2
  exit 2
fi

run_dir="outputs/goalstep/runs/z1_end_m6_lobs8_vna_ep10"
log_dir="$run_dir/logs"
config="configs/step1/goalstep/z1_end_m6_lobs8_vna_ep10.yaml"
mkdir -p "$log_dir"

echo "[$(date -Is)] action_end-6s / 8s / VNA ep10 started" > "$log_dir/pipeline.log"
for split in val train; do
  echo "[$(date -Is)] feature extraction started: $split" >> "$log_dir/pipeline.log"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" scripts/step1/ego4d_lta/extract_features.py \
    --config "$config" --split "$split" > "$log_dir/extract_${split}.log" 2>&1
  extract_rc=$?
  echo "[$(date -Is)] feature extraction finished: $split rc=$extract_rc" >> "$log_dir/pipeline.log"
  if [[ "$extract_rc" -ne 0 ]]; then
    exit "$extract_rc"
  fi
done

echo "[$(date -Is)] full 10-epoch V/N/A training started" >> "$log_dir/pipeline.log"
CUDA_VISIBLE_DEVICES=0 "$python_bin" src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py \
  --config "$config" > "$log_dir/train.log" 2>&1
train_rc=$?
echo "[$(date -Is)] training finished rc=$train_rc" >> "$log_dir/pipeline.log"
exit "$train_rc"
