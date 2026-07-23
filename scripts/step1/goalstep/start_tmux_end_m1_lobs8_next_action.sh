#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
cloudflared_bin="${CLOUDFLARED_BIN:-/tmp/cloudflared-goalstep}"
session="ego_goalstep_end_m1_lobs8_next_action"
run_dir="$repo_dir/outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10"
cache_dir="$repo_dir/../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna"
log_dir="$run_dir/logs"
port=17866
mkdir -p "$log_dir"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session" >&2
  exit 1
fi

tmux new-session -d -s "$session" -n pipeline -c "$repo_dir" \
  "bash scripts/step1/goalstep/run_end_m1_lobs8_next_action_vna_ep10.sh; rc=\$?; echo pipeline_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n dashboard -c "$repo_dir" \
  "'$python_bin' tools/goalstep_live_dashboard.py --host 0.0.0.0 --port '$port' --run-dir '$run_dir' --cache-dir '$cache_dir' --train-total 30374 --val-total 7214 --epochs 10 --title 'GoalStep · A2.end−1s / 8s → next strict-future A3' >> '$log_dir/dashboard.log' 2>&1; rc=\$?; echo dashboard_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n tunnel -c "$repo_dir" \
  "'$cloudflared_bin' tunnel --url 'http://127.0.0.1:$port' --no-autoupdate 2>&1 | tee '$log_dir/cloudflared.log'; exec bash"

echo "$session"
tmux list-windows -t "$session"
