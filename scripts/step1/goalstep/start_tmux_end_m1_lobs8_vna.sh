#!/usr/bin/env bash
set -euo pipefail

session="ego_goalstep_end_m1_lobs8_vna"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
run_dir="$repo_dir/outputs/goalstep/runs/z1_end_m1_lobs8_vna"
log_dir="$run_dir/logs"
cache_dir="$repo_dir/../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna"
cloudflared_bin="/tmp/cloudflared-goalstep"
port=17861

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session"
  tmux list-windows -t "$session"
  exit 0
fi

if [[ ! -x "$cloudflared_bin" ]]; then
  echo "Missing executable: $cloudflared_bin" >&2
  exit 1
fi

mkdir -p "$log_dir"

tmux new-session -d -s "$session" -n pipeline -c "$repo_dir" \
  "bash scripts/step1/goalstep/run_end_m1_lobs8_vna.sh; rc=\$?; echo pipeline_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n dashboard -c "$repo_dir" \
  "python tools/goalstep_live_dashboard.py --host 0.0.0.0 --port $port --run-dir '$run_dir' --cache-dir '$cache_dir' --train-total 30374 --val-total 7214 --epochs 15 --title 'GoalStep · action_end−1s · 8s · V/N/A' >> '$log_dir/dashboard.log' 2>&1; rc=\$?; echo dashboard_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n tunnel -c "$repo_dir" \
  "'$cloudflared_bin' tunnel --url http://127.0.0.1:$port --no-autoupdate 2>&1 | tee '$log_dir/cloudflared.log'; exec bash"
tmux new-window -d -t "$session" -n reporter -c "$repo_dir" \
  "bash scripts/step1/goalstep/watch_end_m1_lobs8_report.sh >> '$log_dir/reporter.log' 2>&1; rc=\$?; echo reporter_exit=\$rc; exec bash"

echo "started detached tmux session: $session"
tmux list-windows -t "$session"
