#!/usr/bin/env bash
set -euo pipefail

session="ego_goalstep_jihun2"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
log_dir="$repo_dir/outputs/goalstep/runs/z1_jihun2/logs"
cloudflared_bin="/tmp/cloudflared-goalstep"

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
  "bash scripts/step1/goalstep/run_full_jihun2.sh; rc=\$?; echo pipeline_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n dashboard -c "$repo_dir" \
  "python tools/goalstep_live_dashboard.py --host 0.0.0.0 --port 7860 >> '$log_dir/dashboard.log' 2>&1; rc=\$?; echo dashboard_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n tunnel -c "$repo_dir" \
  "'$cloudflared_bin' tunnel --url http://127.0.0.1:7860 --no-autoupdate 2>&1 | tee '$log_dir/cloudflared.log'; exec bash"

echo "started detached tmux session: $session"
tmux list-windows -t "$session"
