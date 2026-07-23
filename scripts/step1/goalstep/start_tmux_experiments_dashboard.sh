#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
cloudflared_bin="${CLOUDFLARED_BIN:-/tmp/cloudflared-goalstep}"
session="ego_goalstep_overview"
log_dir="$repo_dir/outputs/goalstep/dashboard_overview"
port=17867
mkdir -p "$log_dir"

if tmux has-session -t "$session" 2>/dev/null; then
  tmux kill-session -t "$session"
fi

tmux new-session -d -s "$session" -n dashboard -c "$repo_dir" \
  "'$python_bin' tools/goalstep_experiments_dashboard.py --host 0.0.0.0 --port '$port' >> '$log_dir/dashboard.log' 2>&1; rc=\$?; echo dashboard_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n tunnel -c "$repo_dir" \
  "'$cloudflared_bin' tunnel --url 'http://127.0.0.1:$port' --no-autoupdate 2>&1 | tee '$log_dir/cloudflared.log'; exec bash"

echo "$session"
tmux list-windows -t "$session"
