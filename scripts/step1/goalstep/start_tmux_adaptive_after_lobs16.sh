#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
cloudflared_bin="${CLOUDFLARED_BIN:-/tmp/cloudflared-goalstep}"
session="ego_goalstep_adaptive_transition"
run_dir="$repo_dir/outputs/goalstep/runs/z1_adaptive_transition_mr24x8_vna_ep10"
cache_dir="$repo_dir/../datasets/Ego4D/goalstep_feature_cache_adaptive_transition_mr24x8_vna"
log_dir="$run_dir/logs"
port="${ADAPTIVE_DASHBOARD_PORT:-17868}"
mkdir -p "$log_dir"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found or not executable: $python_bin" >&2
  exit 2
fi
if [[ ! -x "$cloudflared_bin" ]]; then
  echo "cloudflared not found or not executable: $cloudflared_bin" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "ADAPTIVE_DASHBOARD_PORT must be a valid TCP port: $port" >&2
  exit 2
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists; refusing to create a duplicate queue: $session" >&2
  exit 1
fi
if ! "$python_bin" - "$port" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", int(sys.argv[1])))
except OSError as exc:
    print(f"Dashboard port is unavailable: {sys.argv[1]} ({exc})", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
PY
then
  exit 1
fi

tmux new-session -d -s "$session" -n queue -c "$repo_dir" \
  "bash scripts/step1/goalstep/queue_adaptive_after_start_m1_lobs16.sh; rc=\$?; echo queue_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n dashboard -c "$repo_dir" \
  "'$python_bin' tools/goalstep_live_dashboard.py --host 0.0.0.0 --port '$port' --run-dir '$run_dir' --cache-dir '$cache_dir' --train-total 18962 --val-total 4458 --epochs 10 --title 'GoalStep · adaptive A1 boundary · MR24+8 · next A2' >> '$log_dir/dashboard.log' 2>&1; rc=\$?; echo dashboard_exit=\$rc; exec bash"
tmux new-window -d -t "$session" -n tunnel -c "$repo_dir" \
  "'$cloudflared_bin' tunnel --url 'http://127.0.0.1:$port' --no-autoupdate 2>&1 | tee '$log_dir/cloudflared.log'; exec bash"

echo "$session"
tmux list-windows -t "$session"
