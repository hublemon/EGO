#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
cloudflared_bin="${CLOUDFLARED_BIN:-/tmp/cloudflared-goalstep}"

if [[ ! -x "$python_bin" || ! -x "$cloudflared_bin" ]]; then
  echo "Missing python or cloudflared executable" >&2
  exit 2
fi

ensure_window() {
  local session="$1" name="$2" command="$3"
  if ! tmux list-windows -t "$session" -F '#{window_name}' | grep -Fxq "$name"; then
    tmux new-window -d -t "$session" -n "$name" -c "$repo_dir" "$command"
  fi
}

start_run() {
  local suffix="$1" seconds="$2" port="$3"
  local session="ego_goalstep_start_m1_lobs${suffix}_vna"
  local run_dir="$repo_dir/outputs/goalstep/runs/z1_start_m1_lobs${suffix}_vna"
  local cache_dir="$repo_dir/../datasets/Ego4D/goalstep_feature_cache_start_m1_lobs${suffix}_vna"
  local log_dir="$run_dir/logs"
  mkdir -p "$log_dir"

  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" -n pipeline -c "$repo_dir" \
      "bash scripts/step1/goalstep/run_start_m1_lobs${suffix}_vna.sh; rc=\$?; echo pipeline_exit=\$rc; exec bash"
  fi
  ensure_window "$session" dashboard \
    "'$python_bin' tools/goalstep_live_dashboard.py --host 0.0.0.0 --port '$port' --run-dir '$run_dir' --cache-dir '$cache_dir' --train-total 30374 --val-total 7214 --epochs 10 --title 'GoalStep · action_start−1s · ${seconds}s · V/N/A' >> '$log_dir/dashboard.log' 2>&1; rc=\$?; echo dashboard_exit=\$rc; exec bash"
  ensure_window "$session" tunnel \
    "'$cloudflared_bin' tunnel --url 'http://127.0.0.1:$port' --no-autoupdate 2>&1 | tee '$log_dir/cloudflared.log'; exec bash"
  echo "$session"
  tmux list-windows -t "$session"
}

start_run 8 8 17862
start_run 16 16 17863

baseline_session="ego_goalstep_end_m1_lobs8_vna_ep10"
baseline_run="$repo_dir/outputs/goalstep/runs/z1_end_m1_lobs8_vna_ep10"
baseline_cache="$repo_dir/../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna"
mkdir -p "$baseline_run/logs"
if ! tmux has-session -t "$baseline_session" 2>/dev/null; then
  tmux new-session -d -s "$baseline_session" -n pipeline -c "$repo_dir" \
    "bash scripts/step1/goalstep/run_end_m1_lobs8_vna_ep10.sh; rc=\$?; echo pipeline_exit=\$rc; exec bash"
fi
ensure_window "$baseline_session" dashboard \
  "'$python_bin' tools/goalstep_live_dashboard.py --host 0.0.0.0 --port 17864 --run-dir '$baseline_run' --cache-dir '$baseline_cache' --train-total 30374 --val-total 7214 --epochs 10 --title 'GoalStep · action_end−1s · 8s · matched ep10' >> '$baseline_run/logs/dashboard.log' 2>&1; rc=\$?; echo dashboard_exit=\$rc; exec bash"
ensure_window "$baseline_session" tunnel \
  "'$cloudflared_bin' tunnel --url 'http://127.0.0.1:17864' --no-autoupdate 2>&1 | tee '$baseline_run/logs/cloudflared.log'; exec bash"
echo "$baseline_session"
tmux list-windows -t "$baseline_session"

end_m6_session="ego_goalstep_end_m6_lobs8_vna_ep10"
end_m6_run="$repo_dir/outputs/goalstep/runs/z1_end_m6_lobs8_vna_ep10"
end_m6_cache="$repo_dir/../datasets/Ego4D/goalstep_feature_cache_end_m6_lobs8_vna"
mkdir -p "$end_m6_run/logs"
if ! tmux has-session -t "$end_m6_session" 2>/dev/null; then
  tmux new-session -d -s "$end_m6_session" -n waiting -c "$repo_dir" "exec bash"
fi
ensure_window "$end_m6_session" dashboard \
  "'$python_bin' tools/goalstep_live_dashboard.py --host 0.0.0.0 --port 17865 --run-dir '$end_m6_run' --cache-dir '$end_m6_cache' --train-total 30374 --val-total 7214 --epochs 10 --title 'GoalStep · action_end−6s · 8s · ep10' >> '$end_m6_run/logs/dashboard.log' 2>&1; rc=\$?; echo dashboard_exit=\$rc; exec bash"
ensure_window "$end_m6_session" tunnel \
  "'$cloudflared_bin' tunnel --url 'http://127.0.0.1:17865' --no-autoupdate 2>&1 | tee '$end_m6_run/logs/cloudflared.log'; exec bash"
echo "$end_m6_session"
tmux list-windows -t "$end_m6_session"
