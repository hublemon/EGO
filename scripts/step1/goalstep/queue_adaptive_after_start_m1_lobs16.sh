#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"

source_run="outputs/goalstep/runs/z1_start_m1_lobs16_vna"
source_log="$source_run/logs/pipeline.log"
target_run="outputs/goalstep/runs/z1_adaptive_transition_mr24x8_vna_ep10"
target_final="$target_run/final_metrics.json"
queue_log="$target_run/logs/queue.log"
poll_seconds="${QUEUE_POLL_SECONDS:-30}"
mkdir -p "$(dirname "$queue_log")"

stamp() {
  echo "[$(date -Is)] $*" | tee -a "$queue_log"
}

source_succeeded() {
  [[ -f "$source_run/final_metrics.json" ]] \
    && grep -Eq 'training finished rc=0$' "$source_log" 2>/dev/null
}

source_failed() {
  grep -Eq 'feature extraction finished: (val|train) rc=[1-9][0-9]*$|training finished rc=[1-9][0-9]*$' \
    "$source_log" 2>/dev/null
}

source_is_running() {
  pgrep -f '[r]un_start_m1_lobs16_vna[.]sh' >/dev/null
}

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "QUEUE_POLL_SECONDS must be a positive integer: $poll_seconds" >&2
  exit 2
fi

if [[ -f "$target_final" ]]; then
  stamp "adaptive experiment is already complete; nothing to launch"
  exit 0
fi

if pgrep -f '[r]un_adaptive_transition_mr24x8_vna_ep10[.]sh' >/dev/null; then
  stamp "ERROR: adaptive pipeline is already running"
  exit 3
fi

stamp "queued: adaptive transition MR24+8 will start after action_start-1s/16s succeeds"
while ! source_succeeded; do
  if source_failed; then
    stamp "ERROR: action_start-1s/16s reported a non-zero stage exit; adaptive launch cancelled"
    exit 1
  fi
  if ! source_is_running; then
    stamp "ERROR: action_start-1s/16s stopped without a successful final_metrics marker; adaptive launch cancelled"
    exit 1
  fi
  sleep "$poll_seconds"
done

stamp "action_start-1s/16s completed successfully; starting adaptive transition MR24+8 now"
bash scripts/step1/goalstep/run_adaptive_transition_mr24x8_vna_ep10.sh
adaptive_rc=$?
stamp "adaptive transition MR24+8 pipeline finished rc=$adaptive_rc"
exit "$adaptive_rc"
