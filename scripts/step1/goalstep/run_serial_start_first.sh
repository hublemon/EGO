#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
log="outputs/goalstep/serial_start_first.log"
mkdir -p "$(dirname "$log")"

stamp() {
  echo "[$(date -Is)] $*" | tee -a "$log"
}

stamp "serial order: start-1s/8s -> start-1s/16s -> end-1s/8s -> end-6s/8s"
stamp "waiting for the already-running start-1s/8s pipeline"
while [[ ! -f outputs/goalstep/runs/z1_start_m1_lobs8_vna/final_metrics.json ]]; do
  if ! pgrep -f '^bash scripts/step1/goalstep/run_start_m1_lobs8_vna[.]sh$' >/dev/null; then
    stamp "ERROR: start-1s/8s stopped before final_metrics.json was written"
    exit 1
  fi
  sleep 30
done

stamp "start-1s/8s completed; resuming start-1s/16s"
bash scripts/step1/goalstep/run_start_m1_lobs16_vna.sh
rc=$?
stamp "start-1s/16s finished rc=$rc"
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi

stamp "starting end-1s/8s matched ep10 last"
bash scripts/step1/goalstep/run_end_m1_lobs8_vna_ep10.sh
rc=$?
stamp "end-1s/8s matched ep10 finished rc=$rc"
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi

stamp "starting end-6s/8s ep10 last"
bash scripts/step1/goalstep/run_end_m6_lobs8_vna_ep10.sh
rc=$?
stamp "end-6s/8s ep10 finished rc=$rc"
exit "$rc"
