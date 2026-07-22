#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"

run_dir="outputs/goalstep/runs/z1_end_m1_lobs8_vna"
output="docs/experiments/2026-07-21_goalstep-action-end-m1-lobs8-vna-results.md"
common=(
  --run-dir "$run_dir"
  --index-stats src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8/build_stats.json
  --config configs/step1/goalstep/z1_end_m1_lobs8_vna.yaml
  --output "$output"
)

while true; do
  if [[ -f "$run_dir/final_metrics.json" ]]; then
    python tools/write_goalstep_endpoint_report.py "${common[@]}" --status completed
    exit 0
  fi
  if grep -Eq 'training not started because extraction failed|training finished rc=[1-9]' "$run_dir/logs/pipeline.log" 2>/dev/null; then
    python tools/write_goalstep_endpoint_report.py "${common[@]}" --status failed
    exit 1
  fi
  python tools/write_goalstep_endpoint_report.py "${common[@]}" --status running >/dev/null
  sleep 30
done
