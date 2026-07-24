#!/usr/bin/env bash
# ablation 체인 기동 (ssh 분리). 메인 체인 완료를 스스로 기다린다.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUNS="${RETRO3_RUNS:-runs/retro3}"
mkdir -p "$RUNS/logs"

if [[ -f "$RUNS/ablation.pid" ]] && kill -0 "$(cat "$RUNS/ablation.pid")" 2>/dev/null; then
  echo "이미 실행 중: PID $(cat "$RUNS/ablation.pid")"
  exit 0
fi
rm -f "$RUNS/markers/ABLATION_FAILED" 2>/dev/null || true

nohup setsid bash scripts/step2_retrospection/retro3_ablation_chain.sh \
  >> "$RUNS/logs/ablation_stdout.log" 2>&1 < /dev/null &
echo $! > "$RUNS/ablation.pid"
echo "ablation 체인 기동: PID $(cat "$RUNS/ablation.pid") — 메인 체인 완료 후 자동 시작"
