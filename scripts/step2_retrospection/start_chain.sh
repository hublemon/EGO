#!/usr/bin/env bash
# 체인을 ssh 세션과 분리해 기동한다 (nohup + setsid). 재실행 안전 (marker resume).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUNS="${RETRO3_RUNS:-runs/retro3}"
mkdir -p "$RUNS/logs"

if [[ -f "$RUNS/chain.pid" ]] && kill -0 "$(cat "$RUNS/chain.pid")" 2>/dev/null; then
  echo "이미 실행 중: PID $(cat "$RUNS/chain.pid")"
  exit 0
fi
rm -f "$RUNS/markers/CHAIN_FAILED" 2>/dev/null || true

nohup setsid bash scripts/step2_retrospection/supervisor.sh \
  >> "$RUNS/logs/chain_stdout.log" 2>&1 < /dev/null &
echo $! > "$RUNS/chain.pid"
echo "체인 기동: PID $(cat "$RUNS/chain.pid") — 로그 $RUNS/logs/chain.log"
echo "대시보드: python3 tools/retro3_dashboard.py  (port 7867)"
