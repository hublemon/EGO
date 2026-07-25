#!/usr/bin/env bash
# retro4 체인 기동 (ssh 분리) — Phase-1 K8 prior 재구축. supervisor 재사용(CHAIN_SCRIPT 주입).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUNS=runs/retro4
mkdir -p "$RUNS/logs"

if [[ -f "$RUNS/chain.pid" ]] && kill -0 "$(cat "$RUNS/chain.pid")" 2>/dev/null; then
  echo "이미 실행 중: PID $(cat "$RUNS/chain.pid")"
  exit 0
fi
rm -f "$RUNS/markers/CHAIN_FAILED" 2>/dev/null || true

RETRO3_RUNS=runs/retro4 CHAIN_SCRIPT=scripts/step2_retrospection/retro4_chain.sh \
  nohup setsid bash scripts/step2_retrospection/supervisor.sh \
  >> "$RUNS/logs/chain_stdout.log" 2>&1 < /dev/null &
echo $! > "$RUNS/chain.pid"
echo "retro4 체인 기동: PID $(cat "$RUNS/chain.pid") — 로그 $RUNS/logs/chain.log"
echo "대시보드: python3 tools/retro3_dashboard.py --runs runs/retro4 --port 7868"
