#!/usr/bin/env bash
# S3 굳히기 체인 기동 (ssh 분리). 재실행 안전 (marker resume).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
RUNS=runs/s3harden; mkdir -p "$RUNS/logs"
if [[ -f "$RUNS/chain.pid" ]] && kill -0 "$(cat "$RUNS/chain.pid")" 2>/dev/null; then
  echo "이미 실행 중: PID $(cat "$RUNS/chain.pid")"; exit 0
fi
rm -f "$RUNS/markers/CHAIN_FAILED" "$RUNS/markers/CHAIN_STUCK" 2>/dev/null || true
nohup setsid bash scripts/step2_retrospection/s3_harden_chain.sh \
  >> "$RUNS/logs/stdout.log" 2>&1 < /dev/null &
echo $! > "$RUNS/chain.pid"
echo "s3harden 기동: PID $(cat "$RUNS/chain.pid") — 로그 $RUNS/logs/chain.log"
