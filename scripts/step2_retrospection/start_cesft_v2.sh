#!/usr/bin/env bash
# cesft_v2 조합 체인 기동 — setsid 로 세션에서 분리 (SSH/VS Code 끊겨도 지속).
# EGO_jihun/CLAUDE.md 규약: setsid + disown, PPID=1 재부모화 확인, 단일 GPU.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUNS="runs/cesft_v2"
mkdir -p "$RUNS/logs" "$RUNS/markers"

# 이미 살아있으면 중복 기동 거부
if [[ -f "$RUNS/chain.pid" ]]; then
  OLD=$(cat "$RUNS/chain.pid")
  if kill -0 "$OLD" 2>/dev/null; then
    echo "[start] 이미 실행 중 (pid=$OLD) — 중복 기동 거부"; exit 0
  fi
fi
rm -f "$RUNS/markers/CHAIN_FAILED" "$RUNS/markers/CHAIN_STUCK"

RETRO3_RUNS="$RUNS" \
CHAIN_SCRIPT="scripts/step2_retrospection/cesft_v2_parallel.sh" \
MAX_PARALLEL="${MAX_PARALLEL:-1}" MIN_FREE_MB="${MIN_FREE_MB:-60000}" \
RAM_FLOOR_GB="${RAM_FLOOR_GB:-100}" \
MAX_RETRY="${MAX_RETRY:-5}" \
  setsid nohup bash scripts/step2_retrospection/supervisor.sh \
  >> "$RUNS/logs/chain_stdout.log" 2>&1 < /dev/null &
echo $! > "$RUNS/chain.pid"
disown -a
sleep 2
echo "[start] cesft_v2 supervisor 기동 pid=$(cat "$RUNS/chain.pid")"
ps -eo pid,ppid,stat,cmd | grep -E "supervisor.sh|cesft_v2_chain" | grep -v grep || true
echo "[start] PPID=1(init) 재부모화 확인 필요 (위 STAT/PPID)"
