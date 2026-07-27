#!/usr/bin/env bash
# cesft_v2_fp 체인 기동 — setsid 분리 + supervisor + ram_alarm + 스톨 워치독.
# 규약: 단독 스크립트 직접 기동 금지(07-25 run_gdelta.sh 사고) — 반드시 이 스크립트로만.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUNS="runs/cesft_v2_fp"
mkdir -p "$RUNS/logs" "$RUNS/markers" "$RUNS/status"

if [[ -f "$RUNS/chain.pid" ]]; then
  OLD=$(cat "$RUNS/chain.pid")
  if kill -0 "$OLD" 2>/dev/null; then
    echo "[start] 이미 실행 중 (pid=$OLD) — 중복 기동 거부"; exit 0
  fi
fi
rm -f "$RUNS/markers/CHAIN_FAILED" "$RUNS/markers/CHAIN_STUCK"

RETRO3_RUNS="$RUNS" \
CHAIN_SCRIPT="scripts/step2_retrospection/cesft_fp_chain.sh" \
MAX_PARALLEL=1 MIN_FREE_MB="${MIN_FREE_MB:-60000}" \
RAM_FLOOR_GB="${RAM_FLOOR_GB:-100}" \
MAX_RETRY="${MAX_RETRY:-5}" \
  setsid nohup bash scripts/step2_retrospection/supervisor.sh \
  >> "$RUNS/logs/chain_stdout.log" 2>&1 < /dev/null &
echo $! > "$RUNS/chain.pid"
disown -a

# 2026-07-26: ram_alarm 은 RETRO3_RUNS_ABS(절대경로)를 읽음 — 미전달로 구 cesft_v2
# 로그에 기록되던 버그 수정 (07-25 서버 B OOM 때 fp ram_alarm.log 가 비어 있던 원인).
RETRO3_RUNS_ABS="$REPO/$RUNS" setsid nohup bash tools/oom_opt/ram_alarm.sh \
  >> "$RUNS/logs/ram_alarm.log" 2>&1 < /dev/null & disown
RETRO3_RUNS="$RUNS" setsid nohup bash scripts/step2_retrospection/stall_watchdog_fp.sh \
  >> "$RUNS/logs/stall_stdout.log" 2>&1 < /dev/null & disown

sleep 2
echo "[start] cesft_v2_fp supervisor 기동 pid=$(cat "$RUNS/chain.pid")"
ps -eo pid,ppid,stat,cmd | grep -E "supervisor.sh|cesft_fp_chain|ram_alarm|stall_watchdog_fp" | grep -v grep || true
