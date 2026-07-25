#!/usr/bin/env bash
# 자가 복구 supervisor — 체인이 죽으면 자동 재시작 (marker resume이라 안전).
# 정지 조건: RETRO3_CHAIN_DONE | GPU 접근 불가 | 같은 지점 연속 MAX_RETRY회 실패(CHAIN_STUCK).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUNS="${RETRO3_RUNS:-runs/retro3}"
MK="$RUNS/markers"
LOG="$RUNS/logs"
mkdir -p "$MK" "$LOG"
MAX_RETRY="${MAX_RETRY:-5}"
# retro4 재사용: CHAIN_SCRIPT 환경변수로 체인 스크립트 교체 (기본 retro3)
CHAIN_SCRIPT="${CHAIN_SCRIPT:-scripts/step2_retrospection/retro3_chain.sh}"

# 메모리 워치독 — cgroup memory.current/peak + top-3 RSS 프로세스를 주기 기록.
# 2026-07-23: S3 77% OOM-kill(컨테이너 240GiB 한도) 원인 특정용. 관측만, 개입 없음.
#
# cur 는 page cache(file) 를 포함하므로 압박 지표로 쓰면 안 된다 — frame_cache JPEG
# 쓰기만으로 cur 가 한도에 붙는다(2026-07-24 20:23: anon 4.2G / file 227.7G / cur 232.9G).
# 실제로 회수 불가한 분만 더한 hard 를 함께 남기고, 경보(ram_alarm.sh)는 hard 를 본다.
cg_hard_bytes() {
  awk '$1=="anon"||$1=="unevictable"||$1=="slab_unreclaimable"||$1=="file_dirty"||$1=="file_writeback"{s+=$2}
       END{print s+0}' /sys/fs/cgroup/memory.stat 2>/dev/null \
  || awk '$1=="total_rss"||$1=="total_unevictable"||$1=="total_dirty"||$1=="total_writeback"{s+=$2}
          END{print s+0}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null \
  || echo 0
}
MEM_LOG_EVERY="${MEM_LOG_EVERY:-60}"
mem_watchdog() {
  while true; do
    awk -v ts="$(date '+%F %T')" \
        -v cur="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)" \
        -v hard="$(cg_hard_bytes)" \
        -v peak="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)" \
        -v lim="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)" \
        'BEGIN{g=1073741824; printf "%s cur=%.1fG hard=%.1fG peak=%.1fG limit=%.0fG |", ts, cur/g, hard/g, peak/g, lim/g}' >> "$LOG/mem.log"
    ps -eo rss,comm --sort=-rss --no-headers 2>/dev/null | head -3 | \
      awk '{printf " %s=%.1fG", $2, $1/1048576} END{print ""}' >> "$LOG/mem.log"
    sleep "$MEM_LOG_EVERY"
  done
}
mem_watchdog &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null' EXIT

fail_sig() { cat "$MK/CHAIN_FAILED" 2>/dev/null | tr -d ' \n' || echo ""; }

retry=0
last_sig=""
while true; do
  if [[ -f "$MK/RETRO3_CHAIN_DONE" ]]; then
    echo "[supervisor] 체인 완료 — 종료" | tee -a "$LOG/supervisor.log"
    break
  fi
  if ! nvidia-smi > /dev/null 2>&1; then
    echo "{\"reason\": \"gpu_down\", \"ts\": $(date +%s)}" > "$MK/CHAIN_STUCK"
    echo "[supervisor] GPU 접근 불가 — 정지 (사람/에이전트 개입 필요)" | tee -a "$LOG/supervisor.log"
    break
  fi
  echo "[supervisor] 체인 시작 (try=$((retry + 1)))" | tee -a "$LOG/supervisor.log"
  rm -f "$MK/CHAIN_FAILED"
  bash "$CHAIN_SCRIPT" >> "$LOG/supervisor.log" 2>&1
  rc=$?
  [[ -f "$MK/RETRO3_CHAIN_DONE" ]] && continue
  sig="rc=${rc}:$(fail_sig)"
  if [[ "$sig" == "$last_sig" ]]; then
    retry=$((retry + 1))
  else
    retry=1
    last_sig="$sig"
  fi
  echo "[supervisor] 체인 중단 ($sig, 연속 $retry/$MAX_RETRY)" | tee -a "$LOG/supervisor.log"
  if [[ $retry -ge $MAX_RETRY ]]; then
    echo "{\"reason\": \"repeated_failure\", \"sig\": \"$sig\", \"ts\": $(date +%s)}" > "$MK/CHAIN_STUCK"
    echo "[supervisor] 같은 지점 반복 실패 — CHAIN_STUCK (코드 수정 필요)" | tee -a "$LOG/supervisor.log"
    break
  fi
  sleep 30
done
