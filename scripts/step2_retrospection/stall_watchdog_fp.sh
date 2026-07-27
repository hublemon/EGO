#!/usr/bin/env bash
# 스톨 워치독 (cesft_v2_fp) — status 무갱신 스테이지의 프로세스만 kill → supervisor 가 marker resume.
# 2026-07-24 sft_r0 사고(무갱신 7h 방치) 재발 방지. supervisor 는 죽은 건 살리지만 멈춘 건 못 본다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUNS="${RETRO3_RUNS:-runs/cesft_v2_fp}"
ST="$RUNS/status"; MK="$RUNS/markers"; LOG="$RUNS/logs"
STALL_SEC="${STALL_SEC:-1800}"
MAX_RESTART="${MAX_RESTART:-3}"
restarts=0
mkdir -p "$LOG"

log() { echo "[stall_wd $(date '+%F %T')] $*" >> "$LOG/stall.log"; }
log "기동 — STALL_SEC=$STALL_SEC MAX_RESTART=$MAX_RESTART"

while true; do
  sleep 60
  [[ -f "$MK/RETRO3_CHAIN_DONE" || -f "$MK/CHAIN_STUCK" ]] && { log "체인 종료 감지 — 워치독 종료"; exit 0; }
  now=$(date +%s)
  for f in "$ST"/*.json; do
    [[ -f "$f" ]] || continue
    state=$(python3 -c "import json;print(json.load(open('$f')).get('state',''))" 2>/dev/null || echo "")
    [[ "$state" == "running" ]] || continue
    upd=$(python3 -c "import json;print(int(json.load(open('$f')).get('updated_at',0)))" 2>/dev/null || echo 0)
    age=$((now - upd))
    if (( age >= STALL_SEC )); then
      if (( restarts >= MAX_RESTART )); then
        echo "{\"reason\": \"stall_max_restart\", \"stage\": \"$(basename "$f")\", \"ts\": $now}" > "$MK/CHAIN_STUCK"
        log "MAX_RESTART 초과 — CHAIN_STUCK 기록, 종료"
        exit 0
      fi
      log "스톨 감지: $(basename "$f") age=${age}s — 학습/평가 프로세스 kill (restart $((restarts+1))/$MAX_RESTART)"
      pkill -f "ego.step2_retrospection.(train|eval|hindsight)" 2>/dev/null || true
      pkill -f "tools/oom_opt/strip_eval.py" 2>/dev/null || true
      sleep 30
      pkill -9 -f "ego.step2_retrospection.(train|eval|hindsight)" 2>/dev/null || true
      restarts=$((restarts + 1))
      break
    fi
  done
done
