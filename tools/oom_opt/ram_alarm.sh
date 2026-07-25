#!/usr/bin/env bash
# RAM 경보 — 관측 전용. 아무 프로세스도 죽이거나 재시작하지 않는다.
# 핸드오프 §3-3: mem.log 의 cur 가 THRESHOLD_G 를 2틱 연속 초과하면 경보를 남긴다.
# (RAM_FLOOR_GB=100 admission 게이트는 orchestrator 에 이미 활성 — 이건 사후 알림일 뿐.)
#
# theta_ce/supervisor/orchestrator/watchdog 를 건드리지 않는 독립 감시자.
# 기존 watchdog 를 재시작하면 안 되므로(절대 규칙), 경보 로직을 별도 프로세스로 붙인다.
set -uo pipefail
RUN="${RETRO3_RUNS_ABS:-/mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2}"
MEMLOG="$RUN/logs/mem.log"
ALARMLOG="$RUN/logs/ram_alarm.log"
THRESHOLD_G="${THRESHOLD_G:-225}"
POLL="${POLL:-30}"

log(){ echo "[ram_alarm $(date '+%F %T')] $*" >> "$ALARMLOG"; }

# 회수 불가 사용량(GB). memory.current 는 page cache 를 포함해서 압박 지표로 쓸 수 없다:
# frame_cache JPEG 쓰기만으로 cur 가 한도에 붙어(2026-07-24: anon 4.2G / file 227.7G)
# 3시간 내내 "OOM-kill 임박" 오경보가 찍혔지만 실제 oom_kill 은 0 이었다.
# anon + unevictable + slab_unreclaimable + dirty + writeback 만 센다. v2 → v1 폴백.
hard_gb() {
  local v
  v=$(awk '$1=="anon"||$1=="unevictable"||$1=="slab_unreclaimable"||$1=="file_dirty"||$1=="file_writeback"{s+=$2}
           END{if(s>0) printf "%.1f", s/1073741824}' /sys/fs/cgroup/memory.stat 2>/dev/null)
  if [ -z "$v" ]; then
    v=$(awk '$1=="total_rss"||$1=="total_unevictable"||$1=="total_dirty"||$1=="total_writeback"{s+=$2}
             END{if(s>0) printf "%.1f", s/1073741824}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null)
  fi
  [ -n "$v" ] && echo "$v"
}

log "start — threshold=${THRESHOLD_G}G(hard, page cache 제외) poll=${POLL}s memlog=$MEMLOG"

over=0
last_alarm_ts=0
while true; do
  # cgroup 직접 측정이 1순위. 실패하면 supervisor 가 mem.log 에 남긴 hard= 로 폴백.
  # (둘 다 없으면 이번 틱은 건너뛴다 — cur= 로는 경보하지 않는다.)
  cur=$(hard_gb)
  [ -z "$cur" ] && cur=$(tail -1 "$MEMLOG" 2>/dev/null | grep -oE 'hard=[0-9.]+G' | grep -oE '[0-9.]+' | head -1)
  if [ -n "$cur" ]; then
    # bc 없이 정수 비교: 소수 버림
    cur_int=${cur%.*}
    if [ "${cur_int:-0}" -ge "$THRESHOLD_G" ]; then
      over=$((over + 1))
    else
      over=0
    fi
    if [ "$over" -ge 2 ]; then
      now=$(date +%s)
      # 같은 경보 스팸 방지: 5분 쿨다운
      if [ $((now - last_alarm_ts)) -ge 300 ]; then
        log "ALARM — cgroup hard=${cur}G ≥ ${THRESHOLD_G}G (연속 ${over}틱, page cache 제외). OOM-kill(240G) 임박 가능."
        log "  top RSS: $(tail -1 "$MEMLOG" 2>/dev/null | sed 's/.*| //')"
        log "  GPU arm: $(pgrep -af 'ego.step2_retrospection.(train|eval)' 2>/dev/null | head -1)"
        last_alarm_ts=$now
      fi
    fi
  fi
  sleep "$POLL"
done
