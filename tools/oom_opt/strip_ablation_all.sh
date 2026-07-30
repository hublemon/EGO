#!/usr/bin/env bash
# history_strip ablation 러너 — base → sft_r15 → theta_ce(top-up) → 지표 집계.
#
# 사용자 요구(2026-07-25):
#   1) SSH 끊겨도 계속 진행 → setsid + nohup 으로 기동, 상태는 파일로만 남긴다.
#   2) OOM 안 나게 → 사전 admission 게이트 + 실행 중 감시자. 압박이 보이면
#      **커널 OOM-kill 전에 스스로 SIGTERM 으로 멈춘다**(records 는 append-only 라 재개 가능).
#   3) page cache 를 RAM 사용으로 오인 금지 → memory.current 대신 회수 불가분만 센다
#      (anon+unevictable+slab_unreclaimable+dirty+writeback). 2026-07-24 오경보의 원인이
#      file cache 227G 였다(실제 oom_kill 0). ram_alarm.sh 와 같은 규약.
#
# 각 arm 은 covered-only n=1000(seed 42) — base.records.jsonl 과 동일 모집단이라
# 세 체크포인트가 같은 셋에서 paired 비교된다.
#
# 산출물: runs/cesft_v2/eval/{base,sft_r15,theta_ce}_nohist.records.jsonl
#         runs/cesft_v2/eval/strip_metrics.json (대시보드 규약 지표)
#         마커: STRIP_ABLATION_ALL_DONE | STRIP_ABLATION_HALTED(중단 사유 포함)
set -uo pipefail

REPO="${REPO:-/mnt/nvme/migration/jihun/EGO_jihun3}"
cd "$REPO" || exit 1
PY="${PY:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
RUN="$REPO/runs/cesft_v2"
MK="$RUN/markers"
LOG="$RUN/logs/strip_ablation.log"

export RETRO3_RUNS="runs/cesft_v2" PYTHONPATH="src" HF_HOME="/mnt/nvme/cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"
export RETRO_NEXT_GAP_TEXT="after the current action ends"   # 체인과 동일 프롬프트

EVAL_N="${EVAL_N:-1000}"
BATCH="${BATCH:-24}"                 # 32 는 과거 성공값이나 여유를 두고 24
POLL="${POLL:-20}"                   # 감시 주기(초)
CG_LIMIT_G="${CG_LIMIT_G:-0}"        # 0 = cgroup memory.max 자동
RAM_STOP_G="${RAM_STOP_G:-195}"      # hard 사용량이 이 값 이상 2틱 → 중단 (240G 한도)
RAM_START_MAX_G="${RAM_START_MAX_G:-150}"  # 기동 전 hard 상한
GPU_FREE_MIN="${GPU_FREE_MIN:-60000}"      # 기동 전 GPU 여유(MiB)
GPU_FREE_STOP="${GPU_FREE_STOP:-1500}"     # 실행 중 GPU 여유가 이 밑 2틱 → 중단

mkdir -p "$MK" "$RUN/logs"
log(){ echo "[strip_abl $(date '+%F %T')] $*" | tee -a "$LOG"; }

# ── 메모리 판정: 회수 불가분만 (page cache 제외) ────────────────────────────────
hard_gb(){
  awk '$1=="anon"||$1=="unevictable"||$1=="slab_unreclaimable"||$1=="file_dirty"||$1=="file_writeback"{s+=$2}
       END{printf "%.1f", s/1073741824}' /sys/fs/cgroup/memory.stat 2>/dev/null
}
limit_g(){
  if [ "$CG_LIMIT_G" != "0" ]; then echo "$CG_LIMIT_G"; return; fi
  local v; v=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
  [ "$v" = "max" ] || [ -z "$v" ] && { echo 240; return; }
  awk -v v="$v" 'BEGIN{printf "%.0f", v/1073741824}'
}
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
ge(){ awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>=b)}'; }   # a >= b (소수 비교)

LOCK="$MK/STRIP_ABLATION_RUNNING"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "이미 실행 중(pid $(cat "$LOCK")) — 종료"; exit 0
fi
echo $$ > "$LOCK"
HALT_REASON=""
cleanup(){ rm -f "$LOCK"; }
trap cleanup EXIT

halt(){   # 사유를 남기고 전체 러너를 멈춘다 (다음 arm 도 시작하지 않음)
  HALT_REASON="$1"
  log "HALT — $1"
  printf '{"halted_at": %s, "reason": "%s"}\n' "$(date +%s)" "$1" > "$MK/STRIP_ABLATION_HALTED"
}

# ── 실행 중 감시자: 압박이 보이면 python 을 SIGTERM (커널 OOM-kill 선제 회피) ──
guard(){
  local pid="$1" arm="$2" ram_over=0 gpu_over=0 lim; lim=$(limit_g)
  while kill -0 "$pid" 2>/dev/null; do
    local h g; h=$(hard_gb); g=$(gpu_free)
    [ -z "$h" ] && h=0; [ -z "$g" ] && g=999999
    if ge "$h" "$RAM_STOP_G"; then ram_over=$((ram_over+1)); else ram_over=0; fi
    if [ "$g" -lt "$GPU_FREE_STOP" ]; then gpu_over=$((gpu_over+1)); else gpu_over=0; fi
    if [ "$ram_over" -ge 2 ]; then
      log "감시자: hard=${h}G ≥ ${RAM_STOP_G}G (한도 ${lim}G, page cache 제외) — $arm 중단 요청"
      kill -TERM "$pid" 2>/dev/null
      for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
      kill -KILL "$pid" 2>/dev/null
      echo "ram" > "$MK/.strip_guard_trip"; return
    fi
    if [ "$gpu_over" -ge 2 ]; then
      log "감시자: GPU free=${g}MiB < ${GPU_FREE_STOP} — $arm 중단 요청"
      kill -TERM "$pid" 2>/dev/null
      for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
      kill -KILL "$pid" 2>/dev/null
      echo "gpu" > "$MK/.strip_guard_trip"; return
    fi
    echo "$(date '+%F %T') hard=${h}G/${lim}G gpu_free=${g}MiB arm=$arm" >> "$RUN/logs/strip_abl_mem.log"
    sleep "$POLL"
  done
}

run_arm(){   # run_arm <arm> <adapter>
  local arm="$1" adapter="$2" alog="$RUN/logs/strip_${1}.log"
  [ -n "$HALT_REASON" ] && return 1

  # admission 게이트
  local h g; h=$(hard_gb); g=$(gpu_free)
  if ge "$h" "$RAM_START_MAX_G"; then
    halt "$arm 기동 거부 — hard=${h}G ≥ ${RAM_START_MAX_G}G"; return 1
  fi
  if [ "${g:-0}" -lt "$GPU_FREE_MIN" ]; then
    halt "$arm 기동 거부 — GPU free=${g}MiB < ${GPU_FREE_MIN}MiB (다른 GPU 잡?)"; return 1
  fi
  log "$arm 시작 — adapter='${adapter:-none}' n=$EVAL_N bs=$BATCH (hard=${h}G, gpu_free=${g}MiB)"

  rm -f "$MK/.strip_guard_trip"
  "$PY" "$REPO/tools/oom_opt/strip_eval.py" \
      --arm "$arm" --adapter "$adapter" --covered_only \
      --eval_n "$EVAL_N" --batch_size "$BATCH" >>"$alog" 2>&1 &
  local pid=$!
  guard "$pid" "$arm" &
  local gpid=$!
  wait "$pid"; local rc=$?
  kill "$gpid" 2>/dev/null; wait "$gpid" 2>/dev/null

  if [ -f "$MK/.strip_guard_trip" ]; then
    halt "$arm — 자원 압박으로 감시자가 선제 중단($(cat "$MK/.strip_guard_trip")). 재실행하면 이어서 진행."
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    # 커널 OOM-kill 흔적 확인 (page cache 오인과 구분)
    local ok; ok=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
    halt "$arm 실패 rc=$rc (cgroup oom_kill=${ok:-?}) — $alog 확인"
    return 1
  fi
  log "$arm 완료 — $(tail -1 "$alog" | cut -c1-200)"
  return 0
}

log "=== history_strip ablation 시작 (covered-only n=$EVAL_N, 한도 $(limit_g)G) ==="
log "GPU=$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1)"

run_arm base    ""                                                       || true
run_arm sft_r15 "outputs/step2_retrospection/cesft_v2/sft_r15/adapter"   || true
# theta_ce 는 full-set 서브샘플로 이미 돌아 covered 교집합이 187 뿐 → 같은 covered-1000 셋으로
# 보충(기존 행은 resume 로 건너뜀). 세 arm 동일 셋 DiD 를 위해 필요.
run_arm theta_ce "outputs/step2_retrospection/cesft_v2/theta_ce/adapter" || true

if [ -n "$HALT_REASON" ]; then
  log "중단 상태 — 확보된 arm 까지만 집계"
fi

log "지표 집계 (tools/strip_metrics.py · covered · video-cluster bootstrap)"
"$PY" "$REPO/tools/strip_metrics.py" --run runs/cesft_v2 \
    --arms base theta_ce sft_r15 \
    --did sft_r15:theta_ce theta_ce:base sft_r15:base \
    --out "$RUN/eval/strip_metrics.json" >>"$LOG" 2>&1 \
  && log "strip_metrics.json 갱신 완료" || log "집계 실패 — 로그 확인"

if [ -z "$HALT_REASON" ]; then
  printf '{"done_at": %s, "arms": ["base","sft_r15","theta_ce"], "eval_n": %s}\n' \
      "$(date +%s)" "$EVAL_N" > "$MK/STRIP_ABLATION_ALL_DONE"
  log "=== 전체 완료 ==="
else
  log "=== HALTED: $HALT_REASON ==="
fi
