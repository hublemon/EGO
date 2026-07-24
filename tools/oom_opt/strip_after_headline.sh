#!/usr/bin/env bash
# strip-eval 조기 실행 런처 (사용자 승인 2026-07-24: "헤드라인 확보 직후").
#
# 기존 post_theta_ce_hook.sh 는 체인 완료 후 strip-eval 을 돌리게 돼 있다. 이 런처는
# **헤드라인(harden_sft_r15)이 확보된 직후** 같은 작업을 먼저 실행해 history 인과 실측을
# ~4시간 앞당긴다. strip_eval.py 가 S_STRIP_THETA_CE_DONE 마커를 쓰므로, 이쪽이 먼저 끝나면
# 훅의 기존 경로는 자동 no-op (멱등) — 훅을 수정/재시작할 필요가 없다.
#
# 안전 규약:
#   - 헤드라인(G-NH·U_g)이 이미 확보된 뒤에만 기동 → 최악의 경우 잃는 건 대조군 sft_r0 뿐이고
#     supervisor/orchestrator 가 자동 재시도한다.
#   - 다음 스테이지(sft_r0)가 모델 로드를 끝낼 시간을 준 뒤(GRACE) RAM/GPU 사전점검.
#   - RAM 판정은 anon 기준(회수 가능한 page cache 제외) — limit−current 는 캐시 때문에
#     과보수적이라 영원히 대기하게 된다(프레임 추출기가 그 사례).
#   - 여유가 없으면 무기한 대기만 한다. 아무것도 죽이지 않는다.
set -uo pipefail
REPO="${REPO:-/mnt/nvme/migration/jihun/EGO_jihun3}"
cd "$REPO"
PY="${PY:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
RUN="$REPO/runs/cesft_v2"
MK="$RUN/markers"
LOG="$RUN/logs/chain.log"
SLOG="$RUN/logs/strip_early.log"
GRACE="${GRACE:-300}"              # 다음 스테이지 로드 안정화 대기(초)
GPU_FREE_MIN="${GPU_FREE_MIN:-70000}"   # MiB
ANON_FREE_MIN_G="${ANON_FREE_MIN_G:-90}" # GB (240G cgroup 기준 anon 여유)

export RETRO3_RUNS="runs/cesft_v2" PYTHONPATH="src" HF_HOME="/mnt/nvme/cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"
export RETRO_NEXT_GAP_TEXT="after the current action ends"

log(){ echo "[strip_early $(date '+%F %T')] $*" | tee -a "$SLOG" >> "$LOG"; }

anon_free_g(){
  local lim anon
  lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
  [ "$lim" = "max" ] && lim=$((240*1024*1024*1024))
  anon=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null)
  [ -z "$anon" ] && { echo 999; return; }
  awk -v l="$lim" -v a="$anon" 'BEGIN{printf "%.0f",(l-a)/1073741824}'
}
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

LOCK="$MK/STRIP_EARLY_RUNNING"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "이미 실행 중 — 종료"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

log "시작 — S3H_SFT_R15_DONE(헤드라인) 대기"
while [ ! -f "$MK/S3H_SFT_R15_DONE" ]; do
  [ -f "$MK/S_STRIP_THETA_CE_DONE" ] && { log "strip 이미 완료 — 종료"; exit 0; }
  [ -f "$MK/CHAIN_STUCK" ] && { log "CHAIN_STUCK — 대기 중단"; exit 1; }
  sleep 30
done
log "헤드라인 확보 확인 (S3H_SFT_R15_DONE). ${GRACE}s 후 자원 점검"
sleep "$GRACE"

[ -f "$MK/S_STRIP_THETA_CE_DONE" ] && { log "strip 이미 완료 — 종료"; exit 0; }

# 자원 사전점검 — 여유 생길 때까지 대기만
waited=0
while true; do
  g=$(gpu_free); a=$(anon_free_g)
  if [ -n "$g" ] && [ "$g" -ge "$GPU_FREE_MIN" ] && [ "$a" -ge "$ANON_FREE_MIN_G" ]; then
    log "자원 OK — gpu_free=${g}MiB anon_free=${a}G → strip-eval 기동"
    break
  fi
  if [ $((waited % 600)) -eq 0 ]; then
    log "대기: gpu_free=${g}MiB(필요 ${GPU_FREE_MIN}) anon_free=${a}G(필요 ${ANON_FREE_MIN_G})"
  fi
  sleep 60; waited=$((waited + 60))
done

if [ ! -d "$REPO/outputs/step2_retrospection/cesft_v2/theta_ce/adapter" ]; then
  log "θ_CE adapter 없음 — 중단"; exit 1
fi

log "strip_eval.py 실행 시작"
"$PY" "$REPO/tools/oom_opt/strip_eval.py" >>"$SLOG" 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -f "$RUN/eval/strip_verdict.json" ]; then
  log "strip-eval 완료 — $($PY -c "
import json;d=json.load(open('$RUN/eval/strip_verdict.json'))
a=d['delta_acc_all'];print(f\"acc(hist)={d['acc_hist']:.4f} acc(strip)={d['acc_strip']:.4f} Δ={a['delta']:+.2f}pp CI{a['ci']} n={d['n_paired']} H8인과={d['gate_history_causal_H8']}\")" 2>/dev/null)"
else
  log "strip-eval 실패 (rc=$rc) — $SLOG 확인. 훅이 체인 완료 후 재시도함(멱등)"
fi
