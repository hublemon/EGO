#!/usr/bin/env bash
# S3 굳히기 무인 체인 (ssh 끊겨도 진행) — 자기복구 supervisor + 메모리 워치독 내장.
# 단계:
#   [결정적, eval만] retro3 base/r1_sft 개입③ 강화(n=1000, CI·필드분해·직교성)
#   [확증, GPU]      retro4(Phase-1 prior) base 배터리 → SFT → SFT 배터리 → 강화
# marker 기반 resume. start_s3harden.sh로 기동.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export PYTHONPATH="$REPO/src" HF_HOME=/mnt/nvme/cache TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"   # retro4 계약 (retro3 스테이지엔 무해)

RUNS=runs/s3harden
MK="$RUNS/markers"; LOG="$RUNS/logs"
mkdir -p "$MK" "$LOG"
CFG3=configs/step2_retrospection/goalstep_start_m1_lobs8.yaml
CFG4=configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml
ADAPT=outputs/step2_retrospection
MAX_RETRY="${MAX_RETRY:-4}"

# ── 메모리 워치독 (S3 OOM 이력 대응, 60s 주기)
mem_watchdog() {
  while true; do
    awk -v ts="$(date '+%F %T')" \
        -v cur="$(cat /sys/fs/cgroup/memory.current 2>/dev/null||echo 0)" \
        -v peak="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null||echo 0)" \
        'BEGIN{g=1073741824;printf "%s cur=%.1fG peak=%.1fG\n",ts,cur/g,peak/g}' >> "$LOG/mem.log"
    sleep 60
  done
}
mem_watchdog & WD=$!
trap 'kill "$WD" 2>/dev/null' EXIT

# ── run_stage <sentinel> <설명> <RETRO3_RUNS> <cmd...>
run_stage() {
  local sent="$1"; shift; local desc="$1"; shift; local rr="$1"; shift
  if [[ -f "$MK/$sent" ]]; then echo "[s3h] SKIP $desc"; return 0; fi
  echo "[s3h] ==== $desc ===="
  RETRO3_RUNS="$rr" "$@" >> "$LOG/chain.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then echo "{\"ts\":$(date +%s)}" > "$MK/$sent"; echo "[s3h] OK $desc"; return 0; fi
  echo "[s3h] FAIL $desc (rc=$rc)" | tee -a "$LOG/chain.log"
  echo "{\"failed\":\"$desc\",\"rc\":$rc,\"ts\":$(date +%s)}" > "$MK/CHAIN_FAILED"
  return 1
}

preflight() {
  local g i; for i in $(seq 1 120); do
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
    [[ "$g" -lt 30000 ]] && return 0
    echo "[s3h] preflight gpu=${g}MiB" | tee -a "$LOG/chain.log"; sleep 60
  done; return 1
}

chain() {
  # ── 결정적: retro3 강화 (records 이미 존재, eval만) ───────────────────
  run_stage S3H_BASE_DONE   "retro3 강화: base"   runs/retro3 \
    "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG3" --arm base --n 1000 || return 1
  run_stage S3H_R1_SFT_DONE "retro3 강화: r1_sft" runs/retro3 \
    "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG3" --arm r1_sft \
    --adapter "$ADAPT/r1_sft/adapter" --n 1000 || return 1
  echo "{\"ts\":$(date +%s)}" > "$MK/S3H_DECISIVE_DONE"

  # ── 확증: retro4 Phase-1 prior 재현 (GPU) ─────────────────────────────
  preflight || return 1
  run_stage R4_EVAL_BASE_DONE "retro4 배터리: base" runs/retro4 \
    "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG4" --arm base --eval_n 1000 || return 1
  preflight || return 1
  run_stage R4_SFT_DONE "retro4 SFT" runs/retro4 \
    "$PY" -m ego.step2_retrospection.train.sft_r1 --config "$CFG4" --run_name r1_sft_r4 --epochs 1 || return 1
  preflight || return 1
  run_stage R4_EVAL_SFT_DONE "retro4 배터리: r1_sft_r4" runs/retro4 \
    "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG4" --arm r1_sft_r4 \
    --adapter "$ADAPT/retro4/r1_sft_r4/adapter" --eval_n 1000 || return 1
  run_stage R4_S3H_BASE_DONE "retro4 강화: base" runs/retro4 \
    "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG4" --arm base --n 1000 || return 1
  run_stage R4_S3H_SFT_DONE "retro4 강화: r1_sft_r4" runs/retro4 \
    "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG4" --arm r1_sft_r4 \
    --adapter "$ADAPT/retro4/r1_sft_r4/adapter" --n 1000 || return 1

  echo "{\"ts\":$(date +%s)}" > "$MK/S3HARDEN_CHAIN_DONE"
  echo "[s3h] 전체 완료"
}

# ── 자기복구 supervisor 루프
retry=0; last=""
while true; do
  [[ -f "$MK/S3HARDEN_CHAIN_DONE" ]] && { echo "[s3h] 완료 — 종료" | tee -a "$LOG/super.log"; break; }
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "{\"reason\":\"gpu_down\",\"ts\":$(date +%s)}" > "$MK/CHAIN_STUCK"
    echo "[s3h] GPU 불가 — 정지" | tee -a "$LOG/super.log"; break
  fi
  echo "[s3h] chain 시작 (try=$((retry+1)))" | tee -a "$LOG/super.log"
  rm -f "$MK/CHAIN_FAILED"
  chain && continue
  sig="$(cat "$MK/CHAIN_FAILED" 2>/dev/null|tr -d ' \n')"
  [[ "$sig" == "$last" ]] && retry=$((retry+1)) || { retry=1; last="$sig"; }
  echo "[s3h] 중단 ($sig, $retry/$MAX_RETRY)" | tee -a "$LOG/super.log"
  if [[ $retry -ge $MAX_RETRY ]]; then
    echo "{\"reason\":\"repeated\",\"sig\":\"$sig\"}" > "$MK/CHAIN_STUCK"
    echo "[s3h] 반복 실패 — STUCK (개입 필요)" | tee -a "$LOG/super.log"; break
  fi
  sleep 30
done
