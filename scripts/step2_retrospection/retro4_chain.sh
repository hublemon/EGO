#!/usr/bin/env bash
# retro4 무인 체인 — Phase-1 history-K8 prior (end−1s 계약) 재구축.
# retro3_chain.sh 파생: config·runs 디렉토리·S0 빌더·프롬프트 gap 문구가 다르고,
# GPU 무거운 S2 앞에서 retro3 메인 체인 완료(RETRO3_CHAIN_DONE)를 기다린다.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export PYTHONPATH="$REPO/src"
export HF_HOME=/mnt/nvme/cache
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RETRO3_RUNS="${RETRO3_RUNS:-runs/retro4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# end−1s 계약: 다음 행동은 '현재 행동이 끝난 뒤' (가변 horizon) — vlm 프롬프트 문구 주입
export RETRO_NEXT_GAP_TEXT="after the current action ends"

CFG=configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml
MK="$RETRO3_RUNS/markers"
LOG="$RETRO3_RUNS/logs"
mkdir -p "$MK" "$LOG"

# retro3 S7까지 끝나 GPU가 빌 때까지 대기 (최대 12h). ablation은 보류 상태.
wait_retro3() {
  local i
  [[ -f runs/retro3/markers/RETRO3_CHAIN_DONE ]] && return 0
  echo "[chain4] retro3 메인 체인 완료 대기 (RETRO3_CHAIN_DONE)" | tee -a "$LOG/chain.log"
  for i in $(seq 1 720); do
    [[ -f runs/retro3/markers/RETRO3_CHAIN_DONE ]] && return 0
    sleep 60
  done
  echo "[chain4] FAIL retro3 대기 12h 초과" | tee -a "$LOG/chain.log"
  echo "{\"failed\": \"wait_retro3\", \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"
  exit 1
}

preflight() {
  local gpu i
  for i in $(seq 1 120); do
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [[ "$gpu" -lt 30000 ]]; then return 0; fi
    echo "[chain4] preflight 대기: gpu_used=${gpu}MiB" | tee -a "$LOG/chain.log"
    sleep 60
  done
  echo "{\"failed\": \"preflight\", \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"
  exit 1
}

cat > "$RETRO3_RUNS/chain.json" <<'EOF'
{"stages": [
 {"id": "S0_support_train",  "marker": "S0_SUPPORT_DONE",          "est_sec": 600,  "title": "Phase-1 K8 Top-10 support + coverage"},
 {"id": "S1_context_train",  "marker": "S1_CONTEXT_DONE",          "est_sec": 120,  "title": "history/future 조인 (가변 horizon)"},
 {"id": "S1b_subset",        "marker": "S1B_SUBSET_DONE",          "est_sec": 10,   "title": "train 서브셋 선정"},
 {"id": "S2_base_trace_train","marker": "S2_BASE_TRACE_TRAIN_DONE","est_sec": 12600,"title": "Base Qwen trace (y-)"},
 {"id": "S3_hindsight",      "marker": "S3_HINDSIGHT_DONE",        "est_sec": 4000, "title": "teacher+projection+gate (y+)"},
 {"id": "S4_semantic",       "marker": "S4_SEMANTIC_DONE",         "est_sec": 1500, "title": "gemini pair 게이트 (병렬 16)"},
 {"id": "S5_pairs",          "marker": "S5_PAIRS_DONE",            "est_sec": 30,   "title": "DPO pair 구성 (+AO)"},
 {"id": "S6_dpo_d1",         "marker": "S6_DPO_D1_DONE",           "est_sec": 18000,"title": "FB-DPO D1 (fieldwise+clip, SFT 생략)"},
 {"id": "S7_eval_base",      "marker": "S7_EVAL_BASE_DONE",        "est_sec": 1500, "title": "배터리: base (acc|cov, Phase-1 prior)"},
 {"id": "S7_eval_dpo",       "marker": "S7_EVAL_DPO_D1_DONE",      "est_sec": 1500, "title": "배터리: DPO"},
 {"id": "S7_iv_base",        "marker": "S7_INTERVENTION_BASE_DONE","est_sec": 1500, "title": "개입 ③: base"},
 {"id": "S7_iv_dpo",         "marker": "S7_INTERVENTION_DPO_D1_DONE","est_sec": 1500,"title": "개입 ③: DPO"}
]}
EOF

run_stage() {
  local marker="$1"; shift
  local desc="$1"; shift
  if [[ -f "$MK/$marker" ]]; then
    echo "[chain4] SKIP $desc ($marker 존재)"
    return 0
  fi
  echo "[chain4] ==== $desc ===="
  "$@" >> "$LOG/chain.log" 2>&1
  local rc=$?
  if [[ -f "$MK/$marker" ]]; then
    echo "[chain4] OK   $desc (exit=$rc)"
    return 0
  fi
  echo "[chain4] FAIL $desc (exit=$rc, marker 없음)" | tee -a "$LOG/chain.log"
  echo "{\"failed\": \"$desc\", \"rc\": $rc, \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"
  exit 1
}

SUBSET_N="${SUBSET_N:-6000}"
EVAL_N="${EVAL_N:-1000}"
IV_N="${IV_N:-300}"

run_stage S0_SUPPORT_DONE  "S0 Phase-1 support"  "$PY" -m ego.step2_retrospection.data.build_support_hk8 --config "$CFG" --split both
run_stage S1_CONTEXT_DONE  "S1 context 조인"      "$PY" -m ego.step2_retrospection.data.build_context --split both --tau_exact 0
run_stage S1B_SUBSET_DONE  "S1b train 서브셋"     "$PY" -m ego.step2_retrospection.data.make_subset --n "$SUBSET_N"
wait_retro3
preflight
run_stage S2_BASE_TRACE_TRAIN_DONE "S2 base trace(train)" \
  "$PY" -m ego.step2_retrospection.prospection.base_trace --config "$CFG" --split train --subset "$RETRO3_RUNS/data/train_subset.json"
preflight
run_stage S3_HINDSIGHT_DONE "S3 hindsight(Ψ→Φ→gate)" \
  "$PY" -m ego.step2_retrospection.hindsight.projection --config "$CFG" --subset "$RETRO3_RUNS/data/train_subset.json" \
  --batch_size 16

if [[ ! -f "$MK/S4_SEMANTIC_DONE" && ! -f "$MK/S4_SEMANTIC_SKIPPED" ]]; then
  "$PY" -m ego.step2_retrospection.hindsight.semantic_gate --workers 16 >> "$LOG/chain.log" 2>&1 || true
fi

run_stage S5_PAIRS_DONE    "S5 pair 구성(+AO)"    "$PY" -m ego.step2_retrospection.pairs.build_pairs --action_only_aug
preflight
# 2026-07-24 사용자 확정: SFT 생략 — DPO D1은 base 직행(SFT 워밍업 미사용)이라 무관.
# r1_sft 배터리·개입도 함께 생략 (adapter 없음). retro3에서 이미 SFT 결과 확보.
run_stage S6_DPO_D1_DONE   "S6 DPO D1(fieldwise+clip)" "$PY" -m ego.step2_retrospection.train.dpo_fb --config "$CFG" --run_name dpo_d1 --loss_mode fieldwise --margin_clip 3.0

ADAPT_ROOT="outputs/step2_retrospection/retro4"
preflight
run_stage S7_EVAL_BASE_DONE   "S7 배터리 base"    "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm base --eval_n "$EVAL_N"
run_stage S7_EVAL_DPO_D1_DONE "S7 배터리 dpo_d1"  "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm dpo_d1 --adapter "$ADAPT_ROOT/dpo_d1/adapter" --eval_n "$EVAL_N"
run_stage S7_INTERVENTION_BASE_DONE   "S7 개입 base"   "$PY" -m ego.step2_retrospection.eval.intervention --config "$CFG" --arm base --n "$IV_N"
run_stage S7_INTERVENTION_DPO_D1_DONE "S7 개입 dpo_d1" "$PY" -m ego.step2_retrospection.eval.intervention --config "$CFG" --arm dpo_d1 --adapter "$ADAPT_ROOT/dpo_d1/adapter" --n "$IV_N"

echo "{\"ts\": $(date +%s)}" > "$MK/RETRO3_CHAIN_DONE"
echo "[chain4] 전체 완료"
