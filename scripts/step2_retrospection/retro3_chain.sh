#!/usr/bin/env bash
# Retrospection 무인 체인 — ssh 끊겨도 진행 (start_chain.sh로 기동).
# marker 기반 resume: 이미 완료된 스테이지는 건너뛴다. 각 파이썬 모듈도 자체 resume.
# 성공 판정은 exit code가 아니라 marker (decord 소멸자 크래시 내성).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export PYTHONPATH="$REPO/src"
export HF_HOME=/mnt/nvme/cache
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RETRO3_RUNS="${RETRO3_RUNS:-runs/retro3}"
export TOKENIZERS_PARALLELISM=false
# 파일럿 계약 이월 ①: allocator 파편화 방지 (연산·결과 불변, 메모리 배치만 변경)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MK="$RETRO3_RUNS/markers"
LOG="$RETRO3_RUNS/logs"
mkdir -p "$MK" "$LOG"

# 파일럿 계약 이월 ②: GPU 무거운 스테이지 앞에서 GPU<30GB 될 때까지 대기 (최대 2h).
# 시작 시점만 미루는 순수 운영 장치 — 계산에는 관여하지 않는다.
preflight() {
  local gpu i
  for i in $(seq 1 120); do
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [[ "$gpu" -lt 30000 ]]; then return 0; fi
    echo "[chain] preflight 대기: gpu_used=${gpu}MiB (60s 후 재확인)" | tee -a "$LOG/chain.log"
    sleep 60
  done
  echo "[chain] FAIL preflight 2h 초과 — 서버 상태 확인 필요" | tee -a "$LOG/chain.log"
  echo "{\"failed\": \"preflight\", \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"
  exit 1
}

# 대시보드용 체인 계획 (스테이지 이름 = status/<stage>.json 과 매칭)
cat > "$RETRO3_RUNS/chain.json" <<'EOF'
{"stages": [
 {"id": "S0_support_train",  "marker": "S0_SUPPORT_DONE",          "est_sec": 900,  "title": "WM Top-10 support + coverage"},
 {"id": "S1_context_train",  "marker": "S1_CONTEXT_DONE",          "est_sec": 120,  "title": "history/future 조인"},
 {"id": "S1b_subset",        "marker": "S1B_SUBSET_DONE",          "est_sec": 10,   "title": "train 서브셋 선정"},
 {"id": "S2_base_trace_train","marker": "S2_BASE_TRACE_TRAIN_DONE","est_sec": 7200, "title": "Base Qwen trace (y-)"},
 {"id": "S3_hindsight",      "marker": "S3_HINDSIGHT_DONE",        "est_sec": 10000,"title": "teacher+projection+gate (y+)"},
 {"id": "S4_semantic",       "marker": "S4_SEMANTIC_DONE",         "est_sec": 3600, "title": "gemini pair 게이트"},
 {"id": "S5_pairs",          "marker": "S5_PAIRS_DONE",            "est_sec": 30,   "title": "DPO pair 구성"},
 {"id": "S6_r1_sft",         "marker": "S6_R1_SFT_DONE",           "est_sec": 10000,"title": "R1 SFT (projected trace)"},
 {"id": "S6_dpo_d1",         "marker": "S6_DPO_D1_DONE",           "est_sec": 14000,"title": "Field-balanced DPO (D1)"},
 {"id": "S7_eval_base",      "marker": "S7_EVAL_BASE_DONE",        "est_sec": 1500, "title": "배터리: base"},
 {"id": "S7_eval_r1",        "marker": "S7_EVAL_R1_SFT_DONE",      "est_sec": 1500, "title": "배터리: R1"},
 {"id": "S7_eval_dpo",       "marker": "S7_EVAL_DPO_D1_DONE",      "est_sec": 1500, "title": "배터리: DPO"},
 {"id": "S7_iv_base",        "marker": "S7_INTERVENTION_BASE_DONE","est_sec": 1500, "title": "개입 ③: base"},
 {"id": "S7_iv_r1",          "marker": "S7_INTERVENTION_R1_SFT_DONE","est_sec": 1500,"title": "개입 ③: R1"},
 {"id": "S7_iv_dpo",         "marker": "S7_INTERVENTION_DPO_D1_DONE","est_sec": 1500,"title": "개입 ③: DPO"}
]}
EOF

run_stage() {  # run_stage <marker> <설명> <cmd...>
  local marker="$1"; shift
  local desc="$1"; shift
  if [[ -f "$MK/$marker" ]]; then
    echo "[chain] SKIP $desc ($marker 존재)"
    return 0
  fi
  echo "[chain] ==== $desc ===="
  "$@" >> "$LOG/chain.log" 2>&1
  local rc=$?
  if [[ -f "$MK/$marker" ]]; then
    echo "[chain] OK   $desc (exit=$rc, marker 확인)"
    return 0
  fi
  echo "[chain] FAIL $desc (exit=$rc, marker 없음)" | tee -a "$LOG/chain.log"
  echo "{\"failed\": \"$desc\", \"rc\": $rc, \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"
  exit 1
}

SUBSET_N="${SUBSET_N:-6000}"
EVAL_N="${EVAL_N:-1000}"
IV_N="${IV_N:-300}"

run_stage S0_SUPPORT_DONE  "S0 support 덤프"      "$PY" -m ego.step2_retrospection.data.build_support --split both
run_stage S1_CONTEXT_DONE  "S1 context 조인"      "$PY" -m ego.step2_retrospection.data.build_context --split both
run_stage S1B_SUBSET_DONE  "S1b train 서브셋"     "$PY" -m ego.step2_retrospection.data.make_subset --n "$SUBSET_N"
preflight
run_stage S2_BASE_TRACE_TRAIN_DONE "S2 base trace(train)" \
  "$PY" -m ego.step2_retrospection.prospection.base_trace --split train --subset "$RETRO3_RUNS/data/train_subset.json"
preflight
run_stage S3_HINDSIGHT_DONE "S3 hindsight(Ψ→Φ→gate)" \
  "$PY" -m ego.step2_retrospection.hindsight.projection --subset "$RETRO3_RUNS/data/train_subset.json" \
  --batch_size 16

# S4: 키 있으면 판정, 없으면 skip 마커 (pair는 규칙 판정으로 진행)
if [[ ! -f "$MK/S4_SEMANTIC_DONE" && ! -f "$MK/S4_SEMANTIC_SKIPPED" ]]; then
  "$PY" -m ego.step2_retrospection.hindsight.semantic_gate >> "$LOG/chain.log" 2>&1 || true
fi

run_stage S5_PAIRS_DONE    "S5 pair 구성"         "$PY" -m ego.step2_retrospection.pairs.build_pairs
preflight
run_stage S6_R1_SFT_DONE   "S6 R1 SFT"            "$PY" -m ego.step2_retrospection.train.sft_r1 --run_name r1_sft --epochs 1
run_stage S6_DPO_D1_DONE   "S6 DPO D1"            "$PY" -m ego.step2_retrospection.train.dpo_fb --run_name dpo_d1

ADAPT_ROOT="outputs/step2_retrospection"
preflight
run_stage S7_EVAL_BASE_DONE   "S7 배터리 base"    "$PY" -m ego.step2_retrospection.eval.battery --arm base --eval_n "$EVAL_N"
run_stage S7_EVAL_R1_SFT_DONE "S7 배터리 r1_sft"  "$PY" -m ego.step2_retrospection.eval.battery --arm r1_sft --adapter "$ADAPT_ROOT/r1_sft/adapter" --eval_n "$EVAL_N"
run_stage S7_EVAL_DPO_D1_DONE "S7 배터리 dpo_d1"  "$PY" -m ego.step2_retrospection.eval.battery --arm dpo_d1 --adapter "$ADAPT_ROOT/dpo_d1/adapter" --eval_n "$EVAL_N"
run_stage S7_INTERVENTION_BASE_DONE   "S7 개입 base"   "$PY" -m ego.step2_retrospection.eval.intervention --arm base --n "$IV_N"
run_stage S7_INTERVENTION_R1_SFT_DONE "S7 개입 r1_sft" "$PY" -m ego.step2_retrospection.eval.intervention --arm r1_sft --adapter "$ADAPT_ROOT/r1_sft/adapter" --n "$IV_N"
run_stage S7_INTERVENTION_DPO_D1_DONE "S7 개입 dpo_d1" "$PY" -m ego.step2_retrospection.eval.intervention --arm dpo_d1 --adapter "$ADAPT_ROOT/dpo_d1/adapter" --n "$IV_N"

echo "{\"ts\": $(date +%s)}" > "$MK/RETRO3_CHAIN_DONE"
echo "[chain] 전체 완료"
