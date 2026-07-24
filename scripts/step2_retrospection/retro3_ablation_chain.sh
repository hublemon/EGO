#!/usr/bin/env bash
# DPO ablation 3-arm 체인 — 메인 체인(RETRO3_CHAIN_DONE) 완료를 기다렸다가 이어 붙는다.
#   arm 1  dpo_d1  (rule)  — 메인 체인이 이미 학습·평가 (여기서는 안 함)
#   arm 2  dpo_all (무조건 B≻A, 게이트 無, G3=warn — B0 문체-학습 재현 관측이 목적)
#   arm 3  dpo_sem (규칙+gemini) — LETSUR_API_KEY 있을 때만
# marker 기반 resume. start_ablation.sh 로 기동.
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

# 파일럿 계약 이월 ②: GPU 무거운 스테이지 앞에서 GPU<30GB 될 때까지 대기 (최대 2h)
preflight() {
  local gpu i
  for i in $(seq 1 120); do
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [[ "$gpu" -lt 30000 ]]; then return 0; fi
    echo "[ablation] preflight 대기: gpu_used=${gpu}MiB (60s 후 재확인)" | tee -a "$LOG/ablation.log"
    sleep 60
  done
  echo "[ablation] FAIL preflight 2h 초과 — 서버 상태 확인 필요" | tee -a "$LOG/ablation.log"
  exit 1
}
EVAL_N="${EVAL_N:-1000}"
IV_N="${IV_N:-300}"
ADAPT="outputs/step2_retrospection"

# 대시보드 chain.json에 ablation 스테이지 병합 (이미 있으면 그대로)
"$PY" - <<'EOF'
import json, os
p = os.path.join(os.environ.get("RETRO3_RUNS", "runs/retro3"), "chain.json")
try:
    d = json.load(open(p))
except Exception:
    d = {"stages": []}
add = [
    {"id": "S5_pairs_all",   "marker": "S5_PAIRS_ALL_DONE",          "est_sec": 30,   "title": "pair: 무조건 B≻A (ungated)"},
    {"id": "S6_dpo_all",     "marker": "S6_DPO_ALL_DONE",            "est_sec": 12000,"title": "DPO-all (게이트 無, G3=warn)"},
    {"id": "S7_eval_dpo_all","marker": "S7_EVAL_DPO_ALL_DONE",       "est_sec": 1500, "title": "배터리: DPO-all"},
    {"id": "S7_iv_dpo_all",  "marker": "S7_INTERVENTION_DPO_ALL_DONE","est_sec": 1500,"title": "개입 ③: DPO-all"},
    {"id": "S4_semantic",    "marker": "S4_SEMANTIC_DONE",           "est_sec": 3600, "title": "gemini pair 게이트 (sem)"},
    {"id": "S5_pairs_sem",   "marker": "S5_PAIRS_SEM_DONE",          "est_sec": 30,   "title": "pair: 규칙+gemini"},
    {"id": "S6_dpo_sem",     "marker": "S6_DPO_SEM_DONE",            "est_sec": 10000,"title": "DPO-sem"},
    {"id": "S7_eval_dpo_sem","marker": "S7_EVAL_DPO_SEM_DONE",       "est_sec": 1500, "title": "배터리: DPO-sem"},
    {"id": "S7_iv_dpo_sem",  "marker": "S7_INTERVENTION_DPO_SEM_DONE","est_sec": 1500,"title": "개입 ③: DPO-sem"},
]
have = {s["id"] for s in d["stages"]}
d["stages"].extend(s for s in add if s["id"] not in have)
json.dump(d, open(p, "w"), ensure_ascii=False)
print("chain.json merged")
EOF

echo "[ablation] 메인 체인 완료 대기 (RETRO3_CHAIN_DONE)"
until [[ -f "$MK/RETRO3_CHAIN_DONE" ]]; do
  [[ -f "$MK/CHAIN_STUCK" ]] && { echo "[ablation] 메인 체인 STUCK — 대기 유지"; }
  sleep 300
done
echo "[ablation] 메인 체인 완료 확인 — ablation 시작"

run_stage() {
  local marker="$1"; shift
  local desc="$1"; shift
  if [[ -f "$MK/$marker" ]]; then echo "[ablation] SKIP $desc"; return 0; fi
  echo "[ablation] ==== $desc ===="
  "$@" >> "$LOG/ablation.log" 2>&1
  local rc=$?
  if [[ -f "$MK/$marker" ]]; then echo "[ablation] OK $desc (exit=$rc)"; return 0; fi
  echo "[ablation] FAIL $desc (exit=$rc)" | tee -a "$LOG/ablation.log"
  echo "{\"failed\": \"$desc\", \"rc\": $rc, \"ts\": $(date +%s)}" > "$MK/ABLATION_FAILED"
  exit 1
}

# ---- arm: DPO-all (무조건 B≻A) ----
run_stage S5_PAIRS_ALL_DONE "S5 pair(all)" "$PY" -m ego.step2_retrospection.pairs.build_pairs --mode all
preflight
run_stage S6_DPO_ALL_DONE   "S6 DPO-all"   "$PY" -m ego.step2_retrospection.train.dpo_fb \
  --run_name dpo_all --pairs_file "$RETRO3_RUNS/data/pairs_train_all.jsonl" --g3_mode warn
run_stage S7_EVAL_DPO_ALL_DONE "S7 배터리 dpo_all" "$PY" -m ego.step2_retrospection.eval.battery \
  --arm dpo_all --adapter "$ADAPT/dpo_all/adapter" --eval_n "$EVAL_N"
run_stage S7_INTERVENTION_DPO_ALL_DONE "S7 개입 dpo_all" "$PY" -m ego.step2_retrospection.eval.intervention \
  --arm dpo_all --adapter "$ADAPT/dpo_all/adapter" --n "$IV_N"

# ---- arm: DPO-sem (규칙+gemini) — 키 있을 때만 ----
if [[ -n "${LETSUR_API_KEY:-}" ]]; then
  run_stage S4_SEMANTIC_DONE  "S4 semantic gate" "$PY" -m ego.step2_retrospection.hindsight.semantic_gate
  run_stage S5_PAIRS_SEM_DONE "S5 pair(sem)"     "$PY" -m ego.step2_retrospection.pairs.build_pairs --mode sem
  preflight
  run_stage S6_DPO_SEM_DONE   "S6 DPO-sem"       "$PY" -m ego.step2_retrospection.train.dpo_fb \
    --run_name dpo_sem --pairs_file "$RETRO3_RUNS/data/pairs_train_sem.jsonl"
  run_stage S7_EVAL_DPO_SEM_DONE "S7 배터리 dpo_sem" "$PY" -m ego.step2_retrospection.eval.battery \
    --arm dpo_sem --adapter "$ADAPT/dpo_sem/adapter" --eval_n "$EVAL_N"
  run_stage S7_INTERVENTION_DPO_SEM_DONE "S7 개입 dpo_sem" "$PY" -m ego.step2_retrospection.eval.intervention \
    --arm dpo_sem --adapter "$ADAPT/dpo_sem/adapter" --n "$IV_N"
else
  echo "[ablation] LETSUR_API_KEY 없음 — DPO-sem arm 보류 (키 export 후 재기동하면 이어서 실행)"
  echo "{\"ts\": $(date +%s)}" > "$MK/ABLATION_SEM_PENDING"
fi

echo "{\"ts\": $(date +%s)}" > "$MK/RETRO3_ABLATION_DONE"
echo "[ablation] 완료"
