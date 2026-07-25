#!/usr/bin/env bash
# candidate-CE ↔ projected-SFT 조합 실험 무인 체인 (cesft_v2).
# SSOT: EGO_jihun/docs/experiments/2026-07-24_ce_sft_combination_literature_handoff.md (조합 §2/§5 + 부록A)
#       + EGO_jihun3/docs/experiments/2026-07-24_ce_sft_methodology_v2_handoff.md (CE/SFT 상세·게이트)
# marker 멱등: 완료 스테이지 skip. 각 스테이지 후 artifact.html 재굽기.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export PYTHONPATH="$REPO/src"
export HF_HOME=/mnt/nvme/cache
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RETRO3_RUNS="${RETRO3_RUNS:-runs/cesft_v2}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"

CFG=configs/step2_retrospection/cesft_v2.yaml
RUNS="$RETRO3_RUNS"
MK="$RUNS/markers"; LOG="$RUNS/logs"; EVAL="$RUNS/eval"
ADAPT="outputs/step2_retrospection/cesft_v2"
mkdir -p "$MK" "$LOG" "$EVAL"

EVAL_N="${EVAL_N:-1000}"        # battery covered eval 크기
IV_N="${IV_N:-800}"            # harden_s3 개입 크기
CE_EPOCHS="${CE_EPOCHS:-1}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
TAU="${TAU:-1.0}"
CSTACK_STEPS="${CSTACK_STEPS:-150}"   # 부록A equal-budget 추가 CE 스텝
BEST_R="${BEST_R:-sft_r15}"           # 기본 조합 arm (G-NH 로 사후 선택 가능)
RUN_BASELINE_EXTRA="${RUN_BASELINE_EXTRA:-1}"  # no_video/no_history/random 포함(Standard)
RUN_APPENDIX_A="${RUN_APPENDIX_A:-1}"          # 부록A 3-stage (P-UTIL 게이트)

bake() { "$PY" tools/cesft_v2_artifact.py --run "$RUNS" --out "$RUNS/artifact.html" \
         --now "$(date -Iseconds)" >/dev/null 2>&1 || true; }

preflight() {
  local gpu i
  for i in $(seq 1 180); do
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [[ "$gpu" -lt 30000 ]]; then return 0; fi
    echo "[cesft_v2] preflight 대기: gpu_used=${gpu}MiB" | tee -a "$LOG/chain.log"
    sleep 60
  done
  echo "{\"failed\": \"preflight\", \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"; exit 1
}

run_stage() {
  local marker="$1"; shift; local desc="$1"; shift
  if [[ -f "$MK/$marker" ]]; then echo "[cesft_v2] SKIP $desc ($marker)"; return 0; fi
  echo "[cesft_v2] ==== $desc ===="
  "$@" >> "$LOG/chain.log" 2>&1; local rc=$?
  if [[ -f "$MK/$marker" ]]; then echo "[cesft_v2] OK $desc (exit=$rc)"; bake; return 0; fi
  echo "[cesft_v2] FAIL $desc (exit=$rc, marker 없음)" | tee -a "$LOG/chain.log"
  echo "{\"failed\": \"$desc\", \"rc\": $rc, \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"; exit 1
}

# 게이트 유틸 (records 재처리 — GPU 불필요)
gate() { "$PY" tools/paired_boot.py --run "$RUNS" "$@" >> "$LOG/chain.log" 2>&1 || true; }

# dashboard plan
cat > "$RUNS/chain.json" <<'EOF'
{"stages": [
 {"id": "E1_theta_ce",   "marker": "S_CE_THETA_CE_DONE",     "est_sec": 9000, "title": "E1 θ_CE (WM-candidate selection-CE, video-grounded)"},
 {"id": "E1_eval",       "marker": "S7_EVAL_THETA_CE_DONE",  "est_sec": 1200, "title": "E1 θ_CE 배터리 + G-ACC1(vs WM-top1)"},
 {"id": "E1b_candfree",  "marker": "S_CE_CAND_FREE_DONE",    "est_sec": 8000, "title": "E1b candidate-free CE (GT CE 자체 효과)"},
 {"id": "E1b_eval",      "marker": "S7_EVAL_CAND_FREE_DONE", "est_sec": 1200, "title": "E1b 배터리 + 성립부등식(WM-cand>cand-free)"},
 {"id": "E1b_nohist",    "marker": "S_CE_NO_HISTORY_DONE",   "est_sec": 8000, "title": "E1b no-history CE"},
 {"id": "E2_r0",         "marker": "S6_SFT_R0_DONE",         "est_sec": 7500, "title": "① sft_r0 (앵커없는 순차 — 덮어쓰기 baseline)"},
 {"id": "E2_r15",        "marker": "S6_SFT_R15_DONE",        "est_sec": 8500, "title": "① sft_r15 (CE-replay ρ=0.15 기본)"},
 {"id": "E2_r30",        "marker": "S6_SFT_R30_DONE",        "est_sec": 9500, "title": "① sft_r30 (CE-replay ρ=0.30 fallback)"},
 {"id": "E2_eval",       "marker": "S7_EVAL_SFT_R30_DONE",   "est_sec": 4500, "title": "① 배터리+harden(U_g)+G-NH (r0/r15/r30)"},
 {"id": "E4_wise",       "marker": "S_WISE_DONE",            "est_sec": 4500, "title": "④ WiSE-FT α∈{.25,.5,.75} frontier (학습0)"},
 {"id": "EA_cstack",     "marker": "S_CE_C_STACK_DONE",      "est_sec": 3500, "title": "부록A C-stack (B0+CE, P-UTIL 게이트)"},
 {"id": "EA_cctrl",      "marker": "S_CE_C_CTRL_DONE",       "est_sec": 3500, "title": "부록A C-ctrl (θ_CE+CE equal-budget)"},
 {"id": "EA_eval",       "marker": "S7_EVAL_C_CTRL_DONE",     "est_sec": 2400, "title": "부록A 배터리 + T-ACC (C-stack>B0,>C-ctrl)"},
 {"id": "E_report",      "marker": "CESFT_V2_CHAIN_DONE",    "est_sec": 60,   "title": "리포트 아티팩트 확정"}
]}
EOF

bake

# ───────────────────────── E1: θ_CE (생사: G-ACC1) ─────────────────────────
preflight
run_stage S_CE_THETA_CE_DONE "E1 θ_CE (wm_cand)" \
  "$PY" -m ego.step2_retrospection.train.select_ce --config "$CFG" --run_name theta_ce \
    --arm wm_cand --tau "$TAU" --epochs "$CE_EPOCHS"
preflight
run_stage S7_EVAL_THETA_CE_DONE "E1 배터리 theta_ce" \
  "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm theta_ce \
    --adapter "$ADAPT/theta_ce/adapter" --eval_n "$EVAL_N"
gate --arm_a theta_ce --gate G-ACC1 --out "$EVAL/paired_G-ACC1_theta_ce.json"

# ───────────────────────── E1b: baseline arms (§3) ─────────────────────────
preflight
run_stage S_CE_CAND_FREE_DONE "E1b candidate-free CE" \
  "$PY" -m ego.step2_retrospection.train.select_ce --config "$CFG" --run_name cand_free \
    --arm cand_free --tau "$TAU" --epochs "$CE_EPOCHS"
preflight
run_stage S7_EVAL_CAND_FREE_DONE "E1b 배터리 cand_free" \
  "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm cand_free \
    --adapter "$ADAPT/cand_free/adapter" --eval_n "$EVAL_N"
# 성립부등식: WM-candidate CE > candidate-free CE (SelAcc paired Δ)
gate --arm_a theta_ce --arm_b cand_free --gate G-DELTA --metric SelAcc \
     --out "$EVAL/paired_G-DELTA_theta_ce_vs_cand_free.json"

if [[ "$RUN_BASELINE_EXTRA" == "1" ]]; then
  # 2026-07-24: baseline은 no_history만 (random_cand·no_video 제거 — WM prior 후보가
  # 데이터에 이미 shuffle 저장되어 위치/객관식 confound는 wm_cand 안에서 통제됨).
  for arm in no_history; do
    MARK="S_CE_$(echo "$arm" | tr '[:lower:]' '[:upper:]')_DONE"
    preflight
    run_stage "$MARK" "E1b $arm CE" \
      "$PY" -m ego.step2_retrospection.train.select_ce --config "$CFG" --run_name "$arm" \
        --arm "$arm" --tau "$TAU" --epochs "$CE_EPOCHS"
    preflight
    run_stage "S7_EVAL_$(echo "$arm"|tr '[:lower:]' '[:upper:]')_DONE" "E1b 배터리 $arm" \
      "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm "$arm" \
        --adapter "$ADAPT/$arm/adapter" --eval_n "$EVAL_N"
  done
fi

# ───────────────── ① CE-replay 앵커드 SFT (r ∈ {0, .15, .30}) ─────────────────
declare -A RHO=( [sft_r0]=0.0 [sft_r15]=0.15 [sft_r30]=0.30 )
for run in sft_r0 sft_r15 sft_r30; do
  MARK="S6_$(echo "$run"|tr '[:lower:]' '[:upper:]')_DONE"
  preflight
  run_stage "$MARK" "① $run (ρ=${RHO[$run]})" \
    "$PY" -m ego.step2_retrospection.train.sft_v2 --config "$CFG" --run_name "$run" \
      --init_adapter "$ADAPT/theta_ce/adapter" --ce_replay_rho "${RHO[$run]}" \
      --ce_tau "$TAU" --epochs "$SFT_EPOCHS"
done
# 배터리 + harden(U_g/D_g/correct-switch) + G-NH(vs θ_CE) — 각 r
for run in sft_r0 sft_r15 sft_r30; do
  preflight
  run_stage "S7_EVAL_$(echo "$run"|tr '[:lower:]' '[:upper:]')_DONE" "① 배터리 $run" \
    "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm "$run" \
      --adapter "$ADAPT/$run/adapter" --eval_n "$EVAL_N"
  preflight
  run_stage "S3H_$(echo "$run"|tr '[:lower:]' '[:upper:]')_DONE" "① harden $run (U_g)" \
    "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG" --arm "$run" \
      --adapter "$ADAPT/$run/adapter" --n "$IV_N"
  gate --arm_a "$run" --arm_b theta_ce --gate G-NH \
       --out "$EVAL/paired_G-NH_${run}_vs_theta_ce.json"
done

# ───────────────── ④ WiSE-FT frontier (θ_CE ⊕ BEST_R, 학습 0) ─────────────────
if [[ ! -f "$MK/S_WISE_DONE" ]]; then
  echo "[cesft_v2] ==== ④ WiSE-FT frontier (BEST_R=$BEST_R) ===="
  for a in 0.25 0.50 0.75; do
    tag="wise_a${a/./}"
    if [[ ! -d "$ADAPT/$tag/adapter" ]]; then
      "$PY" tools/merge_adapters.py --adapter_a "$ADAPT/theta_ce/adapter" \
        --adapter_b "$ADAPT/$BEST_R/adapter" --alpha "$a" \
        --out "$ADAPT/$tag/adapter" >> "$LOG/chain.log" 2>&1 || true
    fi
    preflight
    [[ -f "$EVAL/$tag.json" ]] || "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" \
      --arm "$tag" --adapter "$ADAPT/$tag/adapter" --eval_n "$EVAL_N" >> "$LOG/chain.log" 2>&1
    preflight
    [[ -f "$EVAL/$tag.harden_s3.json" ]] || "$PY" -m ego.step2_retrospection.eval.harden_s3 \
      --config "$CFG" --arm "$tag" --adapter "$ADAPT/$tag/adapter" --n "$IV_N" >> "$LOG/chain.log" 2>&1
  done
  # frontier json (α: SelAcc × causal_sensitivity) — 베이커가 읽음
  "$PY" - "$RUNS" "$BEST_R" >> "$LOG/chain.log" 2>&1 <<'PYEOF'
import json,sys,pathlib
runs=pathlib.Path(sys.argv[1]); best=sys.argv[2]; ev=runs/"eval"
pts=[]
def load(a):
    b=ev/f"{a}.json"; h=ev/f"{a}.harden_s3.json"
    if not b.is_file(): return None
    bj=json.loads(b.read_text()); hj=json.loads(h.read_text()) if h.is_file() else {}
    cs=(hj.get("causal_sensitivity_ci",{}).get("both",{}) or {}).get("point")
    ug=(hj.get("utility_belief_only_ci",{}) or {}).get("point")
    return {"SelAcc":bj.get("acc"),"GADR":bj.get("G2_correction"),"causal_sensitivity":cs,"U_g":ug}
for a,arm in [(0.0,"theta_ce"),(0.25,"wise_a025"),(0.5,"wise_a050"),(0.75,"wise_a075"),(1.0,best)]:
    d=load(arm)
    if d: pts.append({"alpha":a,"arm":arm,**d})
(ev/"wise_ft_frontier.json").write_text(json.dumps(pts,indent=1,ensure_ascii=False))
print("[wise] frontier pts:",len(pts))
PYEOF
  echo "{\"ts\": $(date +%s)}" > "$MK/S_WISE_DONE"; bake
fi

# ───────────────── 부록A: 3-stage (P-UTIL 게이트) ─────────────────
if [[ "$RUN_APPENDIX_A" == "1" ]]; then
  # P-UTIL: B0(=BEST_R) 의 belief-only utility U_g CI 하한 > 0 이어야 C-stack 착수
  PUTIL=$("$PY" - "$RUNS" "$BEST_R" <<'PYEOF'
import json,sys,pathlib
h=pathlib.Path(sys.argv[1])/"eval"/f"{sys.argv[2]}.harden_s3.json"
try:
    lo=json.loads(h.read_text())["utility_belief_only_ci"]["lo"]
    print("PASS" if lo>0 else "FAIL")
except Exception:
    print("MISSING")
PYEOF
)
  echo "[cesft_v2] P-UTIL($BEST_R) = $PUTIL" | tee -a "$LOG/chain.log"
  echo "{\"p_util\": \"$PUTIL\", \"best_r\": \"$BEST_R\", \"ts\": $(date +%s)}" > "$MK/P_UTIL_$PUTIL"
  if [[ "$PUTIL" == "PASS" ]]; then
    preflight
    run_stage S_CE_C_STACK_DONE "부록A C-stack (B0+${CSTACK_STEPS}CE)" \
      "$PY" -m ego.step2_retrospection.train.select_ce --config "$CFG" --run_name c_stack \
        --arm wm_cand --tau "$TAU" --init_adapter "$ADAPT/$BEST_R/adapter" --max_steps "$CSTACK_STEPS"
    preflight
    run_stage S_CE_C_CTRL_DONE "부록A C-ctrl (θ_CE+${CSTACK_STEPS}CE equal-budget)" \
      "$PY" -m ego.step2_retrospection.train.select_ce --config "$CFG" --run_name c_ctrl \
        --arm wm_cand --tau "$TAU" --init_adapter "$ADAPT/theta_ce/adapter" --max_steps "$CSTACK_STEPS"
    for run in c_stack c_ctrl; do
      preflight
      run_stage "S7_EVAL_$(echo "$run"|tr '[:lower:]' '[:upper:]')_DONE" "부록A 배터리 $run" \
        "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm "$run" \
          --adapter "$ADAPT/$run/adapter" --eval_n "$EVAL_N"
    done
    # T-ACC: SelAcc(C-stack) > B0 그리고 > C-ctrl (paired ΔCI 하한 > 0)
    gate --arm_a c_stack --arm_b "$BEST_R" --gate G-DELTA --metric SelAcc \
         --out "$EVAL/paired_TACC_cstack_vs_b0.json"
    gate --arm_a c_stack --arm_b c_ctrl --gate G-DELTA --metric SelAcc \
         --out "$EVAL/paired_TACC_cstack_vs_cctrl.json"
    preflight
    run_stage "S3H_C_STACK_DONE" "부록A harden c_stack (T-CAUS)" \
      "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG" --arm c_stack \
        --adapter "$ADAPT/c_stack/adapter" --n "$IV_N"
  else
    echo "[cesft_v2] 부록A SKIP — P-UTIL=$PUTIL (복리 전제 미성립, C-stack 보류)" | tee -a "$LOG/chain.log"
    echo "{\"skipped\": \"P-UTIL=$PUTIL\"}" > "$MK/S_CE_C_STACK_DONE"
    echo "{\"skipped\": \"P-UTIL=$PUTIL\"}" > "$MK/S_CE_C_CTRL_DONE"
    echo "{\"skipped\": \"P-UTIL=$PUTIL\"}" > "$MK/S7_EVAL_C_CTRL_DONE"
  fi
fi

bake
echo "{\"ts\": $(date +%s)}" > "$MK/CESFT_V2_CHAIN_DONE"
echo "{\"ts\": $(date +%s)}" > "$MK/RETRO3_CHAIN_DONE"   # supervisor.sh 정지 신호 (공유 러너 재사용)
echo "[cesft_v2] 전체 완료"
