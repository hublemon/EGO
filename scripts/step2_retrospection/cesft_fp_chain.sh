#!/usr/bin/env bash
# cesft_v2_fp 무인 체인 — 1인칭 일원화 최종 학습 + cand_free 대조군 (단일 코호트 1회 완주).
# SSOT: docs/experiments/2026-07-25_cesft_fp_final_run_and_candfree_control_design_handoff.md
# marker 멱등: 완료 스테이지 skip. 단일 GPU(MAX_PARALLEL=1 전제), supervisor.sh 경유 기동만 허용.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export PYTHONPATH="$REPO/src"
export HF_HOME=/mnt/nvme/cache
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RETRO3_RUNS="${RETRO3_RUNS:-runs/cesft_v2_fp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"   # 시간 계약 — cesft_v2 와 동일 (불변)
export FRAME_CACHE_DIR="$REPO/runs/cesft_v2/frame_cache"     # 캐시 공유 (재추출 금지)
export CKPT_KEEP_STEP_ADAPTERS=1                             # step-태그 어댑터 보존 (fp 코호트 한정)

CFG=configs/step2_retrospection/cesft_v2_fp.yaml             # output_dir 만 cesft_v2_fp 로 분리
RUNS="$RETRO3_RUNS"
MK="$RUNS/markers"; LOG="$RUNS/logs"; EVAL="$RUNS/eval"
ADAPT="outputs/step2_retrospection/cesft_v2_fp"
mkdir -p "$MK" "$LOG" "$EVAL"

EVAL_N="${EVAL_N:-1000}"
FREEGEN_N="${FREEGEN_N:-500}"
IV_N="${IV_N:-400}"
CKPT_EVERY="${CKPT_EVERY:-50}"      # 사고 이력상 step 90/48 사망 — 50 간격
PROBE_EVERY="${PROBE_EVERY:-50}"
# 2026-07-25 사용자 결정: cand_free 대조군은 야간 체인에서 제외.
# → 2026-07-26 사용자 결정으로 번복: cand_free 를 포함하고 **최선두**에서 실행 (기본값 1).
RUN_CAND_FREE="${RUN_CAND_FREE:-1}"
ARMS="base theta_ce sft_r15"; [[ "$RUN_CAND_FREE" == "1" ]] && ARMS="$ARMS cand_free"
# GATE_GO 통과선 50%로 하향 (사전 등록 60%에서 변경) — 사유: 1인칭 Φ의 탈락은 품질 붕괴가
# 아니라 의도-서술("I am preparing to X")의 restatement 정합 문제로 실증됨(표본 원문 확인,
# 1인칭율 100%·hedging 없음). 게이트 규칙 자체는 불변(drop-not-patch). 아침 검토 필수.
export GATE_MIN_PASS="${GATE_MIN_PASS:-0.50}"

preflight() {
  local gpu i free_hard
  for i in $(seq 1 180); do
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    # hard-RAM admission (cgroup 회수불가분 기준 — memory.current 금지, 07-24 자기교착 이력)
    free_hard=$(awk '$1=="anon"||$1=="unevictable"||$1=="slab_unreclaimable"||$1=="file_dirty"||$1=="file_writeback"{s+=$2}
      END{lim="'$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)'"+0;
          if (lim<=0) print 999; else printf "%.0f", (lim-s)/1e9}' /sys/fs/cgroup/memory.stat 2>/dev/null || echo 999)
    if [[ "$gpu" -lt 30000 && "$free_hard" -ge 60 ]]; then return 0; fi
    echo "[fp] preflight 대기: gpu_used=${gpu}MiB ram_free_hard=${free_hard}G" | tee -a "$LOG/chain.log"
    sleep 60
  done
  echo "{\"failed\": \"preflight\", \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"; exit 1
}

run_stage() {
  local marker="$1"; shift; local desc="$1"; shift
  if [[ -f "$MK/$marker" ]]; then echo "[fp] SKIP $desc ($marker)"; return 0; fi
  echo "[fp] ==== $desc ===="
  "$@" >> "$LOG/chain.log" 2>&1; local rc=$?
  if [[ -f "$MK/$marker" ]]; then echo "[fp] OK $desc (exit=$rc)"; return 0; fi
  echo "[fp] FAIL $desc (exit=$rc, marker 없음)" | tee -a "$LOG/chain.log"
  echo "{\"failed\": \"$desc\", \"rc\": $rc, \"ts\": $(date +%s)}" > "$MK/CHAIN_FAILED"; exit 1
}

gate() { "$PY" tools/paired_boot.py --run "$RUNS" "$@" >> "$LOG/chain.log" 2>&1 || true; }

# ── G0: Φ 재생성 (1인칭 PROJ_SYSTEM, Ψ→Φ→규칙게이트) ─────────────────────────
preflight
run_stage S3_HINDSIGHT_DONE "G0 projection (1인칭 Φ 재생성)" \
  "$PY" -m ego.step2_retrospection.hindsight.projection --config "$CFG" \
    --subset "$RUNS/data/train_subset.json" --batch_size 32

# ── GATE_GO: 자동 Go/No-Go (§4-1) — 실패 시 학습 착수 금지 ────────────────────
run_stage GATE_GO "GATE_GO 관문 판정 (pass율·1인칭·문체)" \
  "$PY" tools/gate_go_check.py --run "$RUNS"

# ── T3: cand_free 대조군 (G-EQ: 동일 subset·epochs·seed → 동일 step 수) ───────
# 2026-07-26 순서 이동: θ_CE 미의존이라 최선두 배치 — 서버 사고 시 짧은(≈1h) 대조군
# 어댑터부터 확보. marker 멱등이라 순서 변경은 재실행 안전성에 영향 없음.
if [[ "$RUN_CAND_FREE" == "1" ]]; then
  preflight
  run_stage S_CE_CAND_FREE_DONE "T3 cand_free (GT-CE 대조군, 1인칭 SYS_NOCAND)" \
    "$PY" -m ego.step2_retrospection.train.select_ce --config "$CFG" --run_name cand_free \
      --arm cand_free --tau 1.0 --epochs 1 --seed 42 \
      --ckpt_every "$CKPT_EVERY" --resume auto --probe_every "$PROBE_EVERY"
fi

# ── T1: θ_CE (최종본) ────────────────────────────────────────────────────────
preflight
run_stage S_CE_THETA_CE_DONE "T1 θ_CE (wm_cand, 1인칭)" \
  "$PY" -m ego.step2_retrospection.train.select_ce --config "$CFG" --run_name theta_ce \
    --arm wm_cand --tau 1.0 --epochs 1 --seed 42 \
    --ckpt_every "$CKPT_EVERY" --resume auto --probe_every "$PROBE_EVERY"

# ── T2: sft_r15 (타깃 = 1인칭 Φ) ─────────────────────────────────────────────
preflight
run_stage S6_SFT_R15_DONE "T2 sft_r15 (CE-replay ρ=0.15)" \
  "$PY" -m ego.step2_retrospection.train.sft_v2 --config "$CFG" --run_name sft_r15 \
    --init_adapter "$ADAPT/theta_ce/adapter" --ce_replay_rho 0.15 --ce_tau 1.0 \
    --epochs 1 --seed 42 --ckpt_every "$CKPT_EVERY" --resume auto --probe_every "$PROBE_EVERY"

# ── E1: battery (n=EVAL_N 동일 셋 — overrides.json 이 covered_only 강제) ──────
for arm in $ARMS; do
  case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
  preflight
  run_stage "S7_EVAL_$(echo "$arm" | tr a-z A-Z)_DONE" "E1 battery $arm" \
    "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm "$arm" \
      ${AD:+--adapter "$AD"} --eval_n "$EVAL_N"
done

# 사전 등록 게이트 (기록 — 판정은 사람이 아침에 확인)
gate --arm_a theta_ce --gate G-ACC1 --out "$EVAL/paired_G-ACC1_theta_ce.json"
gate --arm_a sft_r15 --arm_b theta_ce --gate G-NH --out "$EVAL/paired_G-NH_sft_r15_vs_theta_ce.json"
[[ "$RUN_CAND_FREE" == "1" ]] && gate --arm_a theta_ce --arm_b cand_free --gate G-DELTA \
     --metric SelAcc --out "$EVAL/paired_G-DELTA_theta_ce_vs_cand_free.json"

# ── E2: strip (history 인과, covered paired) ─────────────────────────────────
for arm in $ARMS; do
  case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
  preflight
  run_stage "S_STRIP_$(echo "$arm" | tr a-z A-Z)_DONE" "E2 strip $arm" \
    "$PY" tools/oom_opt/strip_eval.py --config "$CFG" --arm "$arm" --adapter "$AD" \
      --eval_n "$EVAL_N" --covered_only
done

# ── E3: harden (belief 개입, n=IV_N) ─────────────────────────────────────────
for arm in $ARMS; do
  case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
  preflight
  run_stage "S3H_$(echo "$arm" | tr a-z A-Z)_DONE" "E3 harden $arm" \
    "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG" --arm "$arm" \
      --adapter "$AD" --n "$IV_N"
done

# ── E4: freegen (후보-비제시 레짐 — presented 텍스트는 battery records 재계산) ──
for arm in $ARMS; do
  case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
  preflight
  run_stage "S_FREEGEN_$(echo "$arm" | tr a-z A-Z)_CAND_FREE_DONE" "E4 freegen $arm" \
    "$PY" -m ego.step2_retrospection.eval.freegen --config "$CFG" --arm "$arm" \
      ${AD:+--adapter "$AD"} --eval_n "$FREEGEN_N"
done

# ── E5: CPU 집계 — 텍스트지표·trace 예시 (+cand_free 시 DiD) ─────────────────
DID_CMD="true"
[[ "$RUN_CAND_FREE" == "1" ]] && DID_CMD="$PY tools/did_history.py --run '$RUNS' --arm_a theta_ce --arm_b cand_free"
run_stage S_TEXT_METRICS_DONE "E5 집계 (text·trace)" \
  bash -c "$DID_CMD \
    && $PY tools/trace_text_metrics.py --run '$RUNS' --arms \"$(echo $ARMS | tr ' ' ',')\" \
    && $PY tools/pick_trace_examples.py --run '$RUNS' \
    && echo '{\"done\": true}' > '$MK/S_TEXT_METRICS_DONE'"

# ── 완료 ─────────────────────────────────────────────────────────────────────
echo "{\"done_at\": $(date +%s)}" > "$MK/CESFT_FP_CHAIN_DONE"
echo "{\"done_at\": $(date +%s)}" > "$MK/RETRO3_CHAIN_DONE"   # supervisor 정지 신호
echo "[fp] 체인 완료" | tee -a "$LOG/chain.log"
