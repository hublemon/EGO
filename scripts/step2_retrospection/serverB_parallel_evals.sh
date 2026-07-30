#!/usr/bin/env bash
# 서버 B 병렬 평가 — cesft_v2_fp 8시간 완주 분업 (2026-07-26 KST 13:10 사용자 지시).
#
# 분업 계약 (서버 A = 학습 임계경로, 서버 B = 이 스크립트):
#   A: T3 cand_free → T1 θ_CE → T2 sft_r15 → [E1 battery r15 → E2 strip r15] → 게이트
#   B: base 4종 평가 → cand_free 4종 평가 → θ_CE 4종 평가 + G-ACC1
#      → (T2 완료 대기) → sft_r15 의 E3 harden + E4 freegen 만   ← battery/strip 은 A 몫
# marker 멱등이 유일한 조정 장치다. B 가 먼저 끝내면 A 체인이 SKIP 하고, 그 역도 같다.
# sft_r15 에서 B 가 E3/E4 만 맡는 이유: A 는 E1→E2 순서로 ~35분 걸리므로 A 가 E3 에
# 도달하기 전에 B 의 marker 가 먼저 찍혀 중복 실행 경쟁이 구조적으로 발생하지 않는다.
#
# 전제: 2026-07-26 03:53 OOM 수정(vlm cache-first + close_readers) 적용 커밋 상태.
# 기동(서버 B 에서):
#   cd /mnt/nvme/migration/jihun/EGO_jihun3
#   setsid nohup bash scripts/step2_retrospection/serverB_parallel_evals.sh \
#     >> runs/cesft_v2_fp/logs/serverB_evals.log 2>&1 < /dev/null & disown
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON_BIN:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
export PYTHONPATH="$REPO/src"
export HF_HOME=/mnt/nvme/cache
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RETRO3_RUNS="runs/cesft_v2_fp"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"
export FRAME_CACHE_DIR="$REPO/runs/cesft_v2/frame_cache"

CFG=configs/step2_retrospection/cesft_v2_fp.yaml
RUNS="$RETRO3_RUNS"; MK="$RUNS/markers"; LOG="$RUNS/logs"; EVAL="$RUNS/eval"
ADAPT="outputs/step2_retrospection/cesft_v2_fp"
EVAL_N="${EVAL_N:-1000}"; FREEGEN_N="${FREEGEN_N:-500}"; IV_N="${IV_N:-400}"

say(){ echo "[B-evals $(date '+%F %T')] $*"; }

# GPU 선점 확인 — B 에서 다른 작업이 돌고 있으면 대기 (30분 한도)
for i in $(seq 1 30); do
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [[ "$g" -lt 30000 ]] && break
  say "GPU 대기: used=${g}MiB"; sleep 60
done

run_stage(){ local marker="$1"; shift; local desc="$1"; shift
  if [[ -f "$MK/$marker" ]]; then say "SKIP $desc ($marker)"; return 0; fi
  say "==== $desc ===="
  "$@"; local rc=$?
  if [[ -f "$MK/$marker" ]]; then say "OK $desc"; return 0; fi
  # 실패해도 다음 스테이지 진행 (A 체인이 나중에 재시도할 수 있게 marker 만 안 남긴다).
  # 단, stale status 가 A 의 stall watchdog 을 오발시키지 않도록 running 상태를 지운다.
  say "FAIL $desc (rc=$rc) — 계속 진행"
  "$PY" - <<PYEOF
import glob, json
for f in glob.glob("$RUNS/status/*.json"):
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("state") == "running" and "$desc" and d.get("stage","").startswith(("S7_","S_STRIP","S3H","S_freegen","S_FREEGEN")):
        d["state"] = "failed"; json.dump(d, open(f, "w"))
PYEOF
  return 0; }

wait_marker(){ local m="$1" limit_min="$2"
  say "대기: $m (최대 ${limit_min}분)"
  for i in $(seq 1 "$limit_min"); do [[ -f "$MK/$m" ]] && return 0; sleep 60; done
  say "TIMEOUT: $m — 이후 스테이지 건너뜀"; return 1; }

arm_evals(){ local arm="$1" AD; case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
  local U; U=$(echo "$arm" | tr a-z A-Z)
  run_stage "S7_EVAL_${U}_DONE" "E1 battery $arm" \
    "$PY" -m ego.step2_retrospection.eval.battery --config "$CFG" --arm "$arm" \
      ${AD:+--adapter "$AD"} --eval_n "$EVAL_N"
  run_stage "S_STRIP_${U}_DONE" "E2 strip $arm" \
    "$PY" tools/oom_opt/strip_eval.py --config "$CFG" --arm "$arm" --adapter "$AD" \
      --eval_n "$EVAL_N" --covered_only
  run_stage "S3H_${U}_DONE" "E3 harden $arm" \
    "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG" --arm "$arm" \
      --adapter "$AD" --n "$IV_N"
  run_stage "S_FREEGEN_${U}_CAND_FREE_DONE" "E4 freegen $arm" \
    "$PY" -m ego.step2_retrospection.eval.freegen --config "$CFG" --arm "$arm" \
      ${AD:+--adapter "$AD"} --eval_n "$FREEGEN_N"; }

# ── 1) base — 어댑터 불필요, 즉시 ─────────────────────────────────────────────
arm_evals base

# ── 2) cand_free — T3 완료 대기 (A 에서 ~13:30 KST 완료 예정) ──────────────────
wait_marker S_CE_CAND_FREE_DONE 60 && arm_evals cand_free

# ── 3) θ_CE — T1 완료 대기 (~18:30 KST) + G-ACC1 게이트 ───────────────────────
if wait_marker S_CE_THETA_CE_DONE 420; then
  arm_evals theta_ce
  "$PY" tools/paired_boot.py --run "$RUNS" --arm_a theta_ce --gate G-ACC1 \
    --out "$EVAL/paired_G-ACC1_theta_ce.json" || true
fi

# ── 4) sft_r15 — T2 완료 대기 (~20:30 KST). E3+E4 만 (E1/E2 는 A 몫) ──────────
if wait_marker S6_SFT_R15_DONE 300; then
  AD="$ADAPT/sft_r15/adapter"
  run_stage "S3H_SFT_R15_DONE" "E3 harden sft_r15" \
    "$PY" -m ego.step2_retrospection.eval.harden_s3 --config "$CFG" --arm sft_r15 \
      --adapter "$AD" --n "$IV_N"
  run_stage "S_FREEGEN_SFT_R15_CAND_FREE_DONE" "E4 freegen sft_r15" \
    "$PY" -m ego.step2_retrospection.eval.freegen --config "$CFG" --arm sft_r15 \
      --adapter "$AD" --eval_n "$FREEGEN_N"
fi

say "서버 B 분업 완료"
