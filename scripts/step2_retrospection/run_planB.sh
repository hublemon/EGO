#!/usr/bin/env bash
# Plan B — harden_paired 4-arm. belief 주장의 교란 통제 재측정 + arm 간 paired CI + G_CC2 정식 판정.
#
# 왜 필요한가 (harden_s3 의 3대 교란, harden_paired.py 헤더):
#   (1) arm 마다 표본 셋이 다르다 — 실측 겹침 27% (theta_ce ∩ sft_r15_c = 170/400)
#   (2) swap partner 가 arm 내부에서 온다 — belief 문체가 arm 마다 달라 주입 정보량이 다르다
#   (3) paraphrase 를 평가 대상 모델 자신이 생성 — arm 마다 다른 잣대
# → 공통 셋 1개(4-arm 공통 901건 중 300) · 전 arm 동일 donor 문자열(swap_b_shared) ·
#   base 모델 1회 생성 paraphrase 공유. 프레임 포함 vision-grounded 채점.
#
# 최종 arm 확정 근거 (2026-07-27 01:5x): rho=0.30 이 모든 축에서 rho=0.15 이하.
#   belief 0.3700 vs 0.3825 / acc_own 0.2125 vs 0.2600 / SelAcc 0.2760 vs 0.2850
#   / GADR 0.2348 vs 0.2427 / H8 이력인과 +9.18 vs +10.30 / malformed 4.5% vs 2.8%
#
# plan 은 확장 불가다(harden_paired.py:190-196 공통셋 = plan 시점 arms 의 교집합, :240 이
# plan 에 없는 arm 을 차단). 나중에 arm 을 추가하려면 4개 전부 재실행이다.
#
# 멱등: 각 스테이지 산출물이 있으면 건너뛴다. 중단 후 같은 스크립트를 다시 걸면 이어서 간다.
set -uo pipefail
cd /mnt/nvme/migration/jihun/EGO_jihun3
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH=/mnt/nvme/migration/jihun/EGO_jihun3/src
export RETRO3_RUNS=runs/cesft_v2_fp_c
export HF_HOME=/mnt/nvme/cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"       # 시간 계약 — 불변
export FRAME_CACHE_DIR=/mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2/frame_cache
CFG=configs/step2_retrospection/cesft_v2_fp.yaml
ADAPT=outputs/step2_retrospection/cesft_v2_fp
R=runs/cesft_v2_fp_c
EV=$R/eval
ARMS=base,theta_ce,sft_r15,sft_r15_c
REF=sft_r15_c
DONOR=base
N=300
say(){ echo "[PB $(date '+%F %T')] $*"; }

# ── GPU 선점 대기 — 앞선 r30 체인이 끝날 때까지 (최대 40분) ───────────────────
for i in $(seq 1 80); do
  pgrep -f run_r30.sh >/dev/null || break
  [ "$i" = 1 ] && say "r30 체인 진행 중 — GPU 반납 대기"
  sleep 30
done
pgrep -f run_r30.sh >/dev/null && { say "!! r30 이 40분 넘게 안 끝남 — 중단"; exit 5; }
say "GPU 확보 (r30 종료 확인)"

# ── stage: plan ───────────────────────────────────────────────────────────────
if [ -f "$EV/harden_paired_plan.json" ]; then
  say "plan 이미 존재 — SKIP"
else
  say "==== plan (arms=$ARMS, n=$N, donor=$DONOR) ===="
  $PY tools/harden_paired.py --stage plan --arms "$ARMS" --n "$N" --donor "$DONOR" \
      --config "$CFG" >> "$R/logs/planB.log" 2>&1
  rc=$?; say "plan exit=$rc"
  [ -f "$EV/harden_paired_plan.json" ] || { say "!! plan 산출물 없음 — 중단"; exit 4; }
  $PY -c "
import json; d=json.load(open('$EV/harden_paired_plan.json'))
print(f\"[PB] plan 확정: n={d['n']} arms={d['arms']} donor={d['donor_arm']}\")" | tee -a "$R/logs/run_planB.log"
fi

# ── stage: run (arm 마다 별도 프로세스 — GPU 메모리 반납) ─────────────────────
run_arm(){
  local arm="$1" ad="$2"
  if [ -f "$EV/$arm.harden_paired.json" ]; then say "run:$arm 이미 존재 — SKIP"; return 0; fi
  say "==== run:$arm ===="
  if [ -z "$ad" ]; then
    $PY tools/harden_paired.py --stage run --arm "$arm" --config "$CFG" >> "$R/logs/planB.log" 2>&1
  else
    $PY tools/harden_paired.py --stage run --arm "$arm" --adapter "$ad" --config "$CFG" >> "$R/logs/planB.log" 2>&1
  fi
  local rc=$?
  say "run:$arm exit=$rc"
  [ -f "$EV/$arm.harden_paired.json" ] || { say "!! run:$arm 산출물 없음"; return 1; }
  $PY -c "
import json; d=json.load(open('$EV/$arm.harden_paired.json'))
s=d['sensitivity_ci'] if 'sensitivity_ci' in d else d.get('causal_sensitivity_ci',{})
print(f\"[PB] $arm: \"+json.dumps({k:(v.get('point') if isinstance(v,dict) else v) for k,v in s.items()},ensure_ascii=False))" 2>/dev/null | tee -a "$R/logs/run_planB.log"
  return 0
}

run_arm base       ""
run_arm theta_ce   "$ADAPT/theta_ce/adapter"
run_arm sft_r15    "$ADAPT/sft_r15/adapter"
run_arm sft_r15_c  "$ADAPT/sft_r15_c/adapter"

# ── stage: agg ────────────────────────────────────────────────────────────────
say "==== agg (ref=$REF) ===="
$PY tools/harden_paired.py --stage agg --arms "$ARMS" --ref "$REF" --config "$CFG" \
    >> "$R/logs/planB.log" 2>&1
say "agg exit=$?"
if [ -f "$EV/harden_paired_summary.json" ]; then
  $PY -c "
import json; d=json.load(open('$EV/harden_paired_summary.json'))
print(json.dumps({k:d[k] for k in ('n','arms','ref','paired_diff_vs_ref','G_CC2') if k in d},
                 ensure_ascii=False, indent=1))" | tee -a "$R/logs/run_planB.log"
fi
say "Plan B 완료"
