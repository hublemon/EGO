#!/usr/bin/env bash
# step2_candidate_ce_chain.sh — 2단계: candidate CE 완주 (H200 1장, GPU cuda:1)
#
# 실행 (ssh/VS Code 세션 독립):
#   cd /mnt/nvme/migration/jihun/EGO_jihun
#   setsid nohup bash scripts/step2/step2_candidate_ce_chain.sh \
#     > /mnt/nvme/migration/jihun/EGO/runs/cce/launch.log 2>&1 < /dev/null &
#
# 현황: http://<host>:7865/   (tools/cce_dashboard.py)
#
# ── 왜 이걸 돌리는가 ──────────────────────────────────────────────────────────
# 실측 세 개가 한 그림을 만든다:
#   base 모델의 후보 스코어링 능력   0.3876   (teacher_headroom, train 표본 n=1,370)
#   L0 = WM top-1 그냥 따르기        0.7657   (동일 샘플)
#   버려진 f0_gx 첫 로그             0.395 → 0.545  (37 optimizer step 만에)
# f0_gx 의 시작점이 base 스코어링과 정확히 일치하고, 37 step 만에 크게 올랐는데
# 그 run 은 800샘플에서 버려졌다 (checkpoint 없음). 회수 대상이 0.22 로,
# Retrospection 이 줄 수 있는 최대치(+0.044)보다 5배 크다.
#
# 완주한 모든 학습이 텍스트 생성을 최적화했고 action 은 부산물이었다. 평가 지표를
# **직접** 겨냥한 objective 는 이 프로젝트에서 한 번도 완주된 적이 없다.
#
# ── 사전 등록 판정 ────────────────────────────────────────────────────────────
#   성공 = heldout 전체 정확도가 L0(0.3994)를 **초과**
#   부분 = L0 미달이나 조건부 정확도가 base(0.388) 대비 유의하게 상승
#   실패 = 조건부 정확도가 base 수준에 머묾 → 후보 판별 정보 자체가 부족
# 실행 중에 기준을 바꾸지 않는다.
set -u

EGO_ROOT=/mnt/nvme/migration/jihun/EGO
REPO=/mnt/nvme/migration/jihun/EGO_jihun
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
OUT=$EGO_ROOT/runs/cce
TR=$EGO_ROOT/runs/f0_battery/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
HO=$EGO_ROOT/runs/f0_battery/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
ADAPT=$REPO/outputs/step2/cce_1f
DEV=cuda:1
LOG=$OUT/chain.log
mkdir -p "$OUT"
cd "$REPO" || exit 1

say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
die() { say "치명적: $*"; touch "$OUT/CHAIN_FAILED"; exit 1; }

MAXS=${MAXS:-5000}      # ≈ oracle subset(4,618) 1 에폭
LR=${LR:-1e-5}
ACCUM=${ACCUM:-16}

say "===== 2단계 candidate CE 체인 시작 (GPU $DEV · max_samples=$MAXS) ====="

# ── 1) 스모크는 제거했다 ────────────────────────────────────────────────────
# 1차 시도에서 스모크 게이트가 학습을 중단시켰는데 **게이트가 틀렸다**:
#   (a) 판정식이 loss[-1] < loss[0] 였다. 6개 노이즈 점에서 마지막 한 점이 결과를
#       뒤집는 추정량으로, 같은 세션에서 cw 스윕을 두고 내가 직접 비판한 그 방식이다.
#   (b) 더 결정적으로 300샘플 = 18 optimizer step 인데, 참조 run(f0_gx, 07-19)의
#       loss 급락은 seen 400~600 = 37 step 에서 일어났다. 볼 수 있는 지점 전에 끊었다.
#         f0_gx:  seen 200 loss 6.22 / 400 4.95 / 600 1.79 / 800 1.48
#         스모크: seen 300 까지 loss 4.4~6.9 진동 (급락 구간 미도달)
# 실측 처리율이 0.75 s/샘플 → 본실행 5,000 이 63분뿐이다. 6시간짜리를 지키려던 장치를
# 1시간짜리에 붙일 이유가 없다. 본실행 로그가 seen 600 에서 같은 신호를 준다.
# 300샘플 스모크 결과는 $OUT/smoke 에 진단 기록으로 남긴다.

# ── 2) 본실행 ────────────────────────────────────────────────────────────────
if [ ! -d "$ADAPT/checkpoint-final" ]; then
  say "--- 본실행 ($MAXS 샘플, 약 1.5h) ---"
  $PY scripts/step2/pro_gx_train.py --train_jsonl "$TR" --output_dir "$ADAPT" \
    --max_samples "$MAXS" --accum "$ACCUM" --lr "$LR" --device $DEV \
    --log_every 100 --save_every 1000 > "$OUT/train.log" 2>&1
  say "  본실행 rc=$?"
fi
[ -d "$ADAPT/checkpoint-final" ] || die "본실행이 체크포인트를 남기지 않았다."

# ── 3) 후보 스코어링 평가 (학습 목적함수와 동일 경로) ────────────────────────
if [ ! -s "$OUT/eval_scored.json" ]; then
  say "--- 후보 스코어링 평가 (heldout 전량) ---"
  $PY scripts/step2/eval_candidate_scored.py --jsonl "$HO" --adapter "$ADAPT/checkpoint-final" \
    --limit 0 --action_only --device $DEV --out "$OUT/eval_scored.json" \
    > "$OUT/eval_scored.log" 2>&1
  say "  스코어링 평가 rc=$?"
fi

# ── 4) 대조: 학습 전 base 의 스코어링 (같은 경로) ───────────────────────────
if [ ! -s "$OUT/eval_scored_base.json" ]; then
  say "--- 대조군: base 모델 스코어링 ---"
  $PY scripts/step2/eval_candidate_scored.py --jsonl "$HO" \
    --limit 0 --action_only --device $DEV --out "$OUT/eval_scored_base.json" \
    > "$OUT/eval_scored_base.log" 2>&1
  say "  base 스코어링 rc=$?"
fi

# ── 5) 생성 평가 (기존 arm 들과 비교 가능하게) ───────────────────────────────
if [ ! -s "$OUT/eval_gen.json" ]; then
  say "--- 생성 평가 (기존 arm 대조용) ---"
  $PY scripts/step2/eval_harness_v2.py --jsonl "$HO" --adapter "$ADAPT/checkpoint-final" \
    --limit 0 --n_boot 10000 --action_only --device $DEV --out "$OUT/eval_gen.json" \
    > "$OUT/eval_gen.log" 2>&1
  say "  생성 평가 rc=$?"
fi

# ── 6) G1/G2 분해 + 사전 등록 판정 ──────────────────────────────────────────
say "--- G1/G2 분해 ---"
$PY scripts/step2/decompose_g1g2.py --jsonl "$HO" \
  --records "$OUT/eval_scored.records.jsonl" "$OUT/eval_scored_base.records.jsonl" \
  --out "$OUT/decompose.json" 2>&1 | tee -a "$LOG"

$PY - "$OUT/eval_scored.json" "$OUT/eval_scored_base.json" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
try:
    a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
except Exception as e:
    print(f"  판정 불가: {e}"); raise SystemExit(0)
L0=a["L0_wm_top1"]
print(f"  L0 (WM top-1 무학습)      {L0}")
print(f"  base 스코어링             acc {b['acc']}  조건부 {b['conditional_acc']}")
print(f"  학습 후 스코어링           acc {a['acc']}  조건부 {a['conditional_acc']}")
print(f"  G1 보존 {b['g1_retention']} -> {a['g1_retention']}   "
      f"G2 교정 {b['g2_correction']} -> {a['g2_correction']}")
if a["acc"] > L0:
    print("  판정: 성공 — L0 초과. 이 프로젝트에서 처음이다.")
elif a["conditional_acc"] > b["conditional_acc"]:
    print("  판정: 부분 성공 — L0 미달이나 조건부 정확도가 base 대비 상승.")
else:
    print("  판정: 실패 — 후보 판별력이 base 수준에 머묾.")
PY

say "===== 2단계 종료 ====="
touch "$OUT/CHAIN_DONE"
