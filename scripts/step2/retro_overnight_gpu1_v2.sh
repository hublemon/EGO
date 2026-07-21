#!/usr/bin/env bash
# retro_overnight_gpu1_v2.sh — 무인 retro 재실행 v2 (GPU 1 전용) 2026-07-20 야간
#
# v1 대비 변경 세 가지. 근거는 각 단계 주석에.
#   1) 본실행 보상을 --reward wm → --reward gt 로 바꾼다.        [치명적 결함 수정]
#   2) action+sum 본실행(4.7h)을 버리고 그 자리에 wm 대조를 넣는다. [정보량 재배분]
#   3) 정합적인 arm(gt)을 먼저 돌린다.                            [하나만 완주해도 옳은 쪽]
#
# ── 왜 --reward wm 을 버리는가 ──────────────────────────────────────────────
# pro_gr_train.py:199-207 의 wm 보상은  r = lik(선택 후보) / Σlik(top-5)  이다.
# train pool 4998건 실측:
#     항상 WM top-1 선택   r = 0.5017   ← 이 보상함수의 전역 최적해
#     항상 GT 선택         r = 0.4716
#     무작위               r = 0.2000
#     현재 정책(실측)      r ≈ 0.30     ← 최적해까지 +0.19 여지가 남아 실제로 움직인다
# GT 를 완벽히 맞히는 정책의 보상이 WM top-1 맹목 추종보다 **낮다**. 그리고 GT 가
# rank2~5 인 24.3% 구간 — 정확히 B0/Retrospection 의 영역 — 에서는 보상 최대화가
# GT 와 **반대 방향**으로 움직인다.
# 보상이 GT-free 이고 WM 에서 파생되면 정책은 원리적으로 WM 이 모르는 것을 배울 수 없다.
# heldout 실측이 이를 뒷받침한다: WM top-1 argmax = 0.374 인데 학습된 정책은 전부 0.24~0.28.
#
# --reward gt 는 r∈{0,1} (GT 일치) 로, 최적해가 GT 완벽 정책이다. GT 는 학습 시점
# 검증자로만 쓰이고 추론 시엔 쓰이지 않는다(verifiable-reward RL). 다만 F0 W-EMA
# 라인은 의도적으로 GT-free 였으므로 **이 arm 은 다른 주장**임을 기록에 남긴다.
#
# ── 왜 credit=belief + reduction=sum 인가 (v1 에서 승계, 게이트 A 로 확인됨) ──
#     credit=action · mean   mean|loss| 0.000092   (어제 ARM B — gradient 소멸)
#     credit=action · sum    mean|loss| 0.002800   ← mean→sum 효과 30배
#     credit=all    · mean   mean|loss| 0.004160   (참고)
#     credit=belief · sum    mean|loss| 0.034000   ← action→belief 효과 12배 · 총 370배
# <action> span 은 거의 결정적인 JSON 토큰 몇 개라 토큰당 logp ≈ 0 이다(독립 확인:
# best-of-8 스코어링에서 action span 평균 logp = -0.0000). belief 는 고엔트로피
# 자유 텍스트라 gradient 가 살아 있다. span margin 도 belief 가 전부 가져갔다(+0.40 vs -0.005).
#
# 각 단계는 마커 파일로 재개 가능하다. 실패하면 멈추고 다음 단계로 넘어가지 않는다.

set -uo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"; PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

BAT=$EGO_ROOT/runs/f0_battery
OUT=$EGO_ROOT/runs/retro_overnight; mkdir -p "$OUT"
LOG=$OUT/chain.log
TR1F=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
DEV=cuda:1

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
halt(){ say "✗ $*"; touch "$OUT/FAILED"; exit 1; }

meanabs(){ $PY - "$1" <<'PY'
import json,sys
try:
    v=[abs(json.loads(l)["loss"]) for l in open(sys.argv[1])]
    print(f"{sum(v)/len(v):.6f}" if v else "0")
except Exception:
    print("0")
PY
}

say "===== retro 무인 체인 v2 시작 (GPU 1) ====="
[ -s "$TR1F" ] || halt "train jsonl 없음: $TR1F"
[ -s "$J1F" ]  || halt "heldout jsonl 없음: $J1F"

# ── 게이트 A' : gt 보상 + sum 이 안정적인가 ─────────────────────────────────
# 왜 다시 재는가: 바이너리 보상은 advantage 가 크다. r∈{0,1}, baseline≈pass@1≈0.39
#   → mean|adv| ≈ 0.39·0.61 + 0.61·0.39 = 0.476   (wm 의 실측 0.20 대비 2.4배)
# 여기에 sum(이미 370배 증폭)이 곱해진다. 3.8h 앞에 두는 14분 보험이다.
# 하한: credit=action·mean 의 0.000092 대비 10배 = 0.00092  (v1 의 GATE_MIN 과 동일)
# 상한: 발산 감시. mean|loss| > 5.0 이면 LR/정규화 재검토가 필요하다는 신호로 기록만 한다.
GATE_MIN=0.00092
GATE_WARN=5.0
SM=$OUT/smoke_belief_sum_gt
if [ ! -f "$SM/gr_log.jsonl" ]; then
  say "--- 게이트 A': credit=belief · reduction=sum · reward=gt 스모크 (300샘플) ---"
  rm -rf "$SM"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$SM" \
    --full_trace --reward gt --credit belief --credit-reduction sum \
    --max_new_tokens 384 --batch_gen 4 --max_samples 300 --accum 16 \
    --log_every 50 --save_every 100000 --device $DEV \
    > "$OUT/smoke_belief_sum_gt.log" 2>&1
  say "게이트 A' 스모크 rc=$?"
fi
GM=$(meanabs "$SM/gr_log.jsonl")
say "  mean|loss| (belief, sum, gt) = $GM   [하한 $GATE_MIN · 경고선 $GATE_WARN]"
echo "belief_sum_gt $GM" >> "$OUT/gateA.txt"
PASS=$($PY -c "print(1 if float('$GM') >= float('$GATE_MIN') else 0)")
if [ "$PASS" != "1" ]; then
  say "게이트 A' 실패 — gt 보상에서도 gradient 가 죽어 있다 (mean|loss|=$GM)."
  say "REINFORCE 계열을 폐기하고 남은 시간을 쓰지 않는다. 다음 세션에서"
  say "개선 2(belief-swap consistency) 또는 projection 대조(B1/B2/B3)로 자원을 전량 이전할 것."
  touch "$OUT/GATE_A2_FAILED"; exit 0
fi
$PY -c "
import sys
print('  ⚠ 경고: mean|loss| 이 경고선을 넘었다 — 분산 폭주 가능. reward_ma 추세를 반드시 확인할 것.'
      if float('$GM') > float('$GATE_WARN') else '  안정 구간.')"| tee -a "$LOG"
say "게이트 A' 통과 — 본실행으로 진행"
touch "$OUT/GATE_A2_PASSED"

# ── 공통 평가 함수 ──────────────────────────────────────────────────────────
# eval_harness_v2: heldout 전량(1417) + 2 disjoint subset + bootstrap CI + MDE.
#   n=500 → 1417 로 MDE 가 0.044 → 0.026 으로 내려가고, G2 부분집합도 123 → 약 348 이 된다.
# eval_belief_swap: ③ 인과민감도(swap − control).
run_eval(){   # $1=tag  $2=adapter
  local TAG=$1 AD=$2
  if [ ! -s "$OUT/eval_${TAG}.json" ]; then
    say "--- 평가($TAG): 생성 acc · CI · MDE (heldout 전량) ---"
    $PY scripts/step2/eval_harness_v2.py --jsonl "$J1F" --adapter "$AD" \
      --limit 0 --device $DEV --out "$OUT/eval_${TAG}.json" \
      > "$OUT/eval_${TAG}.log" 2>&1
    say "  생성 평가 rc=$?"
  fi
  if [ ! -s "$OUT/swap_${TAG}.json" ]; then
    say "--- 평가($TAG): ③ belief-swap 인과민감도 ---"
    $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" --adapter "$AD" \
      --limit 0 --device $DEV --out "$OUT/swap_${TAG}.json" \
      > "$OUT/swap_${TAG}.log" 2>&1
    say "  개입 평가 rc=$?"
  fi
}

# ── 본실행 1 : credit=belief + sum + reward=gt  (정합 arm, 먼저 돌린다) ──────
# 목표: "보상의 최적해가 GT 인 조건에서, 살아난 belief gradient 가 실제로
#        생성 정확도를 올리는가"를 본다.
# 판정(사전등록):
#   성공 = acc 상승이 MDE 를 넘고  AND  wm_follow 가 비례 상승하지 않고
#          AND  G2 acc(n≈348)가 CI 밖으로 상승
#   실패해석 = wm_follow 만 뛰고 G2 가 안 움직이면 retrospection 이 아니라 WM 추종이다
BOUT=$REPO/outputs/step2/retro_belief_sum_gt_1f
if [ ! -f "$BOUT/TRAINING_DONE" ]; then
  say "--- 본실행 1/2: credit=belief · sum · reward=gt (5000샘플, 약 3.8h) ---"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$BOUT" \
    --full_trace --reward gt --credit belief --credit-reduction sum \
    --max_new_tokens 384 --batch_gen 4 --max_samples 5000 --accum 16 \
    --save_every 1250 --device $DEV > "$OUT/train_belief_sum_gt.log" 2>&1
  RC=$?; say "본실행 1 rc=$RC"
  [ $RC -eq 0 ] && touch "$BOUT/TRAINING_DONE" || halt "본실행 1 실패 — $OUT/train_belief_sum_gt.log"
fi
AD=$BOUT/checkpoint-final; [ -d "$AD" ] || AD=$BOUT
run_eval "belief_sum_gt" "$AD"

# ── 본실행 2 : credit=belief + sum + reward=wm  (보상 대조) ─────────────────
# 목표: 1 과 **보상만** 다르다. 두 결과의 차이가 곧 "보상 정의가 결과를 만들었는가"의 답.
# 예측: wm arm 은 acc 가 0.374(WM argmax) 쪽으로 오르되 wm_follow 가 함께 뛰고
#       G2 는 움직이지 않는다. 그 패턴이 나오면 v1 계획이 왜 위험했는지가 실증된다.
WOUT=$REPO/outputs/step2/retro_belief_sum_wm_1f
if [ ! -f "$WOUT/TRAINING_DONE" ]; then
  say "--- 본실행 2/2: credit=belief · sum · reward=wm (5000샘플, 약 3.8h) ---"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$WOUT" \
    --full_trace --reward wm --credit belief --credit-reduction sum \
    --max_new_tokens 384 --batch_gen 4 --max_samples 5000 --accum 16 \
    --save_every 1250 --device $DEV > "$OUT/train_belief_sum_wm.log" 2>&1
  RC=$?; say "본실행 2 rc=$RC"
  [ $RC -eq 0 ] && touch "$WOUT/TRAINING_DONE" || say "본실행 2 실패(비치명) — $OUT/train_belief_sum_wm.log"
fi
if [ -f "$WOUT/TRAINING_DONE" ]; then
  AD2=$WOUT/checkpoint-final; [ -d "$AD2" ] || AD2=$WOUT
  run_eval "belief_sum_wm" "$AD2"
fi

touch "$OUT/DONE"
say "===== retro 무인 체인 v2 종료 ====="
