#!/usr/bin/env bash
# retro_overnight_gpu1.sh — 무인 9시간 retro 재실행 (GPU 1 전용) 2026-07-20 야간
#
# 오늘 확정된 것: "credit 을 action span 에 국소화하면 belief->action 인과가 생긴다"는
# offline(ARM A)·online(ARM B) 양쪽에서 기각됐다. 다만 ARM B 는 가설이 틀린 것인지
# 구현이 신호를 죽인 것인지 **분리되지 않았다**:
#
#     loss = -(adv * tok_lp.mean())   ...   pro_gr_train.py
#
# <action> span 은 {"verb": "put", "noun": "lid"} 수준의 거의 결정적인 JSON 토큰 몇 개다.
# logp≈0 인데 .mean() 이 span 길이로 또 나눈다. 실측:
#     credit=all     mean|loss| 0.004160   (range -0.0114 .. +0.0038)
#     credit=action  mean|loss| 0.000092   (range -0.0003 .. +0.0001)   -> 45.4배 붕괴
#
# 이 체인은 그 교란을 먼저 제거한다(게이트 A). 통과하면 credit=belief 를 본실행한다 —
# span margin 을 belief 가 전부 가져갔고(+0.40 vs action -0.005), belief 는 action 과 달리
# 고엔트로피 자유 텍스트라 gradient 가 살아 있기 때문이다.
#
# 각 단계는 마커 파일로 재개 가능하다. 실패해도 다음 단계로 넘어가지 않고 멈춘다
# (잘못된 전제 위에 4시간을 더 쓰지 않기 위해).

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

# gr_log.jsonl 의 mean|loss| — 게이트 A 의 판정값.
meanabs(){ $PY - "$1" <<'PY'
import json,sys
try:
    v=[abs(json.loads(l)["loss"]) for l in open(sys.argv[1])]
    print(f"{sum(v)/len(v):.6f}" if v else "0")
except Exception:
    print("0")
PY
}

say "===== retro 무인 체인 시작 (GPU 1) ====="
[ -s "$TR1F" ] || halt "train jsonl 없음: $TR1F"
[ -s "$J1F" ]  || halt "heldout jsonl 없음: $J1F"

# ── 게이트 A : credit-reduction sum 이 gradient 를 되살리는가 ────────────────
# 기준 — credit=action(mean) 의 mean|loss| 0.000092 대비 10배 이상 회복.
# 참고 상한 — credit=all(mean) 이 0.004160.
GATE_MIN=0.00092
say "--- 게이트 A: credit-reduction=sum 스모크 (300샘플) ---"
for CREDIT in belief action; do
  SM=$OUT/smoke_${CREDIT}_sum
  if [ ! -f "$SM/gr_log.jsonl" ]; then
    say "smoke credit=$CREDIT reduction=sum"
    rm -rf "$SM"
    $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$SM" \
      --full_trace --reward wm --credit "$CREDIT" --credit-reduction sum \
      --max_new_tokens 384 --batch_gen 4 --max_samples 300 --accum 16 \
      --log_every 50 --save_every 100000 --device $DEV \
      > "$OUT/smoke_${CREDIT}_sum.log" 2>&1
    say "smoke $CREDIT rc=$?"
  fi
  M=$(meanabs "$SM/gr_log.jsonl")
  say "  mean|loss| ($CREDIT, sum) = $M   [기준 >= $GATE_MIN, credit=all 참고 0.004160]"
  echo "$CREDIT $M" >> "$OUT/gateA.txt"
done

BEL=$(meanabs "$OUT/smoke_belief_sum/gr_log.jsonl")
PASS=$($PY -c "print(1 if float('$BEL') >= float('$GATE_MIN') else 0)")
if [ "$PASS" != "1" ]; then
  say "게이트 A 실패 — credit=belief+sum 도 gradient 가 죽어 있다 (mean|loss|=$BEL)."
  say "핸드오프 §8 중단 조건대로 REINFORCE 계열을 폐기하고, 남은 시간은 쓰지 않는다."
  say "다음 세션에서 개선 2(belief-swap consistency)로 자원을 전량 이전할 것."
  touch "$OUT/GATE_A_FAILED"; exit 0
fi
say "게이트 A 통과 (belief+sum mean|loss|=$BEL) — 본실행으로 진행"
touch "$OUT/GATE_A_PASSED"

# ── 본실행 : credit=belief + sum ────────────────────────────────────────────
BOUT=$REPO/outputs/step2/retro_beliefcredit_sum_1f
if [ ! -f "$BOUT/TRAINING_DONE" ]; then
  say "--- 본실행: credit=belief --credit-reduction sum (5000샘플, 약 3.7h) ---"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$BOUT" \
    --full_trace --reward wm --credit belief --credit-reduction sum \
    --max_new_tokens 384 --batch_gen 4 --max_samples 5000 --accum 16 \
    --save_every 1250 --device $DEV > "$OUT/train_belief_sum.log" 2>&1
  RC=$?; say "본실행 rc=$RC"
  [ $RC -eq 0 ] && touch "$BOUT/TRAINING_DONE" || halt "본실행 실패 — $OUT/train_belief_sum.log"
fi

# ── 평가 : 생성 acc + ③ 인과 + CI ───────────────────────────────────────────
ADAPT=$BOUT/checkpoint-final
[ -d "$ADAPT" ] || ADAPT=$BOUT
say "--- 평가: 전체 heldout + bootstrap CI (harness v2) ---"
if [ ! -s "$OUT/eval_belief_sum.json" ]; then
  $PY scripts/step2/eval_harness_v2.py --jsonl "$J1F" --adapter "$ADAPT" \
    --limit 0 --device $DEV --out "$OUT/eval_belief_sum.json" \
    > "$OUT/eval_belief_sum.log" 2>&1
  say "생성 평가 rc=$?"
fi
if [ ! -s "$OUT/swap_belief_sum.json" ]; then
  say "--- 개입(belief-swap) 평가 — ③ 인과민감도 ---"
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" --adapter "$ADAPT" \
    --limit 0 --device $DEV --out "$OUT/swap_belief_sum.json" \
    > "$OUT/swap_belief_sum.log" 2>&1
  say "개입 평가 rc=$?"
fi

# ── 시간이 남으면 : credit=action + sum (교란 분리) ─────────────────────────
AOUT=$REPO/outputs/step2/retro_actioncredit_sum_1f
if [ ! -f "$AOUT/TRAINING_DONE" ]; then
  say "--- 추가: credit=action --credit-reduction sum (교란 분리용) ---"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$AOUT" \
    --full_trace --reward wm --credit action --credit-reduction sum \
    --max_new_tokens 384 --batch_gen 4 --max_samples 5000 --accum 16 \
    --save_every 1250 --device $DEV > "$OUT/train_action_sum.log" 2>&1
  RC=$?; say "추가 실행 rc=$RC"
  [ $RC -eq 0 ] && touch "$AOUT/TRAINING_DONE"
fi
AAD=$AOUT/checkpoint-final; [ -d "$AAD" ] || AAD=$AOUT
if [ -f "$AOUT/TRAINING_DONE" ] && [ ! -s "$OUT/eval_action_sum.json" ]; then
  $PY scripts/step2/eval_harness_v2.py --jsonl "$J1F" --adapter "$AAD" \
    --limit 0 --device $DEV --out "$OUT/eval_action_sum.json" \
    > "$OUT/eval_action_sum.log" 2>&1
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" --adapter "$AAD" \
    --limit 0 --device $DEV --out "$OUT/swap_action_sum.json" \
    > "$OUT/swap_action_sum.log" 2>&1
  say "추가 평가 완료"
fi

touch "$OUT/DONE"
say "===== retro 무인 체인 종료 ====="
