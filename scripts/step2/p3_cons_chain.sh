#!/usr/bin/env bash
# p3_cons_chain.sh — 개선 2 / P3: belief-swap consistency loss 무인 체인 (GPU 1대)
#
# 실행 (VSCode/ssh 세션 독립 — 연결이 끊겨도 계속 돈다):
#   cd /mnt/nvme/migration/jihun/EGO_jihun
#   setsid nohup bash scripts/step2/p3_cons_chain.sh \
#     > /mnt/nvme/migration/jihun/EGO/runs/p3_cons/launch.log 2>&1 < /dev/null &
#
# 현황 확인:  http://<host>:7863/   (tools/p3_cons_dashboard.py)
#
# ── 왜 이걸 돌리는가 ──────────────────────────────────────────────────────────
# 07-21 야간 2런(belief·sum·{gt,wm})은 credit 을 살렸지만 acc·③ 둘 다 미달이었다.
# 그리고 ③ 복창 제외 재집계(causal_excl_restate_*.json)에서 **복창 가설이 기각**됐다:
#     gt  ③ 0.0135 → 복창제외 0.0137
#     wm  ③ 0.0255 → 복창제외 0.0260
# 복창 때문에 ③ 가 부풀려진 게 아니라, belief 가 실제로 action 을 거의 조향하지 못한다.
# 간접 신호(보상 설계·credit 배분)로는 ③ 가 오르지 않는다는 것이 2런으로 확인됐으므로,
# ③ 를 **학습 목적에 직접** 넣는다. 이것이 남은 유일한 수단이다.
#
# ── 보상 선택: gt ─────────────────────────────────────────────────────────────
# wm 보상은 ③ 가 더 높았지만 전역 최적해가 'WM top-1 맹목 추종'(r=0.5017 > GT 완벽
# 정책 0.4716)이라 원리적 천장이 있다 (retro_overnight_gpu1_v3.sh:10-20 실측).
# P3 에서는 belief 압력을 consistency 항이 직접 공급하므로 wm 보상으로 그것을 대신할
# 이유가 없다. 따라서 최적해가 GT 인 gt 보상 + cons 항으로 간다.
#
# ── 사전 등록 판정 ────────────────────────────────────────────────────────────
#   성공 = ③(복창 제외) CI 하한 > 0.0137 (gt 어제값)  AND  acc 가 0.2371 대비 MDE(0.045) 이내
#   실패 = acc 붕괴(> MDE 하락) 또는 ③ 무변화
# 실행 중에 이 기준을 바꾸지 않는다.
set -u

EGO_ROOT=/mnt/nvme/migration/jihun/EGO
REPO=/mnt/nvme/migration/jihun/EGO_jihun
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
OUT=$EGO_ROOT/runs/p3_cons
TR1F=$EGO_ROOT/runs/f0_battery/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
HO1F=$EGO_ROOT/runs/f0_battery/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
DEV=cuda:1
LOG=$OUT/chain.log
mkdir -p "$OUT"
cd "$REPO" || exit 1

say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
die() { say "치명적: $*"; touch "$OUT/CHAIN_FAILED"; exit 1; }

# consistency 항의 크기. 크래시체크 실측: RL loss ≈ 0.6~1.0 인데 cons_loss ≈ 14 →
# weight=1.0 이면 cons 가 20배로 지배해 정책을 무너뜨릴 위험이 있다. 반대로 너무 작으면
# 아무 일도 안 일어난다. 값을 추측하지 말고 **스모크로 고른다** — 300샘플 스모크는 15분이라
# 3점 스윕(45분)이 6시간짜리 본실행을 잘못된 weight 로 태우는 것보다 훨씬 싸다.
CW_LIST=${CW_LIST:-"0.05 0.2 1.0"}
CWARM=${CWARM:-16}
# hinge margin: belief 를 바꾼 뒤 대안이 원 행동보다 이만큼(nats) 선호되면 목표 달성으로 보고
# gradient 를 끊는다. 1차 시도는 이게 0(무한정 밀기)이라 정책을 파괴했다.
CMARGIN=${CMARGIN:-0.5}
# reward_ma 붕괴 가드. 스모크가 본 구간(300샘플)에는 붕괴가 없었고 1,200샘플에서 나타났다 —
# 짧은 스모크로는 원리적으로 못 잡으므로 본실행이 자기 자신을 감시한다.
RFLOOR=${RFLOOR:-0.20}

say "===== P3 consistency 체인 시작 (GPU $DEV · cons_weight 후보: $CW_LIST) ====="

# ── 1) 스모크 스윕 — cons 가 움직이면서 정책이 살아남는 최대 weight 를 고른다 ──
for CW in $CW_LIST; do
  SM=$OUT/smoke_cw$CW
  if [ -d "$SM/checkpoint-final" ]; then
    say "--- 스모크 cw=$CW: 이미 있음, 건너뜀 ---"
    continue
  fi
  say "--- 스모크 cw=$CW (300샘플) ---"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$SM" \
    --full_trace --reward gt --credit belief --credit-reduction sum \
    --cons_weight "$CW" --cons_warmup "$CWARM" --cons_buffer 64 \
    --max_new_tokens 384 --batch_gen 4 --max_samples 300 --accum 16 \
    --log_every 50 --save_every 100000 --device $DEV \
    > "$OUT/smoke_cw$CW.log" 2>&1
  say "  스모크 cw=$CW rc=$?"
done

# 선택 규칙 (사전 등록): reward_ma 가 살아 있는(≥0.15) 후보 중 cons_loss 하락폭이 가장 큰 것.
# 하락폭이 어디서도 유의하지 않으면 체인을 중단한다 — 본실행을 태울 근거가 없다.
CW=$($PY - "$OUT" $CW_LIST <<'PY'
import json, math, sys
from pathlib import Path
out, cands = Path(sys.argv[1]), sys.argv[2:]
best, best_drop = None, 0.0
for cw in cands:
    p = out / f"smoke_cw{cw}" / "gr_log.jsonl"
    if not p.is_file():
        continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    cons = [r["cons_loss"] for r in rows if r.get("cons_loss") is not None]
    rew = [r["reward_ma"] for r in rows if r.get("reward_ma") is not None]
    if not cons or not all(math.isfinite(c) for c in cons):
        print(f"# cw={cw}: cons 비유한 — 제외", file=sys.stderr); continue
    if rew and rew[-1] < 0.15:
        print(f"# cw={cw}: reward_ma {rew[-1]} 붕괴 — 제외", file=sys.stderr); continue
    drop = cons[0] - cons[-1]
    print(f"# cw={cw}: cons {cons[0]:.3f}→{cons[-1]:.3f} (Δ{drop:+.3f}) rew={rew[-1] if rew else None}",
          file=sys.stderr)
    if drop > best_drop:
        best, best_drop = cw, drop
print(best or "")
PY
) || true
say "  스윕 선택: cons_weight=${CW:-<없음>}"
[ -n "${CW:-}" ] || die "어떤 weight 에서도 cons_loss 가 내려가지 않았다 — 본실행 근거 없음."
echo "$CW" > "$OUT/chosen_cw.txt"
touch "$OUT/GATE_SMOKE_PASSED"

# ── 2) 본실행 5,000샘플 ─────────────────────────────────────────────────────
FU=$OUT/full
if [ ! -d "$FU/checkpoint-final" ]; then
  say "--- 본실행 (5,000샘플, 약 6h · margin=$CMARGIN · reward_floor=$RFLOOR) ---"
  # 1차 시도(full_FAILED_runaway_oom)가 두 가지로 죽었다. 둘 다 고친 뒤 재시작한다:
  #   (a) hinge 없는 원식이 폭주 — cons_loss +10.7 → −13.8 로 부호를 넘겨 계속 밀렸고
  #       reward_ma 0.34 → 0.115 붕괴. → --cons_margin 으로 목표 달성 시 gradient 정지.
  #   (b) softmax 전 fp32 사본이 OOM — reasoning 이 길어지자 seen≈1800 에서 5GB 요구.
  #       → completion 구간만 잘라 softmax (score_candidates 주석 참조).
  # 스윕 선택(cw=1.0)은 재실행하지 않는다. hinge 는 gap<−margin 에서만 작동하는데
  # 스모크 300샘플 구간의 cons 는 7~12 로 그 지점에서 멀어, 초기 동역학이 동일하기 때문이다.
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$FU" \
    --full_trace --reward gt --credit belief --credit-reduction sum \
    --cons_weight "$CW" --cons_warmup "$CWARM" --cons_buffer 64 \
    --cons_margin "$CMARGIN" --reward_floor "$RFLOOR" --reward_floor_patience 2 \
    --max_new_tokens 384 --batch_gen 4 --max_samples 5000 --accum 16 \
    --log_every 200 --save_every 1250 --device $DEV \
    > "$OUT/full.log" 2>&1
  RC=$?
  say "  본실행 rc=$RC"
  if [ "$RC" = "3" ]; then
    say "  → reward_floor 가드가 정책 붕괴를 잡고 중단했다. cons_weight 를 낮춰 재시도할 것."
    die "정책 붕괴 (reward_floor)"
  fi
else
  say "--- 본실행: 이미 있음, 건너뜀 ---"
fi
[ -d "$FU/checkpoint-final" ] || die "본실행이 체크포인트를 남기지 않았다."

# ── 3) 생성 평가 (heldout 전량) ─────────────────────────────────────────────
if [ ! -s "$OUT/eval.json" ]; then
  say "--- 생성 평가 (n=1,417) ---"
  $PY scripts/step2/eval_harness_v2.py --jsonl "$HO1F" --adapter "$FU/checkpoint-final" \
    --limit 0 --n_boot 10000 --device $DEV --out "$OUT/eval.json" \
    > "$OUT/eval.log" 2>&1
  say "  생성 평가 rc=$?"
fi

# ── 4) ③ belief-swap 개입 (--records 필수 — v2 체인이 여기서 rc=2 로 죽었다) ──
if [ ! -s "$OUT/swap.json" ]; then
  say "--- ③ belief-swap 개입 평가 ---"
  $PY scripts/step2/eval_belief_swap.py --jsonl "$HO1F" \
    --records "$OUT/eval.records.jsonl" --adapter "$FU/checkpoint-final" \
    --limit 0 --device $DEV --out "$OUT/swap.json" \
    > "$OUT/swap.log" 2>&1
  say "  개입 평가 rc=$?"
fi

# ── 5) ③ 복창 제외 재집계 (주지표) ──────────────────────────────────────────
if [ ! -s "$OUT/recount.json" ]; then
  say "--- ③ 복창 제외 재집계 ---"
  $PY scripts/step2/recount_causal_excl_restatement.py \
    --eval_records "$OUT/eval.records.jsonl" --swap_records "$OUT/swap.records.jsonl" \
    --out "$OUT/recount.json" > "$OUT/recount.log" 2>&1
  say "  재집계 rc=$?"
fi

# ── 6) 사전 등록 판정 ───────────────────────────────────────────────────────
$PY - "$OUT/eval.json" "$OUT/recount.json" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
try:
    ev = json.load(open(sys.argv[1])); rc = json.load(open(sys.argv[2]))
except Exception as e:
    print(f"  판정 불가: {e}"); raise SystemExit(0)
acc = ev["full"]["acc"]
sub = rc["subsets"]["excl_restatement"]
lo = sub["causal_sensitivity_ci95"]["lo"]
print(f"  acc = {acc}  (어제 gt 0.2371 · MDE 0.045 → 하한 {0.2371-0.045:.4f})")
print(f"  ③(복창제외) = {sub['causal_sensitivity']}  CI 하한 {lo}  (어제 gt 0.0137)")
ok_acc = acc >= 0.2371 - 0.045
ok_c3 = lo > 0.0137
print(f"  판정: acc {'유지' if ok_acc else '붕괴'} · ③ {'상승(유의)' if ok_c3 else '무변화'}"
      f"  → {'성공' if (ok_acc and ok_c3) else '실패'}")
PY

say "===== P3 consistency 체인 종료 ====="
touch "$OUT/CHAIN_DONE"
