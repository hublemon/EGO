#!/usr/bin/env bash
# f0_span_chain.sh — Phase 2b: F0-WA (clean WM + action-span credit + ref-KL) 무인 체인.
#   P1-6 대응: advantage 를 <action> span 에만 부여, reasoning/belief 는 beta(ref-KL)로 보존.
#   기반: wm_clean (Phase 1 F0-W 와 동일 데이터·하이퍼) — 변경 변수는 credit 배분 + beta 뿐.
#   목적 지표: ③ belief-swap 인과 민감도 (F0-W 대비) + acc 유지 여부.
# F0_CLEAN_DONE 대기 → smoke → 2-GPU 500 step → ckpt 곡선 + step500 ③ swap → F0_SPAN_DONE.
set -euo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
[ -f /root/.config/ego/letsur.env ] && source /root/.config/ego/letsur.env
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"; PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
BAT=$EGO_ROOT/runs/f0_battery
TRAIN_JSONL=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
OUT=$REPO/outputs/step2/f0_wa_spancredit_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/F0_SPAN_FAILED"; exit 1; }

say "=== F0-WA span-credit 체인: F0_CLEAN_DONE 대기 ==="
while [ ! -f "$BAT/F0_CLEAN_DONE" ] && [ ! -f "$BAT/F0_CLEAN_FAILED" ]; do sleep 120; done
[ -f "$BAT/F0_CLEAN_FAILED" ] && { say "Phase 1 실패 — span 체인 중단"; touch "$BAT/F0_SPAN_FAILED"; exit 1; }
say "F0_CLEAN_DONE 확인 — Phase 2b 시작"

TRL_V=$($PY -c "import trl; print(trl.__version__)")
CODE_SHA=$(sha256sum src/ego/step2_vlm_alignment/train_grpo_action.py | cut -c1-12)

COMMON_ARGS=(--model_name Qwen/Qwen3-VL-8B-Instruct
  --train_jsonl "$TRAIN_JSONL" --wm_likelihood_norm candidate
  --num_frames 1 --mask_frame_prob 0.0
  --loss_type dr_grpo --scale_rewards none --epsilon_high 0.28
  --min_wm_spread 0.05 --dynamic_sampling_std_threshold 0
  --train_samples 5000
  --num_generations 8 --per_device_train_batch_size 8
  --gradient_accumulation_steps 1
  --hide_scores --shuffle_candidates --temperature 1.0
  --max_completion_length 384 --learning_rate 1e-5
  --lora_r 16 --lora_alpha 32 --completion_log_every 25
  --reward_mode wm_clean --action_span_credit --span_credit_lambda 0.0 --beta 0.04)

# ── smoke: span 탐지 + grad>0 (8 step, 단일 GPU) ────────────────────────────
SKEY="span_trl${TRL_V}_${CODE_SHA}"
if [ ! -f "$BAT/.f0smoke_${SKEY}" ]; then
  say "smoke: span-credit 8 step (cuda:0)"
  SDIR=$BAT/f0smoke_spancredit; rm -rf "$SDIR"
  CUDA_VISIBLE_DEVICES=0 $PY src/ego/step2_vlm_alignment/train_grpo_action.py \
    "${COMMON_ARGS[@]}" --output_dir "$SDIR" \
    --max_steps 8 --logging_steps 1 --save_steps 999 --train_samples 64 \
    > "$SDIR.log" 2>&1 || die "smoke 실행 실패 — $SDIR.log"
  $PY - "$SDIR/reward_log.jsonl" <<'PYEOF' || die "smoke grad_norm=0 — 무학습"
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
assert rows and any(float(r.get("grad_norm",0))>0 for r in rows), "all grad zero"
print("[smoke] grad_norm OK:", [round(float(r.get("grad_norm",0)),4) for r in rows])
PYEOF
  grep -a 'span_credit' "$SDIR.log" | tail -2 || true
  touch "$BAT/.f0smoke_${SKEY}"; say "smoke OK"
else say "smoke skip ($SKEY)"; fi

# ── 학습: 2-GPU DDP 500 step (beta>0 → ref forward 포함) ────────────────────
judge_follow(){ local rd=$1 tp=$2
  while kill -0 "$tp" 2>/dev/null; do sleep 600
    [ -s "$rd/completion_samples.jsonl" ] && \
      $PY -m ego.step2_vlm_alignment.judge_reasoning --run_dir "$rd" \
        --judge_model gemini-2.5-pro --per_step 3 >> "$rd/judge_follow.log" 2>&1 || true
  done
  $PY -m ego.step2_vlm_alignment.judge_reasoning --run_dir "$rd" \
    --judge_model gemini-2.5-pro --per_step 3 >> "$rd/judge_follow.log" 2>&1 || true
}
if [ ! -f "$OUT/TRAINING_DONE" ]; then
  rm -rf "$OUT"
  say "F0-WA 학습 (2-GPU, 500 step, beta 0.04)"
  $PY -m accelerate.commands.launch --multi_gpu --num_processes 2 \
    src/ego/step2_vlm_alignment/train_grpo_action.py \
    "${COMMON_ARGS[@]}" --output_dir "$OUT" \
    --max_steps 500 --save_steps 125 --logging_steps 2 \
    > "$BAT/train_spancredit.log" 2>&1 & TP=$!
  judge_follow "$OUT" $TP & JP=$!
  wait $TP || die "F0-WA 학습 실패 — $BAT/train_spancredit.log"
  ls "$OUT"/checkpoint-500 >/dev/null 2>&1 || die "최종 체크포인트 없음"
  touch "$OUT/TRAINING_DONE"
  wait $JP 2>/dev/null || true
fi

# ── 평가: ckpt 곡선(cuda:1 순차) + step500 생성 records → ③ swap(cuda:0) ──
for st in 125 250 375 500; do
  [ -s "$BAT/f0wa_step${st}.json" ] && continue
  $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:1 \
    --adapter "$OUT/checkpoint-$st" --out "$BAT/f0wa_step${st}.json" \
    > "$BAT/f0wa_step${st}.log" 2>&1 || die "eval step$st 실패"
done
if [ ! -s "$BAT/swap_f0wa.json" ]; then
  say "③ belief-swap on F0-WA (목적 지표)"
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
    --records "$BAT/f0wa_step500.records.jsonl" --adapter "$OUT/checkpoint-500" \
    --device cuda:0 --out "$BAT/swap_f0wa.json" > "$BAT/swap_f0wa.log" 2>&1 \
    || die "③ swap 실패"
fi
# 비교용: F0-W(clean, span 없음) step500 의 ③ 도 측정 (records 는 Phase1 산출)
if [ ! -s "$BAT/swap_f0w.json" ] && [ -s "$BAT/f0w_step500.records.jsonl" ]; then
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
    --records "$BAT/f0w_step500.records.jsonl" \
    --adapter "$REPO/outputs/step2/f0_clean_wm_1f/checkpoint-500" \
    --device cuda:1 --out "$BAT/swap_f0w.json" > "$BAT/swap_f0w.log" 2>&1 || true
fi

# ── 요약 ────────────────────────────────────────────────────────────────────
$PY - "$BAT" "$OUT" > "$BAT/F0_SPAN_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat, out = Path(sys.argv[1]), Path(sys.argv[2])
def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None
print("# F0-WA (action-span credit) 결과 — Phase 2b 자동 생성\n")
print("| 모델 | acc | G2 | wm_follow | parse |")
print("|---|---|---|---|---|")
for name, p in ([("F0-N base", bat/"base_1f_strict.json"),
                 ("F0-W clean step500", bat/"f0w_step500.json")] +
                [(f"F0-WA step{s}", bat/f"f0wa_step{s}.json") for s in (125,250,375,500)]):
    d = J(p)
    if d: print(f"| {name} | {d['acc']} | {d.get('g2_acc')} | {d.get('wm_follow')} | {d.get('parse_rate')} |")
print("\n## ③ 인과 민감도 (이 단계의 목적 지표)")
for name, p in (("base", bat/"swap_base_1f.json"), ("F0-L legacy", bat/"swap_step500_1f.json"),
                ("F0-W clean", bat/"swap_f0w.json"), ("F0-WA span-credit", bat/"swap_f0wa.json")):
    d = J(p)
    if d: print(f"- {name}: {d['causal_sensitivity']} (swap {d['swap_action_change']} / ctrl {d['control_action_change']})")
js = J(out/"judge_curve_summary.json")
if js:
    t = js.get("total", {})
    print(f"\njudge(gemini) total {t.get('first_half')}→{t.get('second_half')} · "
          f"belief_action_link Δ{js.get('belief_action_link',{}).get('delta')}")
PYEOF
cat "$BAT/F0_SPAN_RESULTS.md"
touch "$BAT/F0_SPAN_DONE"; say "=== F0-WA 체인 DONE ==="
