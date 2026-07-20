#!/usr/bin/env bash
# pro_clean_chain.sh — F0 재편 Phase 1 무인 체인 (2026-07-19 통합 핸드오프 반영).
#   F0-W: wm_clean  (validity floor + WM likelihood 단독; think_convergence 제거)
#   F0-G: gt_only   (validity floor + GT binary; oracle-subset(GT∈top5) 학습; skyline)
# 두 arm 을 GPU 1대씩 병렬 학습(500 step, F0-L 과 동일 하이퍼) 후 체크포인트 곡선 평가.
# 학습 중 25 step 간격 completion 샘플을 gemini-2.5-pro 가 7항목 루브릭으로 추적 채점
# (judge follower — 멱등, API 전용, 학습 비용 0).
# 마커: <run>/TRAINING_DONE (train+save 성공시에만) · F0_CLEAN_DONE / F0_CLEAN_FAILED.
# 주의: resume 미지원 대신 "부분 체크포인트는 완료로 절대 오인하지 않음" — TRAINING_DONE 없는
#   run 디렉토리는 재학습 대상 (500 step ≈ 95분이라 재학습 비용 수용).
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
OUT_W=$REPO/outputs/step2/f0_clean_wm_1f
OUT_G=$REPO/outputs/step2/f0_gt_only_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/F0_CLEAN_FAILED"; exit 1; }

# ── 실행 환경 기록 + 해시 묶인 스모크 키 (코드/환경 변경 시 스모크 자동 재실행) ──
TRL_V=$($PY -c "import trl; print(trl.__version__)")
CODE_SHA=$(sha256sum src/ego/step2_vlm_alignment/train_grpo_action.py | cut -c1-12)
SMOKE_KEY="trl${TRL_V}_${CODE_SHA}"
say "=== F0 clean 체인 시작 (trl $TRL_V, code $CODE_SHA) ==="

# ── 공통 학습 인자 (F0-L 과 동일; 차이는 reward_mode 와 GPU 수뿐) ──────────
COMMON_ARGS=(--model_name Qwen/Qwen3-VL-8B-Instruct
  --train_jsonl "$TRAIN_JSONL" --wm_likelihood_norm candidate
  --num_frames 1 --mask_frame_prob 0.0
  --loss_type dr_grpo --scale_rewards none --epsilon_high 0.28
  --min_wm_spread 0.05 --dynamic_sampling_std_threshold 0
  --train_samples 5000
  --num_generations 8 --per_device_train_batch_size 8
  --gradient_accumulation_steps 1
  --hide_scores --shuffle_candidates --beta 0.0 --temperature 1.0
  --max_completion_length 384 --learning_rate 1e-5
  --lora_r 16 --lora_alpha 32 --logging_steps 2 --completion_log_every 25)

# ── 스모크: 실제 optimizer step + grad_norm>0 확인 (모드별) ─────────────────
run_smoke(){ local mode=$1 gpu=$2
  local sdir=$BAT/f0smoke_${mode}
  rm -rf "$sdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY src/ego/step2_vlm_alignment/train_grpo_action.py \
    "${COMMON_ARGS[@]}" --reward_mode "$mode" --output_dir "$sdir" \
    --max_steps 8 --logging_steps 1 --save_steps 999 --train_samples 64 \
    > "$sdir.log" 2>&1 || die "smoke($mode) 실행 실패 — $sdir.log"
  $PY - "$sdir/reward_log.jsonl" <<'PYEOF' || die "smoke($mode) grad_norm=0 — 무학습"
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
assert rows and any(float(r.get("grad_norm",0))>0 for r in rows), "all grad zero"
print("[smoke] grad_norm OK:", [r.get("grad_norm") for r in rows])
PYEOF
}
if [ ! -f "$BAT/.f0smoke_${SMOKE_KEY}" ]; then
  say "smoke: wm_clean(cuda:0) + gt_only(cuda:1) — 8 step 실보상·실역전파 (그룹 zero-advantage 확률 대비)"
  run_smoke wm_clean 0 & P0=$!
  run_smoke gt_only 1 & P1=$!
  wait $P0 || die "smoke wm_clean 실패"; wait $P1 || die "smoke gt_only 실패"
  touch "$BAT/.f0smoke_${SMOKE_KEY}"; say "smoke OK ($SMOKE_KEY)"
else say "smoke skip (marker $SMOKE_KEY)"; fi

# ── judge follower: 학습 중 25-step 곡선을 gemini 가 추적 채점 (멱등) ────────
judge_follow(){ local rd=$1 trainpid=$2
  while kill -0 "$trainpid" 2>/dev/null; do
    sleep 600
    [ -s "$rd/completion_samples.jsonl" ] && \
      $PY -m ego.step2_vlm_alignment.judge_reasoning --run_dir "$rd" \
        --judge_model gemini-2.5-pro --per_step 3 >> "$rd/judge_follow.log" 2>&1 || true
  done
  # 최종 패스 (남은 step 채점)
  $PY -m ego.step2_vlm_alignment.judge_reasoning --run_dir "$rd" \
    --judge_model gemini-2.5-pro --per_step 3 >> "$rd/judge_follow.log" 2>&1 || true
}

# ── 학습 (병렬, GPU 1대/arm) ────────────────────────────────────────────────
train_arm(){ local mode=$1 out=$2 gpu=$3
  if [ -f "$out/TRAINING_DONE" ]; then say "$mode: TRAINING_DONE — skip"; return 0; fi
  rm -rf "$out"   # 부분 체크포인트는 완료로 오인하지 않고 재학습
  CUDA_VISIBLE_DEVICES=$gpu $PY src/ego/step2_vlm_alignment/train_grpo_action.py \
    "${COMMON_ARGS[@]}" --reward_mode "$mode" --output_dir "$out" \
    --max_steps 500 --save_steps 125 > "$BAT/train_${mode}.log" 2>&1
}
say "학습 시작: F0-W(cuda:0) ∥ F0-G(cuda:1) — 500 step, judge follower 가동"
train_arm wm_clean "$OUT_W" 0 & TW=$!
train_arm gt_only  "$OUT_G" 1 & TG=$!
judge_follow "$OUT_W" $TW & JW=$!
judge_follow "$OUT_G" $TG & JG=$!
FAIL=0
wait $TW || { say "✗ F0-W 학습 실패 — $BAT/train_wm_clean.log"; FAIL=1; }
wait $TG || { say "✗ F0-G 학습 실패 — $BAT/train_gt_only.log"; FAIL=1; }
[ $FAIL -eq 0 ] || die "학습 단계 실패"
for out in "$OUT_W" "$OUT_G"; do
  ls "$out"/checkpoint-500 >/dev/null 2>&1 || die "$out 최종 체크포인트 없음"
  touch "$out/TRAINING_DONE"
done
# 실행 manifest
$PY - "$OUT_W" "$OUT_G" "$TRL_V" "$CODE_SHA" <<'PYEOF'
import json, sys
for out in sys.argv[1:3]:
    json.dump({"trl": sys.argv[3], "code_sha": sys.argv[4], "steps": 500,
               "gpus": 1, "effective_prompts_per_step": 1,
               "note": "F0-L 과 동일 하이퍼, GPU 수만 1 (arm 간 상호 비교는 동일 조건)"},
              open(f"{out}/run_manifest.json", "w"), indent=2)
PYEOF
wait $JW $JG 2>/dev/null || true
say "학습 완료 + judge 최종 패스 완료"

# ── 체크포인트 곡선 평가 (arm 병렬, arm 내 순차) ────────────────────────────
eval_arm(){ local tag=$1 out=$2 gpu=$3
  for st in 125 250 375 500; do
    [ -s "$BAT/${tag}_step${st}.json" ] && continue
    $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:$gpu \
      --adapter "$out/checkpoint-$st" --out "$BAT/${tag}_step${st}.json" \
      > "$BAT/${tag}_step${st}.log" 2>&1 || die "eval $tag step$st 실패"
  done
}
say "체크포인트 곡선 평가 (500샘플 × 4ckpt × 2arm)"
eval_arm f0w "$OUT_W" 0 & E0=$!
eval_arm f0g "$OUT_G" 1 & E1=$!
wait $E0 || die "F0-W 평가 실패"; wait $E1 || die "F0-G 평가 실패"

# ── 요약 (5열 표 + judge 곡선 + oracle 3분리) ───────────────────────────────
$PY - "$BAT" "$OUT_W" "$OUT_G" > "$BAT/F0_CLEAN_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat, outw, outg = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None
print("# F0 재편 Phase 1 결과 (자동 생성)\n")
print("| 모델 | acc | G2 | cond.acc(GT∈top5) | wm_follow | parse |")
print("|---|---|---|---|---|---|")
rows = [("F0-N base", bat/"base_1f_strict.json"), ("F0-L legacy step500", bat/"step500_1f.json")]
rows += [(f"F0-W step{s}", bat/f"f0w_step{s}.json") for s in (125,250,375,500)]
rows += [(f"F0-G step{s}", bat/f"f0g_step{s}.json") for s in (125,250,375,500)]
for name, p in rows:
    d = J(p)
    if not d: continue
    print(f"| {name} | {d['acc']} | {d.get('g2_acc')} | {d.get('acc_given_gt_in_top5','—')} "
          f"| {d.get('wm_follow')} | {d.get('parse_rate')} |")
om = J(outg/"oracle_manifest.json")
if om: print(f"\noracle-subset manifest: coverage@5={om['candidate_coverage_at_k']} "
             f"(train {om['num_gt_in_topk']}/{om['num_total_prompts']}, policy=drop)")
for tag, out in (("F0-W", outw), ("F0-G", outg)):
    js = J(out/"judge_curve_summary.json")
    if js:
        t = js.get("total", {})
        print(f"\n{tag} judge(gemini) total: {t.get('first_half')} → {t.get('second_half')} "
              f"(Δ{t.get('delta')}) · belief_action_link Δ"
              f"{js.get('belief_action_link',{}).get('delta')}")
# 판정 힌트
fw = J(bat/"f0w_step500.json"); fg = J(bat/"f0g_step500.json"); base = J(bat/"base_1f_strict.json")
if fw and fg and base:
    print(f"\n판정: F0-W Δacc={fw['acc']-base['acc']:+.3f}, F0-G Δacc={fg['acc']-base['acc']:+.3f} (vs base {base['acc']})")
    if fg['acc'] - base['acc'] < 0.02:
        print("→ F0-G ≈ base: credit-assignment/용량 분리를 위해 F0-GA(action-only 진단) 실행 권고 (NEEDS_DECISION_F0GA)")
PYEOF
cat "$BAT/F0_CLEAN_RESULTS.md"
fg_acc=$($PY -c "import json;print(json.load(open('$BAT/f0g_step500.json'))['acc'])")
base_acc=$($PY -c "import json;print(json.load(open('$BAT/base_1f_strict.json'))['acc'])")
$PY -c "exit(0 if $fg_acc - $base_acc < 0.02 else 1)" && touch "$BAT/NEEDS_DECISION_F0GA" || true
touch "$BAT/F0_CLEAN_DONE"; say "=== F0 clean 체인 DONE ==="
