#!/usr/bin/env bash
# f0_rerun_signal.sh — 신호 개선 재검증 run (무인).
#
# 진단 목적: G1≈0(2026-07-18 500-step run)의 원인이 "신호 굶주림"인지 판정.
#   처방 3종 동시 적용 (전부 GT-free · 동결 레시피 외 항목):
#     ① gradient_accumulation 1→8   (업데이트당 프롬프트 2→16, handoff §9-1 처방)
#     ② min_wm_spread 0.05→0.1     (zero-advantage 프롬프트 축소 — 실측 4~25%)
#     ③ wm_likelihood_temp 0.5     (P1 top-1 sharpening — 후보 간 리워드 분산 확대, run2b 선례)
#   판정 지표: train wm_likelihood_reward 곡선의 기울기 (오르면 학습량 문제 → full run 가치 회복,
#             플랫이면 rank1|in5 병목이 본질 → Q1 재서술).
#
# 실행:  setsid nohup bash scripts/step2/f0_rerun_signal.sh \
#          >> $EGO_ROOT/runs/f0_rerun.log 2>&1 < /dev/null &
set -uo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"
PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
BAT=$EGO_ROOT/runs/f0_battery
TRAIN_JSONL=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
OUT=outputs/step2/f0_rerun_ga8_1f
die() { echo "✗ $*"; touch "$BAT/RERUN_FAILED"; exit 1; }

[ -s "$TRAIN_JSONL" ] || die "train 1f jsonl 없음 (leakage PASS 본)"
echo "[$(date +%H:%M:%S)] 재검증 run 시작: GA8 · spread0.1 · sharpen T=0.5 · 1f · 120 update"

if [ ! -d "$OUT/checkpoint-120" ]; then
  python -m accelerate.commands.launch --multi_gpu --num_processes 2 \
    src/ego/step2_vlm_alignment/train_grpo_action.py \
    --model_name         Qwen/Qwen3-VL-8B-Instruct \
    --train_jsonl        "$TRAIN_JSONL" \
    --output_dir         "$OUT" \
    --reward_mode        wm_likelihood_joint \
    --wm_likelihood_norm candidate \
    --wm_likelihood_temp 0.5 \
    --num_frames         1 \
    --mask_frame_prob    0.0 \
    --loss_type          dr_grpo --scale_rewards none --epsilon_high 0.28 \
    --min_wm_spread      0.1 --dynamic_sampling_std_threshold 0 \
    --train_samples      5000 --max_steps 120 \
    --num_generations    8 --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --hide_scores --shuffle_candidates --beta 0.0 --temperature 1.0 \
    --max_completion_length 384 --learning_rate 1e-5 \
    --lora_r 16 --lora_alpha 32 --save_steps 40 --logging_steps 1 \
    --completion_log_every 10 || die "재검증 학습 실패"
fi
[ -d "$OUT/checkpoint-120" ] || die "checkpoint-120 미생성"

echo "[$(date +%H:%M:%S)] 리워드 곡선 요약 (판정 지표)"
$PY - "$OUT/checkpoint-120/trainer_state.json" <<'PYEOF' | tee "$BAT/rerun_reward_curve.txt"
import json, statistics, sys
st = json.load(open(sys.argv[1]))
h = [e for e in st["log_history"] if "rewards/wm_likelihood_reward/mean" in e]
print(f"logged {len(h)} points")
for a, b in [(0, 30), (30, 60), (60, 90), (90, 121)]:
    seg = [e["rewards/wm_likelihood_reward/mean"] for e in h if a <= e["step"] < b]
    zs = [e.get("frac_reward_zero_std", 0) for e in h if a <= e["step"] < b]
    if seg:
        print(f"update {a:>3}-{b:<3}: wm_reward {statistics.mean(seg):.4f}  zero-adv {statistics.mean(zs):.2f}")
PYEOF

echo "[$(date +%H:%M:%S)] 체크포인트 heldout 평가 (40/80 병렬 → 120)"
run_eval() { local name=$1 dev=$2 ck=$3
  [ -s "$BAT/$name.json" ] || $PY scripts/step2/eval_battery.py --jsonl "$J1F" \
      --device "$dev" --adapter "$OUT/checkpoint-$ck" --out "$BAT/$name.json" \
      > "$BAT/$name.log" 2>&1; }
run_eval rerun_step040_1f cuda:0 40 &
run_eval rerun_step080_1f cuda:1 80 &
wait
run_eval rerun_step120_1f cuda:0 120
for n in rerun_step040_1f rerun_step080_1f rerun_step120_1f; do
  [ -s "$BAT/$n.json" ] || die "$n 평가 실패"; done

$PY - "$BAT" <<'PYEOF'
import json, sys
from pathlib import Path
bat = Path(sys.argv[1])
print("=== 재검증 결과 (base 0.242 · 구 step500 0.230) ===")
for n in ["rerun_step040_1f", "rerun_step080_1f", "rerun_step120_1f"]:
    d = json.loads((bat / f"{n}.json").read_text())
    A = (d["acc"] - 0.246 * d["g2_acc"]) / 0.374
    print(f"{n}: acc={d['acc']} G2={d['g2_acc']} wm_follow={d['wm_follow']} A≈{A:.3f}")
PYEOF
touch "$BAT/RERUN_DONE"
echo "[$(date +%H:%M:%S)] RERUN DONE"
