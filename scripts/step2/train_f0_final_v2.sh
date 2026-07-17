#!/usr/bin/env bash
# F0 final v2 — WM-only GRPO, 4프레임 + r16 + 리즈닝 유지(L2).
# 계획: docs/experiments/2026-07-18_f0_final_plan.md
#
# 런 구성 (RUN_MODE 로 전환):
#   validation : 500-step config 검증 run (loss/reward/parsing/checkpoint 확인). freeze 대상 아님.
#   full       : full-data >=1 epoch 최종 run. freeze 후보는 이 run 에서만 나온다.
#
# ⚠ LoRA r16 (이번 run 의 큰 변수는 멀티프레임 하나 — 교란 분리).
#   r64 는 4프레임 효과 확정 후 별도 ablation 으로 반드시 시도할 것 (계획 §0-3).
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_MODE="${RUN_MODE:-validation}"          # validation | full
MASK_PROB="${MASK_FRAME_PROB:-0.0}"         # full run 에서 0.15~0.2 (프록시 재악화 시)

if [ "$RUN_MODE" = "full" ]; then
  STEP_ARGS="--num_train_epochs 1.0 --max_steps -1"
  OUT="outputs/step2/f0_final_v2_full"
else
  STEP_ARGS="--max_steps 500"
  OUT="outputs/step2/f0_final_v2_val"
fi
echo "[RUN_MODE=$RUN_MODE] mask_frame_prob=$MASK_PROB out=$OUT lora_r=16"

# 2xH200 기준. num_frames 4 는 vision token 증가 → 배치/속도 재측정 필요.
accelerate launch --multi_gpu --num_processes 2 \
  src/ego/step2_vlm_alignment/train_grpo_action.py \
  --model_name         Qwen/Qwen3-VL-8B-Instruct \
  --train_jsonl        "${EGO_TRAIN_JSONL:?set EGO_TRAIN_JSONL}" \
  --output_dir         "$OUT" \
  --reward_mode        wm_likelihood_joint \
  --wm_likelihood_norm candidate \
  --num_frames         4 \
  --mask_frame_prob    "$MASK_PROB" \
  --loss_type          dr_grpo --scale_rewards none --epsilon_high 0.28 \
  --min_wm_spread      0.05 --dynamic_sampling_std_threshold 0 \
  --train_samples      5000 $STEP_ARGS \
  --num_generations    8 --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --hide_scores --shuffle_candidates --beta 0.0 --temperature 1.0 \
  --max_completion_length 384 --learning_rate 1e-5 \
  --lora_r 16 --lora_alpha 32 --save_steps 125 --logging_steps 2 \
  --completion_log_every 25

echo "[done] $RUN_MODE run → $OUT"
echo "  프록시: $OUT/reasoning_proxy.jsonl (history_reference_rate·belief_restatement_rate)"
if [ "$RUN_MODE" = "validation" ]; then
  echo "  ⚠ 검증 run 완료. 문제 없으면 RUN_MODE=full 로 최종 run 후 freeze."
fi
