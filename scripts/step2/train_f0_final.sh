#!/usr/bin/env bash
# F0 final — WM-only GRPO (joint action top-5). 결과: docs/experiments/2026-07-17_f0_final.md
set -euo pipefail
cd "$(dirname "$0")/../.."

# 2×H200 기준 500 step ≈ 90분 (10.9 s/step).
accelerate launch --multi_gpu --num_processes 2 \
  src/ego/step2_vlm_alignment/train_grpo_action.py \
  --model_name         Qwen/Qwen3-VL-8B-Instruct \
  --train_jsonl        "${EGO_TRAIN_JSONL:?set EGO_TRAIN_JSONL}" \
  --output_dir         outputs/step2/f0_final_wm_only \
  --reward_mode        wm_likelihood_joint \
  --wm_likelihood_norm candidate \
  --loss_type          dr_grpo --scale_rewards none --epsilon_high 0.28 \
  --min_wm_spread      0.05 --dynamic_sampling_std_threshold 0 \
  --train_samples      5000 --max_steps 500 \
  --num_generations    8 --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --hide_scores --beta 0.0 --temperature 1.0 \
  --max_completion_length 384 --learning_rate 1e-5 \
  --lora_r 16 --lora_alpha 32 --save_steps 125 --logging_steps 2 \
  --completion_log_every 25
