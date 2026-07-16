#!/bin/bash
# Run 2 분기 C / C' — P4 hacking(수렴 흉내) 또는 format-only 붕괴 관찰 시
#
# 처방 (Run 1 대비):
#   --reward_weights 1,1.5,1,0.5   (fn 순서: format, gate, P1_wmlik, P4_conv)
#       → P4 가중치 하향(1→0.5) + hallucination gate 강화(1→1.5)
#   분기 C'(think 단어수≤5 / 다양성 0)면 추가로:
#       EPS_HIGH=0.35 bash run_grpo_run2_branchC.sh    (clip-higher 강화)
#
# 값은 Run 1 진단(reward_log 의 P1 vs P4 궤적, think_analysis)을 보고 조정.

set -euo pipefail
cd ~/work/jihun/EGO
source activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EPS_HIGH="${EPS_HIGH:-0.28}"
OUTPUT_DIR="runs/grpo_run2_branchC"
mkdir -p "$OUTPUT_DIR"
echo "[$(date)] Starting grpo_run2_branchC (P4 w0.5, gate w1.5, eps_high=$EPS_HIGH)" | tee "$OUTPUT_DIR/launch.log"

accelerate launch --multi_gpu --num_processes 2 \
  train_qwen25vl_grpo_ek100.py \
  --train_jsonl        data/grpo_dataset/grpo_train.jsonl \
  --output_dir         "$OUTPUT_DIR" \
  --reward_mode        wm_likelihood \
  --reward_weights     1,1.5,1,0.5 \
  --wm_likelihood_norm candidate \
  --loss_type          dr_grpo \
  --scale_rewards      none \
  --epsilon_high       "$EPS_HIGH" \
  --min_wm_spread      0.05 \
  --dynamic_sampling_std_threshold 0.02 \
  --train_samples      5000 \
  --max_steps          1250 \
  --num_generations    8 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --hide_scores --beta 0.0 --temperature 0.8 --max_completion_length 256 \
  --learning_rate 1e-5 --lora_r 16 --lora_alpha 32 \
  --save_steps 125 --logging_steps 2 \
  2>&1 | tee -a "$OUTPUT_DIR/launch.log"
