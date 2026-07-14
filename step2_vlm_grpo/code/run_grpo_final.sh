#!/bin/bash
# Exp 14: grpo_final — 최종 통합 실험
#
# exp11(think_gt_combo) 대비 개선점:
#   1. 전체 데이터 4947샘플 (3000 → +65%)
#   2. max_steps 1250 (750 → +67%, 여전히 ~0.5 epoch)
#   3. GT-not-in-top5 필터 (think 모드용, 51개 제거)
#   4. num_generations 8 (4 → 2×, 분산 감소)
#   5. gt_accuracy_reward_think_v3 (퍼지 noun 매칭 추가)
#   6. beta=0.01 (KL 정규화, 더 긴 학습 안정화)
#   7. max_completion_length=256 유지 (메모리 여유분 num_gen=8에 배분)
#
# step 계산:
#   per_device=8, num_gen=8, world=2 → prompts/step = (8×2)/8 = 2
#   4947 / 2 = 2473 steps/epoch → 1250 steps ≈ 0.5 epoch
#
# GPU: 2×H200, ~3.5h 예상

set -euo pipefail
cd ~/work/jihun/EGO
source activate.sh

OUTPUT_DIR="runs/grpo_final"
mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Starting grpo_final experiment" | tee "$OUTPUT_DIR/launch.log"

accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  train_qwen25vl_grpo_ek100.py \
  --train_jsonl        data/grpo_dataset/grpo_train.jsonl \
  --output_dir         "$OUTPUT_DIR" \
  --reward_mode        think_gt_final \
  --train_samples      5000 \
  --max_steps          1250 \
  --num_generations    8 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --drop_unrewardable_samples \
  --hide_scores \
  --beta               0.01 \
  --temperature        0.8 \
  --max_completion_length 256 \
  --learning_rate      1e-5 \
  --lora_r             16 \
  --lora_alpha         32 \
  --save_steps         250 \
  --logging_steps      2 \
  2>&1 | tee -a "$OUTPUT_DIR/launch.log"
