#!/bin/bash
# Run 2 분기 A — Run 1 건강(곡선 상승+무붕괴) 시: Run 1 설정 그대로 + P3 추가
#
# P3 = think_support_reward (coherence regularizer):
#   동결 base 모델(text-only)로 p(선택 행동 | 후보, '결론 마스킹된' think) 측정.
#   가중치 0.25 — P1(최대~0.8)·P4(0.25) 이하 (안전장치 b). 결론 마스킹은 답안 예고편
#   hacking 차단 (안전장치 a). ref 모델 forward 추가로 step ~12s → ~16s 예상 (총 ~5.5h).
#
# 종료 후:
#   bash eval_checkpoints_run1.sh runs/grpo_run2_branchA 500       # G1/G2 곡선
#   python eval_reasoning_trace.py --records runs/grpo_run2_branchA/heldout_eval/step1250.records.jsonl \
#     --mode mask --out runs/grpo_run2_branchA/trace_eval/mask.json  # P3 hacking 판별 (lift 유지 여부)
#   → 마스킹 후 lift 붕괴 시 Run 3 에서 P3 제외 (handoff §2 Run 2 판정)

set -euo pipefail
cd ~/work/jihun/EGO
source activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUTPUT_DIR="runs/grpo_run2_branchA"
mkdir -p "$OUTPUT_DIR"
echo "[$(date)] Starting grpo_run2_branchA (Run1 + P3 w=0.25)" | tee "$OUTPUT_DIR/launch.log"

accelerate launch --multi_gpu --num_processes 2 \
  train_qwen25vl_grpo_ek100.py \
  --train_jsonl        data/grpo_dataset/grpo_train.jsonl \
  --output_dir         "$OUTPUT_DIR" \
  --reward_mode        wm_likelihood_p3 \
  --p3_weight          0.25 \
  --wm_likelihood_norm candidate \
  --loss_type          dr_grpo \
  --scale_rewards      none \
  --epsilon_high       0.28 \
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
