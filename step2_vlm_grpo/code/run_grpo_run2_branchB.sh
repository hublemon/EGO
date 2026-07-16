#!/bin/bash
# Run 2 분기 B — Run 1 이 advantage 소실 잔존(reward 정체 + ds 필터율 과다)일 때
#
# 처방 (Run 1 대비 변경 3~4개, 그 외 동일):
#   --min_wm_spread                  0.05 → 0.02  (정적 필터 완화 — 데이터 더 유지)
#   --dynamic_sampling_std_threshold 0.02 → 0.01  (런타임 마스킹 완화)
#   --temperature                    0.8  → 1.0   (생성 다양성 ↑ → 그룹 내 분산 ↑)
#   NUM_GEN: 기본 8. ★Run 1 reward_log 에서 ds_frac_groups_filtered 가 '후반에 증가
#   추세'였던 경우에만 12 로 상향 (사용자 확정 정책, handoff §3 고정 설정) — 아래 변수로.
#
# 실행 전 Run 1 진단 확인:
#   grep ds_frac runs/grpo_run1_wmonly/diagnostics.log   # "초반 → 최근" 추세
#   증가 추세면: NUM_GEN=12 bash run_grpo_run2_branchB.sh   (학습 ~6h)
#   아니면:      bash run_grpo_run2_branchB.sh              (학습 ~4h)

set -euo pipefail
cd ~/work/jihun/EGO
source activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

NUM_GEN="${NUM_GEN:-8}"
OUTPUT_DIR="runs/grpo_run2_branchB"
mkdir -p "$OUTPUT_DIR"
echo "[$(date)] Starting grpo_run2_branchB (필터 완화 + temp 1.0, num_gen=$NUM_GEN)" | tee "$OUTPUT_DIR/launch.log"

accelerate launch --multi_gpu --num_processes 2 \
  train_qwen25vl_grpo_ek100.py \
  --train_jsonl        data/grpo_dataset/grpo_train.jsonl \
  --output_dir         "$OUTPUT_DIR" \
  --reward_mode        wm_likelihood \
  --wm_likelihood_norm candidate \
  --loss_type          dr_grpo \
  --scale_rewards      none \
  --epsilon_high       0.28 \
  --min_wm_spread      0.02 \
  --dynamic_sampling_std_threshold 0.01 \
  --train_samples      5000 \
  --max_steps          1250 \
  --num_generations    "$NUM_GEN" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --hide_scores --beta 0.0 --temperature 1.0 --max_completion_length 256 \
  --learning_rate 1e-5 --lora_r 16 --lora_alpha 32 \
  --save_steps 125 --logging_steps 2 \
  2>&1 | tee -a "$OUTPUT_DIR/launch.log"
