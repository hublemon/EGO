#!/bin/bash
# Run 1: grpo_run1_wmonly — WM-only (GT-free) 학습 성립 검증 (G1)
#
# docs/GRPO_WMONLY_HANDOFF.md 의 Run 1. Exp.14(run_grpo_final.sh) 대비 변경:
#   1. reward_mode wm_likelihood — GT 신호 완전 제거. P1(후보셋 재정규화 likelihood, 주신호)
#      + P4(think_convergence, 후보 언급 수렴) + 최소 구조 gate(format/candidate).
#      think_quality_reward(길이 보너스)는 '길이 보너스 금지' 원칙으로 제외.
#   2. --loss_type dr_grpo + --scale_rewards none — P5 길이 편향 제거 (Dr. GRPO 정합 세트)
#   3. --epsilon_high 0.28 — P2 clip-higher (하한 ε=0.2 유지, 상한만 완화)
#   4. --min_wm_spread 0.05 — P2 dynamic sampling 정적 절반 (flat 프롬프트 사전 제거, 하위 7%)
#   5. --dynamic_sampling_std_threshold 0.02 — P2 런타임 절반 (무신호 그룹 advantage 마스킹)
#   6. beta 0.0 — Dr.GRPO 계열 KL off (ref 모델 메모리 절약)
#   7. save_steps 125 — trace eval 곡선용 체크포인트 촘촘히 (10개)
#
# step 계산: per_device=8, num_gen=8, world=2 → prompts/step = 2
#   4,646(spread 필터 후) / 2 = 2,323 steps/epoch → 1250 steps ≈ 0.54 epoch
# GPU: 2×H200, Exp.14 기준 ~3.5h + dynamic sampling 오버헤드 소폭
#
# 학습 후 (판정은 docs/GRPO_WMONLY_HANDOFF.md 의 Run 1 분기표):
#   bash eval_checkpoints_run1.sh          # held-out 곡선 (G1/G2)
#   grep ds_frac reward_log.jsonl 등 로그 진단

set -euo pipefail
cd ~/work/jihun/EGO
source activate.sh

# HF Hub 504 장애 대비: 모델/프로세서는 전부 로컬 캐시에 있음 — 원격 HEAD 체크 생략
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTPUT_DIR="runs/grpo_run1_wmonly"
mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Starting grpo_run1_wmonly (WM-only, GT-free)" | tee "$OUTPUT_DIR/launch.log"

accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  train_qwen25vl_grpo_ek100.py \
  --train_jsonl        data/grpo_dataset/grpo_train.jsonl \
  --output_dir         "$OUTPUT_DIR" \
  --reward_mode        wm_likelihood \
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
  --hide_scores \
  --beta               0.0 \
  --temperature        0.8 \
  --max_completion_length 256 \
  --learning_rate      1e-5 \
  --lora_r             16 \
  --lora_alpha         32 \
  --save_steps         125 \
  --logging_steps      2 \
  2>&1 | tee -a "$OUTPUT_DIR/launch.log"
