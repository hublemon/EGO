#!/usr/bin/env bash
# B0 — full-trace sequence-level DPO (핸드오프 §11).
# 선행: build_b0_pairs.sh 로 b0_dpo_train.jsonl 생성 + validate PASS.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="src:${PYTHONPATH:-}"

FAA_ADAPTER="${FAA_ADAPTER:?set FAA_ADAPTER (frozen FAA — 초기값+reference)}"
OUT="${OUT:-outputs/step2/b0_full_trace_dpo}"

accelerate launch --multi_gpu --num_processes 2 \
  -m ego.step2_vlm_alignment.b0.train_b0_dpo \
  --dpo_jsonl    data/grpo_dataset/b0_dpo_train.jsonl \
  --faa_adapter  "$FAA_ADAPTER" \
  --output_dir   "$OUT" \
  --model_name   Qwen/Qwen3-VL-8B-Instruct \
  --beta 0.1 --learning_rate 5e-6 --num_train_epochs 1.0 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --lora_r 16 --lora_alpha 32 --save_steps 100 --logging_steps 2

echo "[done] B0 → $OUT"
