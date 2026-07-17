#!/usr/bin/env bash
# F0 final v2 데이터 빌드 — strict cutoff + 4프레임 + L2-c 정렬 + B0 메타 분리.
# 계획: docs/experiments/2026-07-18_f0_final_plan.md
# 선행: selected_{train,heldout}.jsonl, predictions_{train,heldout}.jsonl (기존 파이프라인)
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=src/ego/step2_vlm_alignment/data

for SPLIT in train validation; do
  echo "=== [$SPLIT] frames (4f grid) ==="
  python $DATA/extract_frame_train.py --split "$SPLIT" --num_frames 4 --resume

  echo "=== [$SPLIT] memory (strict cutoff + frame-aligned + future) ==="
  # ⚠ --legacy_cutoff 절대 금지. strict 가 기본. leakage report 가 수정 규모를 출력.
  python $DATA/extract_memory_train.py --split "$SPLIT" --future-k 5

  echo "=== [$SPLIT] assemble ==="
  python $DATA/assemble_train.py --split "$SPLIT"

  echo "=== [$SPLIT] convert (train jsonl + b0meta 분리) ==="
  if [ "$SPLIT" = "train" ]; then
    python $DATA/convert_to_train_format.py \
      --input  data/grpo_dataset/grpo_dataset.jsonl \
      --output data/grpo_dataset/grpo_train.jsonl
  else
    python $DATA/convert_to_train_format.py \
      --input  data/grpo_dataset/grpo_dataset_heldout.jsonl \
      --output data/grpo_dataset/grpo_heldout.jsonl
  fi
done

echo "=== 자동 leakage 검사 (freeze 게이트) ==="
python scripts/step2/check_leakage.py \
  --memory     data/grpo_dataset/memory_train.jsonl \
  --train_jsonl data/grpo_dataset/grpo_train.jsonl \
  --b0_meta    data/grpo_dataset/grpo_train_b0meta.jsonl \
  --csv        src/epic-kitchens-100-annotations/EPIC_100_train.csv \
  --selected   data/grpo_dataset/selected_train.jsonl

echo "[done] F0 v2 데이터 빌드 완료. leakage PASS 확인 후 학습 진행."
