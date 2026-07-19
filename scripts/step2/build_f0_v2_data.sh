#!/usr/bin/env bash
# F0 final v2 데이터 빌드 — strict cutoff + 4프레임 + L2-c 정렬 + B0 메타 분리.
# 계획: docs/experiments/2026-07-18_f0_final_plan.md
# 선행: selected_{train,heldout}.jsonl, predictions_{train,heldout}.jsonl (기존 파이프라인)
#
# 경로: 데이터 루트는 EGO_ROOT (기본 ~/work/jihun/EGO — 레이아웃이 다르면 export EGO_ROOT=...).
#       리포 클론 위치와 데이터 루트가 달라도 동작한다 (python 스크립트도 EGO_ROOT 를 읽음).
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=src/ego/step2_vlm_alignment/data
export EGO_ROOT="${EGO_ROOT:-$HOME/work/jihun/EGO}"
GD="$EGO_ROOT/data/grpo_dataset"
ANNOT="$EGO_ROOT/src/epic-kitchens-100-annotations"
echo "[env] EGO_ROOT=$EGO_ROOT  EK100_VIDEOS=${EK100_VIDEOS:-<unset>}"

for SPLIT in train validation; do
  echo "=== [$SPLIT] frames (4f grid) ==="
  # --resume 은 기하 검사 포함: v1 1f 잔존물은 건너뛰지 않고 4f 로 재추출된다
  python $DATA/extract_frame_train.py --split "$SPLIT" --num_frames 4 --resume

  echo "=== [$SPLIT] memory (strict cutoff + frame-aligned + future) ==="
  # ⚠ --legacy_cutoff 절대 금지. strict 가 기본. leakage report 가 수정 규모를 출력.
  python $DATA/extract_memory_train.py --split "$SPLIT" --future-k 5

  echo "=== [$SPLIT] assemble ==="
  python $DATA/assemble_train.py --split "$SPLIT"

  echo "=== [$SPLIT] convert (train jsonl + b0meta 분리) ==="
  if [ "$SPLIT" = "train" ]; then
    python $DATA/convert_to_train_format.py \
      --input  "$GD/grpo_dataset.jsonl" \
      --output "$GD/grpo_train.jsonl"
  else
    python $DATA/convert_to_train_format.py \
      --input  "$GD/grpo_dataset_heldout.jsonl" \
      --output "$GD/grpo_heldout.jsonl"
  fi
done

echo "=== 자동 leakage 검사 (freeze 게이트) — train ==="
python scripts/step2/check_leakage.py \
  --memory     "$GD/memory_train.jsonl" \
  --train_jsonl "$GD/grpo_train.jsonl" \
  --b0_meta    "$GD/grpo_train_b0meta.jsonl" \
  --csv        "$ANNOT/EPIC_100_train.csv" \
  --selected   "$GD/selected_train.jsonl"

echo "=== 자동 leakage 검사 (freeze 게이트) — heldout ==="
python scripts/step2/check_leakage.py \
  --memory     "$GD/memory_heldout.jsonl" \
  --train_jsonl "$GD/grpo_heldout.jsonl" \
  --b0_meta    "$GD/grpo_heldout_b0meta.jsonl" \
  --csv        "$ANNOT/EPIC_100_validation.csv" \
  --selected   "$GD/selected_heldout.jsonl"

echo "[done] F0 v2 데이터 빌드 완료. leakage PASS(양 split) 확인 후 학습 진행."
