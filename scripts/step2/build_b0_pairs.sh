#!/usr/bin/env bash
# B0 — offline full-trace DPO pair 구축.
# 핸드오프 §26. 선행: freeze 된 FAA adapter + F0 grpo_{split}.jsonl + *_b0meta.jsonl
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="src:${PYTHONPATH:-}"

FAA_ADAPTER="${FAA_ADAPTER:?set FAA_ADAPTER (frozen FAA LoRA path)}"
SPLIT="${1:-train}"          # train | heldout
GRPO_JSONL="data/grpo_dataset/grpo_${SPLIT}.jsonl"
B0META="data/grpo_dataset/grpo_${SPLIT}_b0meta.jsonl"

echo "=== [1/3] frozen FAA online full-trace rollout ($SPLIT) ==="
python -m ego.step2_vlm_alignment.b0.generate_faa_traces \
  --faa_adapter "$FAA_ADAPTER" \
  --train_jsonl "$GRPO_JSONL" \
  --out "data/grpo_dataset/faa_traces_${SPLIT}.jsonl" \
  --num_generations 4 --temperature 1.0

echo "=== [2/3] 병합 (faa_traces + b0meta → samples) ==="
python -m ego.step2_vlm_alignment.b0.merge_b0_samples \
  --faa_traces "data/grpo_dataset/faa_traces_${SPLIT}.jsonl" \
  --b0meta     "$B0META" \
  --out        "data/grpo_dataset/b0_samples_${SPLIT}.jsonl"

echo "=== [3/3] projection + equivalence + routing → DPO pairs ==="
python -m ego.step2_vlm_alignment.b0.build_dpo_dataset \
  --samples   "data/grpo_dataset/b0_samples_${SPLIT}.jsonl" \
  --out_train "data/grpo_dataset/b0_dpo_${SPLIT}.jsonl" \
  --out_audit "data/grpo_dataset/b0_audit_${SPLIT}.jsonl"

echo "=== 데이터셋 재검증 (leakage / no-splicing) ==="
python -m ego.step2_vlm_alignment.b0.validate_cli \
  --dpo "data/grpo_dataset/b0_dpo_${SPLIT}.jsonl"

echo "[done] B0 pairs ($SPLIT)."
