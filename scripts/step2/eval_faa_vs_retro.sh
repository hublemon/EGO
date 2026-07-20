#!/usr/bin/env bash
# B0 — held-out FAA vs B0 비교 (핸드오프 §16~18: preference margin + relation 분해).
# GT accuracy / coherence 는 generation eval 로 별도 (evaluate_b0 의 순수 함수 사용).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="src:${PYTHONPATH:-}"

FAA_ADAPTER="${FAA_ADAPTER:?set FAA_ADAPTER}"
B0_ADAPTER="${B0_ADAPTER:?set B0_ADAPTER}"

python -m ego.step2_vlm_alignment.retro.evaluate_retro \
  --dpo_jsonl   data/grpo_dataset/b0_dpo_heldout.jsonl \
  --faa_adapter "$FAA_ADAPTER" \
  --b0_adapter  "$B0_ADAPTER" \
  --out         outputs/step2/b0_full_trace_dpo/faa_vs_b0.json

echo "[done] preference 비교 → outputs/step2/b0_full_trace_dpo/faa_vs_b0.json"
echo "  Go 판정: m_B0 > m_FAA (DIFFERENT subset) · SAME/SAME audit margin 과증가 없음"
