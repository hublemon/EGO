#!/usr/bin/env bash
# F0 final 요인 분해 평가. 프롬프트 포맷을 학습과 반드시 일치시킨다(--reward_mode).
set -euo pipefail
cd "$(dirname "$0")/../.."
HELDOUT="${EGO_HELDOUT_JSONL:?set EGO_HELDOUT_JSONL}"
OUT=outputs/step2/f0_final_wm_only/heldout_eval; mkdir -p "$OUT"
E="python src/ego/step2_vlm_alignment/evaluate.py --reward_mode wm_likelihood_joint --jsonl $HELDOUT --limit 500"

# 가치순: 산출물 → G1 → 요인분리 → 곡선
$E --model_name Qwen/Qwen3-VL-8B-Instruct   --adapter outputs/step2/f0_final_wm_only/checkpoint-500 --out "$OUT/step500.json"
$E --model_name Qwen/Qwen3-VL-8B-Instruct   --out "$OUT/step0.json"                     # 모델 효과 분리용
$E --model_name Qwen/Qwen2.5-VL-7B-Instruct --out "$OUT/qwen25_base_joint.json"         # 포맷 효과 분리용
