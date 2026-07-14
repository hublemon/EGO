# Step 2 Pre-baseline — Qwen2.5-VL-7B-Instruct vs Gemini (2026-05-28)

> 원본: `docs/RESULTS.md` §13 (Phase 3 frontier-VLM 종합 문서 내 Step 2 직전 섹션만 발췌).
> GRPO 파인튜닝 대상 소형 모델(Qwen2.5-VL-7B)이 **무학습 상태**에서 어느 수준인지 측정한 baseline.

## 모델 선택 정정
- 회의록은 `Qwen2.5-7B-Instruct` 권장이나 이건 **텍스트 전용**. 파이프라인은 multi-frame 이미지 입력 필수.
- 비전 가능한 **`Qwen2.5-VL-7B-Instruct`** 사용 (HF, ~16GB).
- 로컬 추론 (`src/qwen_runner.py`) — transformers + bf16 + H200, vLLM 미사용 (설치 리스크 회피). `evaluate.py`에 `--backend qwen` 플래그로 드롭인.

## n=10 비교 (think-format, multi-frame, ap=0.0, 동일 10 샘플)

| 지표 | Gemini-2.0-flash | **Qwen2.5-VL-7B (un-finetuned)** | Δ |
|---|---|---|---|
| valid | 10/10 (100%) | 8/10 (80%) — fallback 2 | -20pp |
| verb✓ | 60% | **70%** | +10pp |
| noun✓ | 60% | **80%** | +20pp |
| **action✓** | **40%** | **60%** | **+20pp** |
| follows_WM | 40% | 50% | +10pp |

## Qwen이 Gemini보다 정답에 가까운 케이스 (3건)
| sample | GT | Gemini | Qwen |
|---|---|---|---|
| P01_14_217 | take spatula | move spatula | take spatula ✓ |
| P01_15_195 | put lid | put plate | put lid ✓ |
| P01_15_205 | wash colander | wash pan | wash colander ✓ |

## Qwen fallback 2건 (운 좋은 일치)
- **P01_14_217**: reasoning 양호하나 `<action>stir spatula</action>` 출력 — `stir` 후보 외 → fallback → top-1(`take spatula`) 우연히 GT.
- **P01_15_56**: reasoning이 800 토큰 한계에 잘려 `<action>` 누락 → fallback → top-1(`wash knife`) 우연히 GT.
- 두 케이스 모두 GT∈cands 100%라 top-1 fallback이 운 좋게 맞은 것. 보수적 해석 시 Qwen "strict" 4/10 = Gemini 4/10.

## 정직한 결론
- Raw action 정확도: Qwen +20pp (60% vs 40%).
- "Strict" (fallback 제외) 동률 (4/10 vs 4/10).
- **Reasoning 품질**: Qwen의 Step 1~3 매우 상세 (multi-frame 모션 변화까지 명시적 묘사). Gemini와 비등 또는 우위.
- **약점**: 포맷 준수율 80% (fallback 20%). out-of-vocab verb (`stir`, `pick-up`) 사용 + max_tokens 컷오프.
- GRPO Step 2에서 reward에 "format compliance"와 "candidate-only output"을 강하게 넣으면 즉시 회복 가능. **un-fine-tuned 시점에서 이 정도면 GRPO 후 Gemini 상회 가능성 매우 큼** — 이 가설이 이후 GRPO 실험(`GRPO_TRAINING_LOG.md`)의 출발점.

## 다음 스텝 (Step 2 착수 직전 계획했던 것)
1. n=30/50 확장 — fallback 확률 안정화, 통계적 비교
2. Qwen 출력 포맷 보강 — system prompt에 "MUST use only candidate verbs" 강화 + few-shot
3. GRPO reward 설계 — (a) action_correct (b) format compliance (c) candidate-only
