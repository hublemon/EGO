"""Step 2 Retrospection — non-parametric Prospection + projected-hindsight Retrospection.

EGO_jihun3 신규 트랙. 기존 step2_vlm_alignment(구 F0/B0)와 완전 독립 — 그 패키지를 import하지 않는다.

방법론 문서:
- docs/experiments/2026-07-22_nonparametric_prospection_projected_trace_retrospection_handoff.md (Handoff 1: R1 SFT + R2)
- docs/experiments/2026-07-22_nonparametric_prospection_dpo_retrospection_handoff.md (Handoff 2: field-balanced DPO)
- docs/experiments/2026-07-22_jihun3_retrospection_kickoff_handoff.md (착수 문서: 인터페이스 계약·사전 등록)

파이프라인 (두 방법론 공유 → 마지막에 분기):
    data.build_support   Step-1 probe checkpoint → Top-K support 덤프
    data.build_context   GoalStep annotation → history H<t / future F_t
    prospection.base_trace  Base Qwen zero-shot trace (y-)
    hindsight.teacher    Ψ: future trajectory → task structure
    hindsight.projection Φ: 과거 evidence 수준 재작성 (y+)
    hindsight.quality_gate / semantic_gate   게이트 (규칙 → gemini)
    pairs.build_pairs    DPO pair 구성 (Handoff 2 전용)
    train.sft_r1 / consistency_r2 / dpo_fb   3-arm 학습
    eval.battery / intervention              평가·개입 테스트
"""
