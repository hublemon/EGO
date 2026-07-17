"""B0 — Full-Trace Projected-Hindsight DPO.

핸드오프: EGO_STEP2_B0_FULL_TRACE_DPO_VALIDATION_HANDOFF (2026-07-18)
구현 노트: docs/experiments/2026-07-18_b0_implementation.md

MVP 핵심: reasoning/belief/action 을 **분리하지 않는다**. 전체 trace 를 하나의 의미 단위로
유지하고, 시점 t 로 projection 한 coherent hindsight trace 를 chosen, frozen FAA online trace 를
rejected 로 쓰는 sequence-level DPO.

모듈:
  trace_utils          — full-trace 파싱/정규화 (dependency-free, 순수)
  route_pairs          — routing table · SAME/SAME drop · candidate support (순수 로직)
  validate_dpo_dataset — leakage / no-splicing assertions (순수 로직)
  teacher              — raw hindsight · projection · equivalence (frozen base VLM)
  build_dpo_dataset    — offline pair 오케스트레이션 (teacher 주입 가능)
  generate_faa_traces  — frozen FAA online full-trace rollout (GPU)
  train_b0_dpo         — TRL DPOTrainer wrapper
  evaluate_b0          — held-out preference · GT accuracy · coherence
"""
