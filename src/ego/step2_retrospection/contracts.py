"""Strict temporal contract + Step-1 → Step-2 인터페이스 스키마.

Handoff 1 §3 / Handoff 2 §3 의 assertion 4종을 코드로 강제한다.
이 모듈은 파이프라인의 모든 생산자(build_support, build_context)와
소비자(base_trace, projection, train.*)가 공유하는 유일한 계약 지점이다.

시간 계약 (EXPORT_CONTRACT.md와 동일):
    관찰: [A2.start - 9s, A2.start - 1s]  (l_obs 8s, tau_a 1s, 32frames@4fps)
    정답: A2 (아직 시작하지 않은 다음 action) — anticipation, recognition 아님
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Support 크기 — 2026-07-22 사용자 확정 Top-10 (probe 실측 coverage@10 = 39.8% full val).
TOP_K: int = 10

# Future trajectory F_t 길이 — 2026-07-22 사용자 확정: 3~5 action 또는 24s 캡.
# 구현: target_start부터 최대 5개 action, start가 target_start+24s 안쪽인 것만.
FUTURE_MAX_ACTIONS: int = 5
FUTURE_MAX_SECONDS: float = 24.0

# probe taxonomy (export run_metadata 기준)
NUM_VERBS: int = 81
NUM_NOUNS: int = 140
NUM_ACTIONS: int = 293

TRACE_TAGS = ("reasoning", "task_belief", "action")


@dataclass(frozen=True)
class HistoryAction:
    """Decision point 이전에 *완료된* action (e_i <= t)."""
    verb: str
    noun: str
    start_sec: float
    stop_sec: float


@dataclass(frozen=True)
class SupportSample:
    """Step-1 probe가 넘기는 샘플 하나 — support jsonl의 한 줄.

    candidates는 이미 셔플된 상태로 저장한다 (likelihood/rank 비공개 원칙,
    Handoff 1 §4.1). wm_scores는 별도 필드에 보관하되 프롬프트에는 절대
    노출하지 않는다 (eval의 L0/G1/G2 분해에만 사용).
    """
    sample_id: str
    video_uid: str
    clip_uid: str
    obs_start_sec: float
    obs_end_sec: float          # = decision_time = target_start - 1.0
    target_start_sec: float
    gt_verb: str
    gt_noun: str
    candidates: list[str]       # 셔플된 Top-K "verb noun" 문자열
    wm_scores: list[float]      # candidates와 같은 순서. 프롬프트 노출 금지.
    history: list[HistoryAction] = field(default_factory=list)
    future: list[HistoryAction] = field(default_factory=list)  # offline 전용 (Handoff 1 §6.1)
    boundary_flag: bool = False

    @property
    def gt_action(self) -> str:
        return f"{self.gt_verb} {self.gt_noun}"

    @property
    def gt_in_support(self) -> bool:
        """False면 WM support failure — 학습에서 분리 (Handoff 1 §8)."""
        return self.gt_action in self.candidates


class ContractViolation(AssertionError):
    """Strict temporal contract 위반. 생산·소비 양쪽에서 raise."""


def assert_strict_contract(s: SupportSample, tau_a: float = 1.0, tau_exact: bool = True) -> None:
    """Handoff §3 assertion 4종. 위반 시 ContractViolation.

    데이터 생산 시점(build_support/build_context)과 소비 시점(base_trace,
    projection, train.*) 양쪽에서 호출한다 — 한쪽만 믿지 않는다.

    tau_exact: start−1s 계약(retro3)은 gap ≡ tau_a 를 강제.
    end−1s(A2.end−1s, strict-next A3) 계약(retro4/Phase-1 prior)은 horizon이
    가변(≥ tau_a)이므로 False로 호출 — 최소 gap만 검증한다.
    """
    if not s.obs_end_sec < s.target_start_sec:
        raise ContractViolation(
            f"{s.sample_id}: observation_end({s.obs_end_sec}) >= target_start({s.target_start_sec})")
    gap = s.target_start_sec - s.obs_end_sec
    if tau_exact and abs(gap - tau_a) > 1e-3:
        raise ContractViolation(
            f"{s.sample_id}: anticipation gap != {tau_a}s "
            f"(obs_end={s.obs_end_sec}, target_start={s.target_start_sec})")
    if not tau_exact and gap < tau_a - 1e-3:
        raise ContractViolation(
            f"{s.sample_id}: anticipation gap({gap}) < 최소 {tau_a}s")
    for a in s.history:
        if a.stop_sec > s.obs_end_sec:
            raise ContractViolation(
                f"{s.sample_id}: history action stops({a.stop_sec}) after decision time({s.obs_end_sec})")
    for a in s.future:
        if a.start_sec < s.target_start_sec:
            raise ContractViolation(
                f"{s.sample_id}: future action starts({a.start_sec}) before target_start({s.target_start_sec})")
    if len(s.candidates) != TOP_K:
        raise ContractViolation(f"{s.sample_id}: |candidates|={len(s.candidates)} != TOP_K({TOP_K})")
    if len(s.wm_scores) != len(s.candidates):
        raise ContractViolation(f"{s.sample_id}: wm_scores/candidates length mismatch")
