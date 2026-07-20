"""route_pairs.py — pair routing / SAME-SAME drop / candidate support (순수 로직).

핸드오프 §8, §10, §13 을 그대로 구현. 이 파일은 model/GPU 를 전혀 쓰지 않는다 —
belief_relation(teacher equivalence 결과)과 action_relation(canonical 동치)만 입력받아
KEEP / DROP / AUDIT 를 결정한다. B0 의 검증 가능한 핵심 로직이라 smoke 로 단언한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .trace_utils import canonical_action

BeliefRelation = Literal["SAME", "DIFFERENT", "UNCERTAIN"]
ActionRelation = Literal["SAME", "DIFFERENT"]
RouteDecision = Literal["KEEP", "KEEP_TAG", "DROP_SAME_SAME", "DROP_UNCERTAIN_SAME"]


@dataclass
class Routed:
    decision: RouteDecision
    training: bool                 # 학습 pair 로 emit 하는가
    audit: bool                    # audit/eval 로 보존하는가
    drop_reason: str = ""


# 핸드오프 §10 routing table.
#   DIFFERENT × DIFFERENT → KEEP       (belief/action preference)
#   DIFFERENT × SAME      → KEEP       (reasoning/belief refinement)
#   SAME      × DIFFERENT → KEEP       (action/full-trace refinement)
#   SAME      × SAME      → DROP       (projector style 방지, audit 보존)
#   UNCERTAIN × DIFFERENT → KEEP + tag (action signal 존재)
#   UNCERTAIN × SAME      → DROP/audit (semantic preference 불확실)
def route(belief_relation: BeliefRelation, action_relation: ActionRelation) -> Routed:
    b, a = belief_relation, action_relation
    if b == "SAME" and a == "SAME":
        # 핵심 drop 조건: g_FAA ≡ g_proj ∧ a_FAA = a_GT → semantic tie
        return Routed("DROP_SAME_SAME", training=False, audit=True, drop_reason="semantic_tie")
    if b == "UNCERTAIN" and a == "SAME":
        return Routed("DROP_UNCERTAIN_SAME", training=False, audit=True,
                      drop_reason="uncertain_semantic_preference")
    if b == "UNCERTAIN" and a == "DIFFERENT":
        return Routed("KEEP_TAG", training=True, audit=False)   # action signal 존재
    # (DIFFERENT, *) 는 전부 KEEP
    return Routed("KEEP", training=True, audit=False)


def action_relation(faa_verb, faa_noun, gt_verb, gt_noun) -> ActionRelation:
    """canonical EK100 label 동치 (핸드오프 §8)."""
    return "SAME" if canonical_action(faa_verb, faa_noun) == canonical_action(gt_verb, gt_noun) \
        else "DIFFERENT"


def gt_in_candidates(gt_verb, gt_noun, candidates: list[dict]) -> bool:
    """candidate support (핸드오프 §13): a_GT ∈ D_t.
    candidates: [{"verb","noun"}, ...]. canonical 비교."""
    key = canonical_action(gt_verb, gt_noun)
    return any(canonical_action(c.get("verb"), c.get("noun")) == key for c in candidates)


@dataclass
class RoutingStats:
    kept: int = 0
    kept_tag: int = 0
    dropped_same_same: int = 0
    dropped_uncertain_same: int = 0
    gt_outside_candidates: int = 0
    projection_failures: int = 0

    def as_log(self) -> dict:
        return {
            "data/num_kept_pairs": self.kept + self.kept_tag,
            "data/num_kept_tag": self.kept_tag,
            "data/num_same_same_dropped": self.dropped_same_same,
            "data/num_uncertain_same_dropped": self.dropped_uncertain_same,
            "data/num_gt_outside_candidates": self.gt_outside_candidates,
            "data/num_projection_failures": self.projection_failures,
        }

    def account(self, r: Routed) -> None:
        if r.decision == "KEEP":
            self.kept += 1
        elif r.decision == "KEEP_TAG":
            self.kept_tag += 1
        elif r.decision == "DROP_SAME_SAME":
            self.dropped_same_same += 1
        elif r.decision == "DROP_UNCERTAIN_SAME":
            self.dropped_uncertain_same += 1
