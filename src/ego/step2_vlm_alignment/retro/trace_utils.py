"""trace_utils.py — full-trace 파싱/정규화 (dependency-free).

F0(train_grpo_action.py)의 검증된 파싱 규칙을 그대로 옮긴다 — torch/trl 을 끌어오지 않도록
정규식 파서만 복제(judge_reasoning.py 와 동일한 이유). B0 는 <reasoning>/<task_belief>/<action>
세 태그를 하나의 완결 trace 로 다룬다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

RE_REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
RE_BELIEF = re.compile(r"<task_belief>(.*?)</task_belief>", re.DOTALL)
RE_ACTION = re.compile(r"<action>(.*?)</action>", re.DOTALL)


@dataclass
class Trace:
    reasoning: str
    belief: str
    verb: Optional[str]
    noun: Optional[str]
    raw: str

    @property
    def action_str(self) -> str:
        if self.verb and self.noun:
            return f"{self.verb} {self.noun}"
        return ""

    def is_complete(self) -> bool:
        """세 태그가 모두 파싱되고 action 이 유효한가."""
        return bool(self.reasoning and self.belief and self.verb and self.noun)


def _first(pat: re.Pattern, text: str) -> str:
    m = pat.search(text or "")
    return m.group(1).strip() if m else ""


def parse_action(text: str) -> tuple[Optional[str], Optional[str]]:
    """<action>{"verb","noun"}</action> → (verb, noun). 실패 시 (None, None)."""
    block = _first(RE_ACTION, text)
    if not block:
        return None, None
    m = re.search(r"\{.*\}", block, re.DOTALL)
    if not m:
        return None, None
    try:
        o = json.loads(m.group(0))
        v = o.get("verb")
        n = o.get("noun")
        v = v.strip() if isinstance(v, str) and v.strip() else None
        n = n.strip() if isinstance(n, str) and n.strip() else None
        return v, n
    except Exception:
        return None, None


def parse_full_trace(text: str) -> Trace:
    """completion 문자열 → Trace. 세 태그를 각각 추출."""
    v, n = parse_action(text)
    return Trace(
        reasoning=_first(RE_REASONING, text),
        belief=_first(RE_BELIEF, text),
        verb=v, noun=n, raw=text or "",
    )


def canonical_action(verb: Optional[str], noun: Optional[str]) -> str:
    """EK100 canonical 비교키. noun 의 계층 표기(board:chopping)는 소문자 정규화만.
    (verb/noun class id 가 있으면 그쪽이 정답 — 여기선 문자열 폴백)."""
    v = (verb or "").strip().lower()
    n = (noun or "").strip().lower()
    return f"{v}|{n}"


def build_full_trace(reasoning: str, belief: str, verb: str, noun: str) -> str:
    """정규 full-trace 직렬화 (chosen/rejected 동일 포맷 — no-splicing 계약).
    F0 출력 계약과 태그·순서 동일."""
    action = json.dumps({"verb": verb, "noun": noun}, ensure_ascii=False)
    return (f"<reasoning>\n{reasoning.strip()}\n</reasoning>\n"
            f"<task_belief>{belief.strip()}</task_belief>\n"
            f"<action>{action}</action>")


# 미래 개념 누설 스크리닝용 어휘 (projection 규칙 위반 감지 — 보조 지표, 판정 아님)
FUTURE_LEAK_MARKERS = [
    "because that is what actually happens", "actually happens next",
    "future actions show", "in the future", "will happen", "later they",
    "eventually", "the video shows later", "as we see next",
]


def has_future_leak_language(reasoning: str) -> bool:
    """reasoning 에 '미래를 안다'는 직접 표현이 있는가 (projection 금지 규칙 위반 스크리닝)."""
    r = (reasoning or "").lower()
    return any(m in r for m in FUTURE_LEAK_MARKERS)
