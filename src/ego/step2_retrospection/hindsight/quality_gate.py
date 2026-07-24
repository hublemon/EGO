"""규칙 기반 quality gate — gemini 호출 전에 최대한 걸러낸다 (Handoff 2 §6).

정답이 확인 가능한 판정만 담당: future leakage(어형 변화 포함) · non-restatement ·
temporal correctness(패턴) · malformed/길이. 의미 판단(grounding·specificity·
동등성)은 semantic_gate(gemini)로.

정책: drop-not-patch. 사유 리스트를 반환 — 비어 있으면 통과.
"""
from __future__ import annotations

import re

_SUFFIX = re.compile(r"_\([^)]*\)")


def serialize_trace(reasoning: str, task_belief: str, action: str) -> str:
    """y+/y- 공용 정규 직렬화 — 포맷 지름길 차단의 단일 지점."""
    return (
        f"<reasoning>\n{reasoning.strip()}\n</reasoning>\n\n"
        f"<task_belief>\n{task_belief.strip()}\n</task_belief>\n\n"
        f"<action>\n{action.strip()}\n</action>"
    )


def _word_forms(token: str) -> set[str]:
    """어형 변화 근사 (goal_leaks 아이디어 승계, 독립 구현): add→adds/adding/added."""
    t = _SUFFIX.sub("", token.lower()).strip("_ ")
    if not t or len(t) < 3:
        return {t} if t else set()
    forms = {t, t + "s", t + "es", t + "ing", t + "ed", t + "d"}
    if t.endswith("e"):
        forms |= {t[:-1] + "ing", t + "d"}
    if len(t) > 3 and t[-1] not in "aeiou" and t[-2] in "aeiou" and t[-3] not in "aeiou":
        forms |= {t + t[-1] + "ing", t + t[-1] + "ed"}  # cut→cutting
    return forms


def _contains_any(text: str, tokens: set[str]) -> bool:
    words = set(re.findall(r"[a-z_()]+", text.lower()))
    words = {_SUFFIX.sub("", w).strip("_") for w in words}
    return bool(words & tokens)


def _pair_mentioned(text: str, verb: str, noun: str) -> bool:
    """verb와 noun의 어형이 모두 등장하면 그 action이 언급된 것으로 본다."""
    return _contains_any(text, _word_forms(verb)) and _contains_any(text, _word_forms(noun))


_TEMPORAL_BAD = re.compile(
    r"\b(is|are|am)\s+(currently\s+|now\s+|already\s+)?\w+ing\b.{0,40}\b(finish|complet|done)\w*\b"
    r"|\b(has|have)\s+(already\s+)?(just\s+)?(start|begun|began|finish|complet)\w*\s+to?\b",
    re.I)


def check_chosen(reasoning: str, task_belief: str, rec: dict) -> list[str]:
    """chosen(y+) 게이트. rec: context 레코드 (gt_*, future, candidates)."""
    reasons: list[str] = []
    r, b = reasoning or "", task_belief or ""
    if len(r.split()) < 15:
        reasons.append("reasoning_too_short")
    if not b or len(b.split()) < 3:
        reasons.append("belief_too_short")

    # 6.5 non-restatement: belief에 GT verb+noun 동시 등장 금지
    if _pair_mentioned(b, rec["gt_verb"], rec["gt_noun"]):
        reasons.append("restatement")

    # 6.2 no future leakage: target 이후 미래 행동(future[1:])의 (verb,noun) 언급 금지.
    # 단 후보 목록에 이미 있는 action은 reasoning의 정당한 후보 비교일 수 있어 belief만 검사.
    cand_set = set(rec["candidates"])
    for a in rec["future"][1:]:
        pair = f"{a['verb']} {a['noun']}"
        if _pair_mentioned(b, a["verb"], a["noun"]):
            reasons.append("future_leak_belief")
            break
        if pair not in cand_set and _pair_mentioned(r, a["verb"], a["noun"]):
            reasons.append("future_leak_reasoning")
            break

    # 6.3 temporal: target을 진행/완료로 서술하는 패턴
    if _pair_mentioned(r, rec["gt_verb"], rec["gt_noun"]) and _TEMPORAL_BAD.search(r):
        reasons.append("temporal_bad")

    return reasons


def base_trace_usable(bt: dict) -> bool:
    """rejected(y-)로 쓸 수 있는 base trace인가 — parse 실패는 '너무 쉬운 negative'(§7.3)."""
    return not bt.get("malformed") and bool(bt.get("reasoning")) and bool(bt.get("task_belief"))


def belief_token_jaccard(b1: str, b2: str) -> float:
    tok = lambda s: {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2}  # noqa: E731
    s1, s2 = tok(b1), tok(b2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)
