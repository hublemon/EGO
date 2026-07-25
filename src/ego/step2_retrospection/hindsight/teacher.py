"""Hindsight teacher Ψ: future trajectory → 당시 procedural structure (Handoff 1 §6.2).

텍스트 전용 Qwen 호출 (frozen). F_t만 입력 — h_t(json)를 반환한다.
h_t는 projection Φ의 입력일 뿐 학습 target이 아니며 trace에 그대로 들어가지 않는다.
"""
from __future__ import annotations

import json
import re

TEACHER_SYSTEM = (
    "You are an activity-analysis assistant. You are given the sequence of actions a person "
    "performed AFTER a decision point in a cooking-style egocentric video. Infer what local "
    "procedure was underway AT the decision point.\n"
    "Return ONLY a JSON object with keys:\n"
    '{"activity": str, "stage": str, "completed_subgoal": str, "next_subgoal": str, '
    '"hypotheses": [str, ...], "uncertainty": "low"|"medium"|"high"}\n'
    "Keep each value short (<=12 words). 'next_subgoal' is the immediate procedural step, "
    "NOT a verbatim action label."
)


def teacher_messages(rec: dict) -> list[dict]:
    fut = rec["future"]
    lines = [f"{i + 1}. {a['verb']} {a['noun']}" for i, a in enumerate(fut)]
    txt = ("Actions performed after the decision point, in order:\n" + "\n".join(lines) +
           "\n\nInfer the JSON now.")
    return [
        {"role": "system", "content": [{"type": "text", "text": TEACHER_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": txt}]},
    ]


def parse_h(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        h = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    need = {"activity", "stage", "completed_subgoal", "next_subgoal", "hypotheses", "uncertainty"}
    if not need.issubset(h):
        return None
    return {k: h[k] for k in need}
