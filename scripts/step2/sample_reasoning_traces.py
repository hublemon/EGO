#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sample_reasoning_traces.py — completion_samples.jsonl 에서 리즈닝 트레이스를 step 구간별로 추출.

목적: F/B 회의(2026-07-17) 액션 아이템 "리즈닝 트레이스 정성 검토(0→100→200 step)".
judge 곡선(judge_reasoning.py)이 보여준 선택적 위축 — candidate_review ↑ / belief_globality·
history_grounding ↓ — 이 실제 텍스트에서 어떻게 나타나는지 눈으로 확인하기 위한 도구다.

학습 서버에서 실행 (completion_samples.jsonl 은 리포에 커밋하지 않는다):
  python scripts/step2/sample_reasoning_traces.py --run_dir runs/f0_final
  python scripts/step2/sample_reasoning_traces.py --run_dir runs/f0_final \
      --steps 1,100,200,350,476 --per_step 2 --out traces_review.md

출력: 마크다운 1개 — step 마다 [history / 후보 5개 / reasoning / task_belief / action] 원문과
      간단 지표(단어 수, history 토큰 참조 여부, belief 가 선택 action 의 재진술인지 휴리스틱).
휴리스틱은 스크리닝용 참고치일 뿐이며 판단은 원문으로 한다 (judge 루브릭 §belief_globality).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# judge_reasoning.py 와 동일한 추출 규칙 (검증된 코드는 수정하지 않고 규칙만 복제)
RE_THINK = re.compile(r"<(?:reasoning|think)>(.*?)</(?:reasoning|think)>", re.DOTALL)
RE_BELIEF = re.compile(r"<task_belief>(.*?)</task_belief>", re.DOTALL)
RE_ACTION = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def extract(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text or "")
    return m.group(1).strip() if m else ""


def parse_action(text: str) -> tuple[str, str]:
    block = extract(RE_ACTION, text)
    m = re.search(r"\{.*\}", block, re.DOTALL)
    if not m:
        return "", ""
    try:
        o = json.loads(m.group(0))
        return str(o.get("verb", "")), str(o.get("noun", ""))
    except Exception:
        return "", ""


def history_token_overlap(reasoning: str, memory_context: str) -> int:
    """history 라벨("verb noun" 줄)의 토큰이 reasoning 에 등장하는 개수 (스크리닝용)."""
    r = reasoning.lower()
    tokens = set()
    for line in (memory_context or "").splitlines():
        for tok in re.findall(r"[a-z]{3,}", line.lower()):
            tokens.add(tok)
    return sum(1 for t in tokens if t in r)


def belief_is_restatement(belief: str, verb: str, noun: str) -> bool:
    """belief_globality=0 패턴 스크리닝: belief 가 선택한 action 의 재진술인가."""
    b = belief.lower()
    return bool(verb) and bool(noun) and verb.lower() in b and noun.lower().split(":")[0] in b


def pick_steps(available: list[int], requested: str | None, n_auto: int) -> list[int]:
    if requested:
        want = [int(s) for s in requested.split(",")]
        return [min(available, key=lambda a: abs(a - w)) for w in want]
    if len(available) <= n_auto:
        return available
    idxs = [round(i * (len(available) - 1) / (n_auto - 1)) for i in range(n_auto)]
    return [available[i] for i in idxs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", default=None,
                    help="쉼표 구분 step 목록 (가장 가까운 로그 step 으로 스냅). 기본: 균등 5지점")
    ap.add_argument("--n_auto", type=int, default=5)
    ap.add_argument("--per_step", type=int, default=3, help="step 당 표시할 생성 개수")
    ap.add_argument("--out", default=None, help="기본: <run_dir>/traces_review.md")
    args = ap.parse_args()

    src = Path(args.run_dir) / "completion_samples.jsonl"
    if not src.exists():
        raise SystemExit(f"없음: {src} — 학습 서버의 run 디렉토리에서 실행해야 한다.")

    by_step: dict[int, dict] = {}
    for line in src.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if isinstance(d.get("step"), int):
            by_step[d["step"]] = d   # 같은 step 이 여러 번이면 마지막 기록 사용

    steps = pick_steps(sorted(by_step), args.steps, args.n_auto)
    out_path = Path(args.out) if args.out else Path(args.run_dir) / "traces_review.md"

    lines = ["# 리즈닝 트레이스 정성 검토 샘플",
             "",
             f"원본: `{src}` · step {steps} · step 당 {args.per_step}개",
             "",
             "지표는 스크리닝용 휴리스틱 — 판정은 원문 + judge 루브릭으로 한다.",
             ""]
    for step in steps:
        d = by_step[step]
        hist = d.get("memory_context") or d.get("prompt_tail") or "(없음)"
        cands = d.get("topk_actions_display") or "(기록 안 됨)"
        lines += [f"## step {step}", "", "**Action history (프롬프트에 실제로 보인 것):**",
                  "```", str(hist).strip(), "```", "",
                  f"**후보 5개 (셔플·점수 숨김):** `{cands}`", ""]
        comps = d.get("completions") or []
        for i, c in enumerate(comps[: args.per_step], 1):
            txt = c if isinstance(c, str) else (c.get("text") or c.get("content") or "")
            reasoning = extract(RE_THINK, txt)
            belief = extract(RE_BELIEF, txt)
            verb, noun = parse_action(txt)
            n_words = len(reasoning.split())
            overlap = history_token_overlap(reasoning, str(hist))
            restate = belief_is_restatement(belief, verb, noun)
            lines += [f"### 생성 {i} — {n_words}단어 · history 토큰 참조 {overlap}개 · "
                      f"belief 재진술 의심: {'예' if restate else '아니오'}",
                      "", "```", reasoning or "(reasoning 파싱 실패)", "```", "",
                      f"- **task_belief**: {belief or '(없음)'}",
                      f"- **action**: {verb} {noun}".rstrip(), ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장 → {out_path}  (step {steps})")


if __name__ == "__main__":
    main()
