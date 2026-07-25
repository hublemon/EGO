"""Semantic gate — gemini-2.5-pro 의미 판정 (규칙 게이트 통과 pair 후보만).

judge 정책 (2026-07-22 확정): 성공 판정 금지, pair 품질 게이트 전용.
LETSUR_API_KEY 없으면 전체를 "skipped"로 기록하고 넘어간다 (체인 경고 마커) —
pair 빌더는 규칙 판정만으로 진행하되 semantic="skipped" 플래그를 남긴다.

판정 항목 (pair당 1콜, 구조화 json):
  belief_equivalent: base belief ≡ projected belief 인가 (동등하면 pair 기각)
  style_only: 차이가 문체/길이뿐인가 (그렇다면 기각)
  chosen_grounded: projected reasoning이 관측·history 서술과 정합하는가 (6.1 근사)
  belief_restates_action: 미묘한 paraphrase-restatement (6.5 보강)

사용: PYTHONPATH=src python -m ego.step2_retrospection.hindsight.semantic_gate [--limit N]
출력: runs/retro3/data/semantic_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re

from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, read_jsonl, runs_root, write_marker

JUDGE_MODEL = "gemini-2.5-pro"
BASE_URL = "https://gw.letsur.ai/v1"

PROMPT = """Two rationales were written for the same egocentric-video decision point (choosing the next action).
A = written live (model's own guess).  B = written with hindsight, projected back to decision time.
The actual next action was: "{gt}".

A.belief: {a_belief}
A.reasoning: {a_reasoning}
B.belief: {b_belief}
B.reasoning: {b_reasoning}

Answer ONLY a JSON object:
{{"belief_equivalent": true/false,   // do A.belief and B.belief express the same procedural belief?
 "style_only": true/false,           // is the ONLY difference wording/length/style?
 "chosen_grounded": true/false,      // is B.reasoning internally consistent and plausibly grounded at decision time (no hindsight certainty)?
 "belief_restates_action": true/false}} // does B.belief just paraphrase "{gt}"?"""


def _api_key() -> str | None:
    # 실행 중 체인에 env 수정 없이 키를 주입하는 통로 (overrides.json과 동일 사상).
    # 우선순위: env > runs/retro3/.letsur_key (chmod 600)
    k = os.environ.get("LETSUR_API_KEY")
    if not k:
        p = runs_root() / ".letsur_key"
        if p.is_file():
            k = p.read_text().strip() or None
    return k


def get_client():
    import openai
    return openai.OpenAI(base_url=BASE_URL, api_key=_api_key())


def judge_one(client, payload: dict) -> dict | None:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(**payload)}])
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
        return {k: bool(j[k]) for k in
                ("belief_equivalent", "style_only", "chosen_grounded", "belief_restates_action")}
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16,
                    help="동시 judge 콜 수 (API-only — GPU/RAM 무관)")
    args = ap.parse_args()

    data_dir = runs_root() / "data"
    chosen = {r["sample_id"]: r for r in read_jsonl(data_dir / "chosen_train.jsonl") if r.get("gate") == "pass"}
    base = {r["sample_id"]: r for r in read_jsonl(data_dir / "base_trace_train.jsonl") if not r.get("malformed")}
    ids = sorted(set(chosen) & set(base))
    if args.limit:
        ids = ids[: args.limit]

    out_path = data_dir / "semantic_train.jsonl"
    done = {r["sample_id"] for r in read_jsonl(out_path)}
    todo = [i for i in ids if i not in done]

    if not _api_key():
        write_marker("S4_SEMANTIC_SKIPPED", {"reason": "LETSUR_API_KEY not set", "pending": len(todo)})
        print(f"[S4] SKIPPED — LETSUR_API_KEY 없음. pending={len(todo)} (pair는 규칙 판정만으로 진행)")
        return

    client = get_client()
    sw = StatusWriter("S4_semantic", total=len(ids))
    sw.update(done=len(done), force=True)
    n_err = 0

    def judge_sid(sid: str) -> dict:
        c, b = chosen[sid], base[sid]
        payload = {"gt": c["gt"], "a_belief": b["task_belief"], "a_reasoning": b["reasoning"],
                   "b_belief": c["task_belief"], "b_reasoning": c["reasoning"]}
        verdict = None
        for _ in range(2):  # 재시도 1회
            try:
                verdict = judge_one(client, payload)
            except Exception:
                verdict = None
            if verdict:
                break
        if verdict is None:
            return {"sample_id": sid, "error": True}  # 판정 불능 = 기각 (§7.3)
        return {"sample_id": sid, **verdict}

    # gemini-2.5-pro 콜당 ~9s → 직렬 3.2k샘플 = 8h+. 판정은 샘플별 독립이라 스레드 병렬.
    # append/status는 메인 스레드 단독 (ex.map이 제출 순서대로 반환 — resume 계약 불변).
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(judge_sid, todo)):
            if row.get("error"):
                n_err += 1
            append_jsonl(out_path, row)
            sw.update(done=len(done) + i + 1, metrics={"judge_errors": n_err})
    sw.finish(metrics={"n": len(ids), "judge_errors": n_err})
    write_marker("S4_SEMANTIC_DONE", {"n": len(ids), "errors": n_err})
    print(f"[S4] judged={len(todo)} errors={n_err}")


if __name__ == "__main__":
    main()
