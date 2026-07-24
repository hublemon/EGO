#!/usr/bin/env python3
"""frontier_select_eval.py — Frontier VLM(텍스트-조건) candidate-scored next-action 선택 평가.
main.tex Table 3(tab:reasoning)의 'Frontier VLM' 행과 Table 4(tab:trace)의 Frontier 열을 채운다.

프로토콜(정책 arm과 최대한 정렬, 단 게이트웨이 모델이 text-only 이므로 이미지 대신 텍스트 컨텍스트):
  입력  : scenario + memory_context(행동 이력) + WM Top-K 후보 리스트(순서 shuffle 안 함 — WM rank 그대로)
  출력  : 후보 중 정확히 하나를 골라 {"choice": "<verb> <noun>", "reasoning": "..."} JSON
  채점  : chosen == (gt_verb, gt_noun) → correct. gt_idx(후보 내 GT 위치), wm1_idx=0(WM top-1) 기록
          → G1(gt_idx==0) / G2(0<gt_idx<5) / OUT(gt_idx>=5 or None) 분해, GADR=G2 correct rate.

한계(반드시 보고): frontier 는 이미지 미열람(text-only 게이트웨이). base/C 정책은 프레임을 봄 →
직접 비교 아님. main.tex §er-next 의 'text-conditioned, 별도 평가' 각주와 일치.

재개: --out 의 records.jsonl 에 이미 있는 sample_id 는 건너뜀(중단/재시작 안전).
비밀키: FRONTIER_API_KEY 환경변수에서만 읽음(절대 하드코딩·로깅 금지).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests


def call_api(base_url, api_key, model, system, user, max_retries=5, timeout=90):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    last = None
    for a in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], None
        except Exception as e:  # noqa: BLE001
            last = str(e)[:200]
            time.sleep(min(2 ** a, 12))
    return None, last


def build_prompt(ex, cands, K):
    scenario = ex.get("scenario", "") or "cooking"
    hist = ex.get("memory_context", "") or "No previous action history is available."
    goal = ex.get("task_goal", "") or "(not specified)"
    cand_txt = "\n".join(f"  {i+1}. {v} {n}" for i, (v, n) in enumerate(cands))
    system = (
        "You are an egocentric action-anticipation assistant for cooking videos. "
        "You are given the scenario, the goal, the recent action history, and a list of "
        f"{K} candidate next actions ranked by a video world model. Choose the SINGLE most "
        "likely next action the camera-wearer will perform. You MUST pick exactly one option "
        "COPIED VERBATIM from the candidate list. Respond ONLY with a JSON object: "
        '{"reasoning": "<one or two sentences>", "choice": "<verb> <noun>"}.'
    )
    user = (
        f"SCENARIO: {scenario}\n"
        f"GOAL: {goal}\n\n"
        f"RECENT ACTION HISTORY:\n{hist}\n\n"
        f"CANDIDATE NEXT ACTIONS (world-model ranked, choose exactly one):\n{cand_txt}\n\n"
        "Return ONLY the JSON object."
    )
    return system, user


def parse_choice(content, cands):
    """returns (verb, noun, reasoning, chosen_idx or -1)."""
    if not content:
        return "", "", "", -1
    txt = content.strip()
    reasoning, choice = "", ""
    # JSON 우선
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            reasoning = str(o.get("reasoning", ""))
            choice = str(o.get("choice", "")).strip()
        except json.JSONDecodeError:
            pass
    if not choice:
        choice = txt
    cl = choice.lower()
    # 후보와 매칭 (정확 → 포함)
    for i, (v, n) in enumerate(cands):
        if f"{v} {n}".lower() == cl:
            return v, n, reasoning, i
    for i, (v, n) in enumerate(cands):
        if f"{v} {n}".lower() in cl:
            return v, n, reasoning, i
    return "", "", reasoning, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="v2_heldout_eval.jsonl (topk_actions/gt 포함)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0=전량")
    ap.add_argument("--base-url", default=os.environ.get("FRONTIER_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--model", default=os.environ.get("FRONTIER_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    key = os.environ.get("FRONTIER_API_KEY")
    if not key:
        sys.exit("ERROR: FRONTIER_API_KEY 환경변수를 설정하세요.")

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    rec_p = out_p.with_suffix(".records.jsonl")
    done_ids = set()
    if rec_p.exists():
        for l in open(rec_p, encoding="utf-8"):
            if l.strip():
                try:
                    done_ids.add(json.loads(l)["sample_id"])
                except Exception:  # noqa: BLE001
                    pass
    print(f"[frontier-select] model={args.model} total={len(rows)} resume_done={len(done_ids)}", flush=True)

    rec_f = rec_p.open("a", encoding="utf-8")
    t0 = time.time()
    n = done = fail = 0
    for i, ex in enumerate(rows):
        sid = ex.get("frame_id") or ex.get("episode_id") or str(i)
        if sid in done_ids:
            continue
        cands = [(str(d["verb"]), str(d["noun"])) for d in ex["topk_actions"][:args.top_k]]
        gt = (str(ex["gt_verb"]), str(ex["gt_noun"]))
        gt_idx = cands.index(gt) if gt in cands else -1
        system, user = build_prompt(ex, cands, args.top_k)
        content, err = call_api(args.base_url, key, args.model, system, user)
        v, nn, reasoning, ci = parse_choice(content, cands) if content else ("", "", "", -1)
        malformed = (ci < 0)
        correct = (not malformed) and (v, nn) == gt
        rec = {
            "sample_id": sid, "chosen_verb": v, "chosen_noun": nn, "chosen_idx": ci,
            "gt_verb": gt[0], "gt_noun": gt[1], "gt_idx": gt_idx, "wm1_idx": 0,
            "correct": bool(correct), "malformed": bool(malformed),
            "history_length": ex.get("history_length"),
            "reasoning": reasoning, "raw": (content or "")[:1000],
            "api_error": err,
        }
        rec_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        done += int(correct)
        fail += int(content is None)
        if n % 20 == 0:
            rec_f.flush()
            el = time.time() - t0
            print(f"[frontier-select] {i+1}/{len(rows)} acc={done/n:.3f} apifail={fail} "
                  f"({el/n:.1f}s/it, ETA {(len(rows)-i-1)*el/n/60:.0f}m)", flush=True)
    rec_f.close()

    # 집계 (records 전체 재로드 — resume 포함)
    recs = [json.loads(l) for l in open(rec_p, encoding="utf-8") if l.strip()]
    covered = [r for r in recs if r["gt_idx"] is not None and 0 <= r["gt_idx"] < 5]
    g1 = [r for r in covered if r["gt_idx"] == 0]
    g2 = [r for r in covered if r["gt_idx"] > 0]
    def rate(xs): return round(sum(x["correct"] for x in xs) / len(xs), 4) if xs else None
    metrics = {
        "model": args.model, "n": len(recs),
        "SelAcc_covered_top5": rate(covered), "n_covered": len(covered),
        "G1_retention": rate(g1), "n_G1": len(g1),
        "GADR_G2": rate(g2), "n_G2": len(g2),
        "Acc_full": rate(recs),
        "malformed_rate": round(sum(r["malformed"] for r in recs) / len(recs), 4),
        "coverage_at_k": round(sum(1 for r in recs if r["gt_idx"] is not None and r["gt_idx"] >= 0) / len(recs), 4),
        "note": "text-conditioned frontier (이미지 미열람). base/C 정책과 직접 비교 아님 — §er-next 각주.",
    }
    out_p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] {json.dumps(metrics, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
