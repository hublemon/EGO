#!/usr/bin/env python3
"""trace_text_metrics.py — 트레이스 표면 텍스트 지표 (EGO_jihun 파일럿 이식본, GPU 불필요).

정규식은 `EGO_jihun/scripts/step2/trace_text_metrics.py` 에서 **글자 그대로** 가져왔다
(파일럿 실측 52.4→61.4% 등과 같은 잣대 유지 — 새 정의 금지, cesft_v2_fp 설계 §5 #5).
입력은 EGO_jihun3 records 스키마: `reasoning`/`task_belief` 가 이미 필드로 저장돼 있어
completion 파싱(reasoning_of) 단계가 필요 없다.

레짐 2종을 자동 수집:
  presented : eval/{arm}.records.jsonl          (battery — 후보 제시)
  cand_free : eval/freegen_{arm}_cand_free.records.jsonl (freegen — 후보 비제시)

배제(elimination)는 두 정의를 병기:
  elim_mention_rate : 선택 외 후보 ≥2개를 reasoning 이 문자열로 거명 (파일럿 v3_cf_freegen_eval.py:96-103 정의)
                      — candidates 는 records 에 없으므로 data/context_val.jsonl 로 sample_id 조인.
  elim_lang_rate    : (other|remaining|alternative) (candidates?|options?|actions?) 패턴 (대시보드 §2 정의)

사용: PY tools/trace_text_metrics.py [--run runs/cesft_v2_fp] [--arms base,theta_ce,sft_r15,cand_free]
출력: <run>/eval/text_metrics.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

# ── EGO_jihun 원본 정규식 (변경 금지 — 파일럿과 동일 잣대) ──────────────────────
RE_FP = re.compile(r"\b(I|I'm|I've|I'll|my|me|myself)\b")
RE_SCENE = re.compile(r"\b(shows?|appears?|the frame|visible|can be seen|depicts?|image)\b", re.I)
RE_FUTURE = re.compile(r"\b(will|next|should|going to|about to|likely|plan to|intend)\b", re.I)
RE_CAUSAL = re.compile(r"\b(since|because|having just|given that|as a result|therefore|thus|so that)\b", re.I)
# 대시보드 §2 '배제 언명률' 패턴 (footnote b)
RE_ELIM_LANG = re.compile(r"\b(other|remaining|alternative)\s+(candidates?|options?|actions?)\b", re.I)


def load(p: pathlib.Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def analyze(records: list[dict], cand_map: dict[str, list[str]]) -> dict:
    """non-malformed & reasoning 존재 records 만 채점 (대시보드 §2 규약)."""
    recs = [r for r in records if not r.get("malformed") and r.get("reasoning")]
    n = len(recs)
    if n == 0:
        return {"n": 0}
    fp = fp_b = scene = fut = caus = elim_m = elim_l = 0
    words = 0
    mentioned_total = 0
    for r in recs:
        rz = r["reasoning"]
        bl = r.get("task_belief") or ""
        if RE_FP.search(rz):
            fp += 1
        if RE_FP.search(bl):
            fp_b += 1
        if RE_SCENE.search(rz):
            scene += 1
        if RE_FUTURE.search(rz):
            fut += 1
        if RE_CAUSAL.search(rz):
            caus += 1
        if RE_ELIM_LANG.search(rz):
            elim_l += 1
        cands = cand_map.get(r["sample_id"])
        if cands:
            chosen = r.get("action")
            mentioned = sum(1 for c in cands
                            if c != chosen and (c in rz or c.split(" ", 1)[-1] in rz))
            mentioned_total += mentioned
            if mentioned >= 2:
                elim_m += 1
        words += len(rz.split())
    return {
        "n": n,
        "first_person_rate": round(fp / n, 4),
        "first_person_rate_belief": round(fp_b / n, 4),
        "scene_desc_rate": round(scene / n, 4),
        "future_rate": round(fut / n, 4),
        "causal_rate": round(caus / n, 4),
        "elim_mention_rate": round(elim_m / n, 4),
        "elim_lang_rate": round(elim_l / n, 4),
        "avg_mentioned": round(mentioned_total / n, 2),
        "avg_words": round(words / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cesft_v2_fp")
    ap.add_argument("--arms", default="base,theta_ce,sft_r15,cand_free")
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    ev = run / "eval"
    cand_map = {r["sample_id"]: r["candidates"] for r in load(run / "data" / "context_val.jsonl")}

    out = {"note": "regex는 EGO_jihun 파일럿과 동일(잣대 유지). presented=battery records, "
                   "cand_free=freegen records. non-malformed & reasoning 존재분만 채점.",
           "per_arm": {}}
    for arm in args.arms.split(","):
        arm = arm.strip()
        entry = {}
        p1 = ev / f"{arm}.records.jsonl"
        if p1.is_file():
            entry["presented"] = analyze(load(p1), cand_map)
        p2 = ev / f"freegen_{arm}_cand_free.records.jsonl"
        if p2.is_file():
            entry["cand_free"] = analyze(load(p2), cand_map)
        if entry:
            out["per_arm"][arm] = entry

    dst = ev / "text_metrics.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[done] -> {dst}")


if __name__ == "__main__":
    main()
