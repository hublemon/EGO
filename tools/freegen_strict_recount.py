#!/usr/bin/env python3
"""freegen_strict_recount.py — 자유생성(freegen) GT 일치율의 엄격도별 재집계. CPU 전용.

배경: `eval/freegen_{arm}_cand_free.json` 의 `gt_correct` 는 `vlm.match_candidate(action, [gt])`
로 산출된다. 후보 리스트가 **단일 원소**일 때 match_candidate 의 마지막 폴백(토큰 집합 겹침 최대가
유일하면 채택, vlm.py:244-251)은 항상 성립하므로 — verb 또는 noun 하나만 겹쳐도 정답 처리된다.
즉 로그된 gt_correct 는 strict 일치가 아니라 **any-token-overlap** 이다(본 스크립트가 동치 검증).

제시 레짐(battery)의 acc 는 (verb,noun) strict 이므로, 두 레짐을 한 표에 넣으려면 자유생성도
strict 로 다시 세야 한다. 이 스크립트가 그 재집계를 하고 엄격도 사다리를 함께 남긴다:

  gt_logged        : 기록된 gt_correct (= any-token-overlap. 논문 인용 금지)
  gt_strict        : 정규화(소문자·공백 축약) 완전일치            ← 제시 레짐 acc 와 같은 잣대
  gt_strict_paren  : 위 + taxonomy 접미사 `_(...)` 제거 후 완전일치 ← 표기 누락만 흡수한 관대-strict
  gt_verb_only     : verb 만 일치 (noun 불일치)   } 관대함의 출처 진단
  gt_noun_only     : noun 만 일치 (verb 불일치)   }
  in_support_*     : 경계 내재화 지표도 같은 3단계로 (K=10 리스트라 폴백이 '유일 최대'를 요구해
                     상대적으로 안전하지만, 각주용으로 엄격판을 병기)

사용:
  python3 tools/freegen_strict_recount.py --run runs/cesft_v2_fp
  python3 tools/freegen_strict_recount.py --run runs/cesft_v2_fp --arms base,cand_free --out <path>

출력: <run>/eval/freegen_strict_recount.json (기존 freegen_*.json 을 덮지 않음) + stdout 표.
GPU·모델 로드 없음. 표준 라이브러리만. 체인 마커/산출물 불변 → 실행 중 체인과 병행 안전.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

# vlm.py 의 정규화와 동일 (소문자·공백 축약) — 잣대 일치 유지.
_norm = lambda x: re.sub(r"\s+", " ", x.lower().strip())  # noqa: E731
# taxonomy 접미사 제거: "get_(fetch,_take) ingredient" -> "get ingredient"
_strip_paren = lambda x: re.sub(r"\s+", " ", re.sub(r"_\([^)]*\)", "", _norm(x))).strip()  # noqa: E731


def _tokens(x: str) -> set[str]:
    return set(_strip_paren(x).split())


def match_strict(action: str | None, target: str) -> bool:
    return action is not None and _norm(action) == _norm(target)


def match_strict_paren(action: str | None, target: str) -> bool:
    return action is not None and _strip_paren(action) == _strip_paren(target)


def match_token_overlap(action: str | None, target: str) -> bool:
    """match_candidate(action, [target]) 의 실효 동작 — 토큰 1개라도 겹치면 True."""
    return action is not None and len(_tokens(action) & _tokens(target)) > 0


def in_support_strict(action: str | None, cands: list[str], paren: bool = False) -> bool:
    if action is None:
        return False
    f = _strip_paren if paren else _norm
    return any(f(action) == f(c) for c in cands)


def recount(records: list[dict], ctx: dict[str, dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    acc: dict[str, int] = dict.fromkeys(
        ("malformed", "gt_logged", "gt_strict", "gt_strict_paren", "gt_token_overlap",
         "gt_verb_only", "gt_noun_only", "in_support_logged", "in_support_strict",
         "in_support_strict_paren", "coverage_topk"), 0)
    for r in records:
        a, gt = r.get("action"), r.get("gt", "")
        acc["malformed"] += bool(r.get("malformed"))
        acc["gt_logged"] += bool(r.get("gt_correct"))
        acc["in_support_logged"] += bool(r.get("in_support"))
        acc["coverage_topk"] += bool(r.get("gt_in_support"))
        acc["gt_strict"] += match_strict(a, gt)
        acc["gt_strict_paren"] += match_strict_paren(a, gt)
        acc["gt_token_overlap"] += match_token_overlap(a, gt)
        if a and not match_strict_paren(a, gt):
            av, gv = _strip_paren(a).split(), _strip_paren(gt).split()
            acc["gt_verb_only"] += bool(av and gv and av[0] == gv[0])
            acc["gt_noun_only"] += bool(len(av) > 1 and len(gv) > 1 and av[-1] == gv[-1]
                                        and av[0] != gv[0])
        cands = (ctx.get(r.get("sample_id", ""), {}) or {}).get("candidates", [])
        if cands:
            acc["in_support_strict"] += in_support_strict(a, cands)
            acc["in_support_strict_paren"] += in_support_strict(a, cands, paren=True)
    out = {"n": n, **{k: round(v / n, 4) for k, v in acc.items()}, "counts": acc}
    out["logged_equals_token_overlap"] = acc["gt_logged"] == acc["gt_token_overlap"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cesft_v2_fp", help="run dir (eval/ 하위를 읽는다)")
    ap.add_argument("--arms", default=None, help="쉼표 구분. 생략 시 freegen records 전부 자동 탐색")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    ev = run / "eval"
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        arms = sorted(Path(p).name[len("freegen_"):-len("_cand_free.records.jsonl")]
                      for p in glob.glob(str(ev / "freegen_*_cand_free.records.jsonl")))
    ctx = {}
    ctx_path = run / "data" / "context_val.jsonl"
    if ctx_path.is_file():
        with ctx_path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                ctx[r["sample_id"]] = {"candidates": r.get("candidates", [])}

    res = {"run": str(run), "arms": {},
           "note": ("freegen gt_correct(로그값)은 match_candidate 단일원소 폴백 탓에 "
                    "any-token-overlap 과 동치 — 제시 레짐 acc(strict)와 비교 불가. "
                    "표에는 gt_strict(또는 gt_strict_paren)를 쓴다.")}
    for arm in arms:
        p = ev / f"freegen_{arm}_cand_free.records.jsonl"
        if not p.is_file():
            print(f"[skip] {p} 없음")
            continue
        with p.open(encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        res["arms"][arm] = recount(recs, ctx)

    dst = Path(args.out) if args.out else ev / "freegen_strict_recount.json"
    dst.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = [("n", "n"), ("gt_logged", "gt_logged"), ("gt_strict", "gt_strict"),
            ("gt_strict_paren", "gt_paren"), ("gt_verb_only", "verb_only"),
            ("gt_noun_only", "noun_only"), ("in_support_logged", "insup_log"),
            ("in_support_strict_paren", "insup_paren"), ("malformed", "malformed")]
    print(f"\n{'arm':<12}" + "".join(f"{lab:>12}" for _, lab in cols))
    for arm, d in res["arms"].items():
        row = f"{arm:<12}"
        for key, _ in cols:
            v = d.get(key)
            row += f"{v:>12}" if key == "n" else f"{100 * v:>11.1f}%"
        print(row)
    for arm, d in res["arms"].items():
        if not d.get("logged_equals_token_overlap"):
            print(f"[warn] {arm}: 로그값 ≠ token-overlap — match_candidate 동작 재확인 필요")
    print(f"\n[written] {dst}")


if __name__ == "__main__":
    main()
