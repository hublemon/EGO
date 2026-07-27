#!/usr/bin/env python3
"""freegen malformed 원인 진단 — 실패 표본의 **생성 원문**을 그대로 찍는다.

배경: sft_r15_c 의 cand-free malformed 가 17.8%(theta_ce 2.4%)다. 토큰 예산(320→512)을
올려도 실패 표본 집합이 80건 그대로 일치했고, 정상 생성분의 reasoning 최대 길이도 95단어라
잘림이 아니다. 따라서 parse_trace 의 세 태그 중 무엇이 왜 빠지는지 원문에서 확인해야 한다.

사용: PYTHONPATH=src RETRO3_RUNS=runs/cesft_v2_fp_fg512 python tools/diag_freegen_malformed.py \
        --arm sft_r15_c --n 24
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.eval.battery import load_arm, pick_eval_set
from ego.step2_retrospection.eval.freegen import freegen_messages
from ego.step2_retrospection.runtime import read_jsonl, runs_root

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/cesft_v2_fp.yaml")
    ap.add_argument("--arm", default="sft_r15_c")
    ap.add_argument("--adapter", default="outputs/step2_retrospection/cesft_v2_fp/sft_r15_c/adapter")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", default="runs/cesft_v2_fp_fg512/eval/diag_malformed.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    rows, _ = pick_eval_set(read_jsonl(runs_root() / "data" / "context_val.jsonl"),
                            "heldout", 500, covered_only=True)

    recp = runs_root() / "eval" / f"freegen_{args.arm}_cand_free.records.jsonl"
    bad = {r["sample_id"] for r in read_jsonl(recp) if r.get("malformed")}
    todo = [r for r in rows if r["sample_id"] in bad][: args.n]
    print(f"malformed {len(bad)}건 중 {len(todo)}건 재생성해 원문 확인\n")

    model, processor = load_arm(args.adapter)
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "w", encoding="utf-8")

    miss = {"reasoning": 0, "task_belief": 0, "action": 0}
    empty = 0
    for chunk, frames in vlm.prefetch_chunks(video_root, todo, 8):
        msgs, ok = [], []
        for rec, (imgs, err) in zip(chunk, frames):
            if err is None:
                msgs.append(freegen_messages(rec, imgs)); ok.append(rec)
        texts = vlm.generate_batch(model, processor, msgs, max_new_tokens=args.max_new_tokens)
        for rec, text in zip(ok, texts):
            present = {t: bool(rx.search(text)) for t, rx in vlm.TAG_RE.items()}
            for t, p in present.items():
                if not p:
                    miss[t] += 1
            if not text.strip():
                empty += 1
            fh.write(json.dumps({"sample_id": rec["sample_id"], "present": present,
                                 "n_chars": len(text), "text": text}, ensure_ascii=False) + "\n")
            print(f"── {rec['sample_id']}  chars={len(text)}  "
                  f"태그={''.join(k[0].upper() if v else '.' for k, v in present.items())}")
            print(text[:900].replace("\n", "\n   "))
            print()
    fh.close()
    vlm.close_readers()
    print(f"\n누락 집계 (n={len(todo)}): {miss}   빈 출력 {empty}건")
    print(f"원문 저장: {outp}")
    # 열린 태그는 있는데 닫는 태그가 없는 경우 = 잘림, 태그 자체가 없으면 = 형식 이탈
    txts = [json.loads(l)["text"] for l in open(outp, encoding="utf-8")]
    for t in ("reasoning", "task_belief", "action"):
        o = sum(1 for x in txts if f"<{t}>" in x)
        c = sum(1 for x in txts if f"</{t}>" in x)
        print(f"  <{t}> 열림 {o} / 닫힘 {c}")
    print(f"  <action> 다음 위치에서 문장 끝난 사례: "
          f"{sum(1 for x in txts if re.search(r'<action>[^<]*$', x))}건")


if __name__ == "__main__":
    main()
