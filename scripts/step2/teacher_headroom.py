#!/usr/bin/env python3
"""teacher_headroom.py — Retrospection 가설의 정보량 상한을 학습 없이 측정한다.

핵심 질문:
    미래(완료된 궤적)를 본 teacher 가 후보 5개 중 GT 를 **실제로 더 잘 고르는가?**

이 차이가 Retrospection 이 student 에게 전달할 수 있는 정보량의 상한이다. 차이가 없으면
아무리 정교하게 증류해도 전달할 것이 없고, 논문 가설 자체가 위험하다. 학습 30시간을 쓰기
전에 1~2시간으로 판정한다.

3개 arm (동일 샘플 · 동일 스코어링 — paired 비교):
    present   : x≤t + 후보5                         (현재 정보만)
    future    : x≤t + **실제 미래 suffix** + 후보5    (hindsight)
    shuffled  : x≤t + **다른 샘플의 미래** + 후보5     (대조군)

`shuffled` 가 없으면 future 의 이득이 "미래 정보" 때문인지 "맥락이 길어져서"인지 구분할 수
없다. B>A 이면서 B>C 일 때만 가설이 지지된다 (projection ablation 을 teacher 수준에서 선행).

★ 누설 차단: 미래 suffix 에서 GT 와 canonical 이 같은 action 을 **전부** 제거한다.
   실측 확인 — b0_samples 첫 샘플의 future offset 2 가 GT(`wash spatula`)와 동일하다.
   이걸 안 지우면 teacher 가 자명하게 100% 가 되어 측정이 무의미해진다.
   (build_dpo_dataset_r1.py:55-56 의 규약과 동일)

스코어링은 생성 없이 후보 5개 teacher forcing → argmax. 세 arm 이 완전히 같은 방식이므로
절대값의 왜곡은 공통이고 arm 간 차이만 해석한다.

    python scripts/step2/teacher_headroom.py --samples b0_samples.jsonl --out head.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "step2"))
sys.path.insert(0, str(REPO / "src"))
from pro_gr_train import score_candidates  # noqa: E402  (단위테스트된 스코어러 재사용)


def canon(v, n) -> str:
    return f"{str(v).strip().lower()}|{str(n).strip().lower()}"


def hindsight_block(suffix: list[dict], max_n: int = 8) -> str:
    """미래 suffix → 프롬프트 삽입 블록. 시각·offset 은 넣지 않는다(형식 단서 최소화)."""
    seq = " -> ".join(f"{a.get('verb','')} {a.get('noun','')}".strip() for a in suffix[:max_n])
    return ("\n\nWhat you actually did later (in order, after the action you are about to choose):\n"
            f"  {seq}\n"
            "Use this only to infer what you were trying to accomplish at this moment.\n")


def build_user(base_user: str, block: str | None) -> str:
    """hindsight 블록을 action history 뒤 · 후보 목록 앞에 넣는다.

    후보 뒤에 붙이면 마지막 지시문과 후보 사이가 벌어져 형식이 흔들린다. 앵커는
    'Action candidates' 헤더다 — 세 arm 모두 후보 블록의 위치가 동일해야 공정하다.
    """
    if not block:
        return base_user
    key = "\nAction candidates"
    i = base_user.find(key)
    if i < 0:
        return base_user + block
    return base_user[:i] + block + base_user[i:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="future_gt_actions 포함 jsonl (b0_samples)")
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None, help="없으면 base 모델 (teacher 는 동결)")
    ap.add_argument("--limit", type=int, default=0, help="0=전량")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max_pixels", type=int, default=602112)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)

    # oracle subset: GT ∈ 후보 · 누설 제거 후 미래 suffix 가 남는 샘플만
    usable = []
    drop_out, drop_nofuture = 0, 0
    for s in rows:
        gt = s.get("gt_action") or {}
        cands = [(str(c.get("verb")), str(c.get("noun"))) for c in (s.get("candidates") or [])]
        gk = canon(gt.get("verb"), gt.get("noun"))
        if gk not in [canon(v, n) for v, n in cands]:
            drop_out += 1
            continue
        suffix = [a for a in (s.get("future_gt_actions") or [])
                  if canon(a.get("verb"), a.get("noun")) != gk]      # ★ 누설 차단
        if not suffix:
            drop_nofuture += 1
            continue
        usable.append({"s": s, "cands": cands, "gk": gk, "suffix": suffix})
    if args.limit:
        usable = usable[: args.limit]
    n = len(usable)
    print(f"[data] 전체 {len(rows)} → 사용 {n}  (GT∉후보 {drop_out} · 미래없음 {drop_nofuture})",
          flush=True)
    if n < 2:
        raise SystemExit("사용 가능한 샘플이 부족하다.")

    # shuffled arm: derangement — 자기 자신의 미래를 받지 않도록 고정 offset 회전
    off = max(1, n // 3)
    for i, u in enumerate(usable):
        u["suffix_shuf"] = usable[(i + off) % n]["suffix"]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model_name, padding_side="left", use_fast=True,
        min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": args.device})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w", encoding="utf-8")
    f.write(json.dumps({"meta": {"n_usable": n, "samples": args.samples,
                                 "adapter": args.adapter, "seed": args.seed,
                                 "drop_gt_outside": drop_out,
                                 "drop_no_future": drop_nofuture}}) + "\n")
    f.flush()

    t0 = time.time()
    for i, u in enumerate(usable):
        s = u["s"]
        sys_msg = s["prompt"][0]["content"]
        base_user = s["prompt"][1]["content"]
        img = Image.open(s["image_path"]).convert("RGB")
        rec = {"sample_id": s["sample_id"], "gt": u["gk"]}
        try:
            for arm, block in (("present", None),
                               ("future", hindsight_block(u["suffix"])),
                               ("shuffled", hindsight_block(u["suffix_shuf"]))):
                msgs = [{"role": "system", "content": [{"type": "text", "text": sys_msg}]},
                        {"role": "user", "content": [{"type": "image"},
                                                     {"type": "text",
                                                      "text": build_user(base_user, block)}]}]
                text = processor.apply_chat_template(msgs, tokenize=False,
                                                     add_generation_prompt=True)
                enc = processor(text=[text], images=[[img]],
                                return_tensors="pt").to(model.device)
                with torch.no_grad():
                    sc = score_candidates(model, processor, enc, u["cands"])
                k = int(sc.argmax())
                pick = canon(*u["cands"][k])
                rec[arm] = {"pick": pick, "correct": pick == u["gk"],
                            "margin": round(float(sc.max() - sc.sort().values[-2]), 3)}
                del enc
        except Exception as e:                      # 한 샘플 실패가 실행을 죽이지 않게
            rec["error"] = repr(e)[:200]
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"[{i+1}/{n}] {rate*3600:.0f} 샘플/h · 남은 {int((n-i-1)/rate/60)}분",
                  flush=True)
    f.close()
    print(f"[done] → {out_path}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
