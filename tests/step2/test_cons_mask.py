#!/usr/bin/env python3
"""test_cons_mask.py — P3 consistency loss 의 candidate 마스킹 정합성 (GPU/모델 불필요).

검증 대상은 `pro_gr_train.build_candidate_batch` 다. 여기서 마스크가 어긋나면 loss 는
조용히 잘못된 토큰을 스코어링하고, 학습 로그만 봐서는 절대 드러나지 않는다.

이 테스트는 실제로 버그를 하나 잡았다: 초기 구현은 `base_text + completion` 문자열을 다시
토크나이즈하고 standalone base_len 으로 잘랐는데, base 가 공백으로 끝나면 그 공백과 '<' 가
한 토큰으로 병합돼 completion 첫 글자가 마스크에서 누락됐다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "step2"))

from pro_gr_train import build_candidate_batch, cand_completion  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# 길이가 일부러 크게 다른 후보들 — 행별 패딩량이 달라져야 정합 검증이 실효를 갖는다
CANDS = [("take", "container"), ("put", "lid"), ("open", "microwave_oven_door"),
         ("cut", "x"), ("wash", "plate")]


def main():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL, use_fast=True)
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    fails = 0
    # base 가 공백으로 끝나는 경우(병합 유발)와 아닌 경우 둘 다 본다
    for base_text in ("SYSTEM: predict the next action.\nUSER: what now?\nASSISTANT: ",
                      "SYSTEM: predict.\nASSISTANT:\n<reasoning>I am holding a lid.</reasoning>\n"):
        base_ids = torch.tensor(tok(base_text)["input_ids"], dtype=torch.long)
        ids, attn, mask, base_len, K = build_candidate_batch(tok, base_ids, CANDS, pad_id)
        assert base_len == base_ids.numel() and ids.shape[1] == base_len + K
        tgt = ids[:, 1:]
        print(f"  base={base_text[-24:]!r}  L={ids.shape[1]}")
        ks = set()
        for r, (v, n) in enumerate(CANDS):
            got = tok.decode(tgt[r][mask[r] > 0], skip_special_tokens=True)
            want = cand_completion(v, n)
            ok = got == want
            fails += (not ok)
            ks.add(int(mask[r].sum()))
            print(f"    [{'ok ' if ok else 'FAIL'}] row{r} k={int(mask[r].sum()):3d} {got!r}")
            if not ok:
                print(f"          expected {want!r}")
            # attention 은 base+completion 만 1, 나머지는 패딩
            assert int(attn[r].sum()) == base_ids.numel() + int(mask[r].sum()), "attn 길이 불일치"
            # 마스크는 base 영역을 절대 건드리지 않는다
            assert mask[r][: base_ids.numel() - 1].sum() == 0, "base 영역이 마스크에 포함됨"
        assert len(ks) > 1, "후보 토큰 길이가 모두 같다 — 패딩 정합을 검증하지 못한다"

    # 후보 순서를 셔플해도 각 행의 스코어 대상이 그 행의 후보와 계속 일치하는지
    shuffled = list(reversed(CANDS))
    base_ids = torch.tensor(tok("ASSISTANT: ")["input_ids"], dtype=torch.long)
    ids, _, mask, _, _ = build_candidate_batch(tok, base_ids, shuffled, pad_id)
    for r, (v, n) in enumerate(shuffled):
        got = tok.decode(ids[:, 1:][r][mask[r] > 0], skip_special_tokens=True)
        ok = got == cand_completion(v, n)
        fails += (not ok)
        print(f"    [{'ok ' if ok else 'FAIL'}] shuffled row{r} {got!r}")

    if fails:
        print(f"[FAIL] {fails} 행 불일치")
        return 1
    print("[PASS] 모든 행 일치 — 경계 병합·패딩·순서에 무관하게 마스크 정합")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
