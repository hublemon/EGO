"""validate_cli.py — 저장된 B0 DPO 데이터셋 전체 재검증 CLI (freeze/train 게이트).

build_dpo_dataset 이 emit 직전에 검사하지만, 저장된 파일을 독립적으로 재검사할 수 있어야 한다.
저장 파일에는 _leak_check(원본 GT/future)가 제거돼 있으므로, 여기서는 prompt-substring 누설과
pair invariant(완결성·SAME/SAME·future 언어)만 검사한다. exit 1 = 위반 발견.
"""
from __future__ import annotations

import argparse
import sys

from .validate_dpo_dataset import validate_dataset_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", required=True, help="b0_dpo_{split}.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    n, errs = validate_dataset_file(args.dpo, args.limit)
    print(f"[validate] checked {n} records in {args.dpo}")
    if errs:
        print(f"[FAIL] {len(errs)} issue(s):")
        for e in errs[:50]:
            print("  " + e)
        if len(errs) > 50:
            print(f"  ... and {len(errs) - 50} more")
        sys.exit(1)
    print("[PASS] no leakage / no splicing / no SAME-SAME in training set")


if __name__ == "__main__":
    main()
