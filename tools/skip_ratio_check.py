#!/usr/bin/env python3
"""skip_ratio_check.py — 학습 로그의 skip_decode 비율 점검 (프레임 캐시 결손 조기 감지)."""
import json
import pathlib
import sys

run = sys.argv[1] if len(sys.argv) > 1 else "cand_free"
p = pathlib.Path(f"outputs/step2_retrospection/cesft_v2/{run}/train_log.jsonl")
if not p.is_file():
    print(f"[check] {p} 없음")
    sys.exit(0)
rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
sk = sum(1 for r in rows if "skip_decode" in r)
tr = sum(1 for r in rows if "step" in r)
ratio = sk / max(1, sk + tr)
print(f"[check] {run}: skip_decode={sk} train_rows={tr} skip_ratio={ratio:.1%}"
      + ("  <-- 경고: 프레임 캐시 결손 의심" if ratio > 0.05 else "  OK"))
