#!/usr/bin/env python3
"""gate_go_check.py — Φ 재생성 직후 자동 Go/No-Go 관문 (cesft_v2_fp 설계 §4-1).

무인 야간 운전용: 사람 육안 검수 대신 자동 판정 2건 + 아침 검토용 표본 50개 저장.
  1. 규칙 게이트 통과율 ≥ GATE_MIN_PASS (기본 0.60 — 현행 3인칭 Φ 는 70.3%)
  2. pass 트레이스의 1인칭율 ≥ GATE_MIN_FP (기본 0.30 — 1인칭 Φ 라면 크게 상회해야 정상)
통과 → markers/GATE_GO 기록(체인 진행). 실패 → exit 1 (run_stage 가 CHAIN_FAILED 기록,
supervisor 재시도 후 CHAIN_STUCK 정지 = 학습 착수 금지).
참고 지표(판정 비관여): scene_desc_rate·avg_words — 아침에 구 Φ 와 대조.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time

RE_FP = re.compile(r"\b(I|I'm|I've|I'll|my|me|myself)\b")
RE_SCENE = re.compile(r"\b(shows?|appears?|the frame|visible|can be seen|depicts?|image)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cesft_v2_fp")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    min_pass = float(os.environ.get("GATE_MIN_PASS", "0.60"))
    min_fp = float(os.environ.get("GATE_MIN_FP", "0.30"))

    rows = [json.loads(l) for l in open(run / "data" / "chosen_train.jsonl", encoding="utf-8") if l.strip()]
    n = len(rows)
    passed = [r for r in rows if r.get("gate") == "pass"]
    pass_rate = len(passed) / max(1, n)

    fp = sum(1 for r in passed if RE_FP.search(r.get("reasoning") or ""))
    scene = sum(1 for r in passed if RE_SCENE.search(r.get("reasoning") or ""))
    words = sum(len((r.get("reasoning") or "").split()) for r in passed)
    fp_rate = fp / max(1, len(passed))

    report = {
        "n": n, "pass": len(passed), "pass_rate": round(pass_rate, 4),
        "first_person_rate": round(fp_rate, 4),
        "scene_desc_rate": round(scene / max(1, len(passed)), 4),
        "avg_words": round(words / max(1, len(passed)), 1),
        "thresholds": {"min_pass": min_pass, "min_fp": min_fp},
        "ts": time.time(),
    }
    ev = run / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "gate_go_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # 아침 검토용 표본 50개 (pass 앞쪽 50 — 결정적)
    lines = ["# GATE_GO 표본 50 — hedging('I think/I guess')·근거 희석 여부 육안 점검용", ""]
    for r in passed[:50]:
        lines += [f"## {r['sample_id']} (gt: {r.get('gt')})",
                  f"- belief: {r.get('task_belief')}",
                  f"- reasoning: {r.get('reasoning')}", ""]
    (ev / "gate_go_samples.md").write_text("\n".join(lines), encoding="utf-8")

    ok = pass_rate >= min_pass and fp_rate >= min_fp
    print(f"[gate_go] {json.dumps(report)}")
    if not ok:
        print(f"[gate_go] FAIL — pass_rate {pass_rate:.3f} (≥{min_pass}?) "
              f"fp_rate {fp_rate:.3f} (≥{min_fp}?) → 학습 착수 금지")
        raise SystemExit(1)
    mk = run / "markers"
    mk.mkdir(parents=True, exist_ok=True)
    (mk / "GATE_GO").write_text(json.dumps(report))
    print("[gate_go] PASS → GATE_GO 마커 기록")


if __name__ == "__main__":
    main()
