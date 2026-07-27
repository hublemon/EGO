#!/usr/bin/env python3
"""gdelta_summary.py — G-DELTA 체인 산출물 요약 출력."""
import json
import pathlib

EV = pathlib.Path("runs/cesft_v2/eval")
FILES = ["cand_free.json", "random_cand.json",
         "paired_G-DELTA_theta_ce_vs_cand_free.json",
         "paired_G-DELTA_theta_ce_vs_random_cand.json",
         "DiD_history_theta_ce_vs_cand_free.json",
         "strip_verdict_cand_free.json"]
KEYS = ("n", "acc", "L0_wm_top1", "beats_L0", "n_paired", "n_clusters", "point_a", "point_b",
        "delta", "DiD_pp", "delta_a_pp", "delta_b_pp", "ci95", "pass", "delta_acc_all", "error")

for f in FILES:
    p = EV / f
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f, json.dumps({k: d[k] for k in KEYS if k in d}, ensure_ascii=False))
    else:
        print(f, "(없음)")
