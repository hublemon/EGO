#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_run1_curves.py — Run 1 G1/G2 곡선 figure 생성 (Run 3 메인 figure 의 초안).

의존성: json + matplotlib 만 (시스템 python3 로 실행 — eve-cu124 에는 matplotlib 없음).
입력:  runs/grpo_run1_wmonly/heldout_eval/step*.json   (eval_checkpoints_run1.sh 산출)
       runs/grpo_run1_wmonly/reward_log.jsonl          (train reward 곡선)
       runs/grpo_final/heldout_eval/gtoracle_step1250.json (있으면 GT-oracle 참조선)
출력:  runs/grpo_run1_wmonly/figures/run1_curves.png
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = os.path.expanduser("~/work/jihun/EGO/runs/grpo_run1_wmonly")
GTO = os.path.expanduser("~/work/jihun/EGO/runs/grpo_final/heldout_eval/gtoracle_step1250.json")


def main():
    files = sorted(glob.glob(f"{RUN}/heldout_eval/step*.json"),
                   key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    if not files:
        print("no heldout_eval results yet — run eval_checkpoints_run1.sh first")
        return
    steps, acc, g2, escape, follow = [], [], [], [], []
    wm_ref = None
    for p in files:
        s = json.load(open(p))
        steps.append(int(re.search(r"step(\d+)", p).group(1)))
        acc.append(s.get("gt_action_acc_fuzzy"))
        g2.append(s.get("g2_vlm_acc"))
        escape.append(s.get("candidate_escape_rate"))
        follow.append(s.get("wm_follow_rate"))
        wm_ref = s.get("wm_top1_gt_action_acc") or wm_ref

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(steps, acc, "o-", color="#0f766e", label="VLM action acc (fuzzy)")
    if wm_ref:
        ax.axhline(wm_ref, ls="--", color="#b45309", label=f"WM top-1 참조선 ({wm_ref:.3f})")
    if os.path.exists(GTO):
        gto = json.load(open(GTO)).get("gt_action_acc_fuzzy")
        if gto:
            ax.axhline(gto, ls=":", color="#7c3aed", label=f"GT-oracle 상한 Exp.14 ({gto:.3f})")
    ax.set_title("G1: held-out action accuracy")
    ax.set_xlabel("step"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(steps, g2, "o-", color="#0f766e", label="G2 구간 VLM acc")
    ax.axhline(0.20, ls="--", color="#b45309", label="chance (0.20)")
    ax.set_title("G2: WM top-1 오답 & GT∈top5 구간")
    ax.set_xlabel("step"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(steps, escape, "o-", color="#b91c1c", label="후보 이탈률")
    ax.plot(steps, follow, "s-", color="#5c6b67", label="WM-follow rate")
    ax.set_title("진단: 이탈률 / rank-1 복사 감시")
    ax.set_xlabel("step"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    os.makedirs(f"{RUN}/figures", exist_ok=True)
    out = f"{RUN}/figures/run1_curves.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
