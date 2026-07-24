"""S3 굳히기 — 개입③(reasoning 인과성)을 확증/기각하는 강화 eval 팩.

기존 intervention.py를 확장:
  1. n=1000 + bootstrap CI (효과가 노이즈인지)
  2. 필드 분해 — belief-only / reasoning-only / both swap: 어느 필드가 인과를 나르나
  3. acc 직교성 — flip이 정답/오답 샘플에서 다른가 (causal이 acc와 독립 축인지)
  4. 유용성 순서 — p_gt: own > empty > swap 이 n=1000·CI에서 유지되나

판정 게이트 (멈춤 기준):
  G-S3a: causal_sensitivity(both-swap − paraphrase) CI 하한 > 0  → 인과 실재
  G-S3b: p_gt own > swap_both, CI로 분리                        → 유용성(맞는 방향) 실재
  둘 다 통과 → 논문 spine 확정. 하나라도 실패 → 라인 종료(더 안 태움).

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.eval.harden_s3 --arm r1_sft \
      --adapter outputs/step2_retrospection/r1_sft/adapter --n 1000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.eval.battery import load_arm
from ego.step2_retrospection.eval.intervention import candidate_probs
from ego.step2_retrospection.runtime import StatusWriter, read_jsonl, runs_root, write_marker


def bootstrap_ci(xs: list[float], stat="mean", n_boot=2000, seed=0):
    """per-sample 지표 배열 → (point, lo, hi) 95% bootstrap CI."""
    import statistics as st
    rng = random.Random(seed)
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = sum(xs) / n
    boots = []
    for _ in range(n_boot):
        samp = [xs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(samp) / n)
    boots.sort()
    return (round(point, 4), round(boots[int(.025 * n_boot)], 4), round(boots[int(.975 * n_boot)], 4))


def diff_ci(a: list[int], b: list[int], n_boot=2000, seed=0):
    """paired 차이 mean(a)-mean(b)의 95% CI (같은 인덱스 resample)."""
    rng = random.Random(seed)
    n = len(a)
    point = (sum(a) - sum(b)) / n
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append((sum(a[i] for i in idx) - sum(b[i] for i in idx)) / n)
    boots.sort()
    return (round(point, 4), round(boots[int(.025 * n_boot)], 4), round(boots[int(.975 * n_boot)], 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_start_m1_lobs8.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    yaml.safe_load(open(args.config, encoding="utf-8"))
    ctx = {r["sample_id"]: r for r in read_jsonl(runs_root() / "data" / "context_val.jsonl")}
    recs = [r for r in read_jsonl(runs_root() / "eval" / f"{args.arm}.records.jsonl")
            if not r.get("malformed") and r.get("task_belief") and r.get("reasoning")
            and r.get("gt_in_support")]
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    recs = recs[: args.n]

    model, processor = load_arm(args.adapter)
    device = "cuda"
    sw = StatusWriter(f"S3H_{args.arm}", total=len(recs))

    # paraphrase 통제군 (belief 재작성 — 의미 동일, 문체만)
    para_msgs = [[{"role": "user", "content": [{"type": "text", "text":
        f'Paraphrase this sentence, keeping the exact same meaning: "{r["task_belief"]}"\n'
        "Reply with the paraphrase only."}]}] for r in recs]
    paras = []
    for i0 in range(0, len(para_msgs), 16):
        paras.extend(vlm.generate_batch(model, processor, para_msgs[i0:i0 + 16], max_new_tokens=60))

    # per-sample 지표 (bootstrap용 배열)
    P = {k: [] for k in ("own", "empty", "swap_b", "swap_r", "swap_both")}   # p_gt
    F = {k: [] for k in ("swap_b", "swap_r", "swap_both", "para")}           # flip vs own top1
    correct = []   # own top1 == gt
    out_rows = []  # 정성 UI 덤프 (own vs belief-swap)
    n = len(recs)
    for i, r in enumerate(recs):
        rec = ctx[r["sample_id"]]
        gt_idx = rec["candidates"].index(r["gt"])
        partner = recs[(i + 7) % n]
        b_other, r_other = partner["task_belief"], partner["reasoning"]
        variants = {
            "own":       (r["reasoning"], r["task_belief"]),
            "empty":     (r["reasoning"], None),
            "swap_b":    (r["reasoning"], b_other),
            "swap_r":    (r_other, r["task_belief"]),
            "swap_both": (r_other, b_other),
            "para":      (r["reasoning"], paras[i].strip().strip('"')),
        }
        top1 = {}
        for name, (rr, bb) in variants.items():
            p = candidate_probs(model, processor, rec, rr, bb, device)
            top1[name] = int(p.argmax())
            if name in P:
                P[name].append(float(p[gt_idx]))
        own_t = top1["own"]
        correct.append(int(own_t == gt_idx))
        for k in ("swap_b", "swap_r", "swap_both", "para"):
            F[k].append(int(top1[k] != own_t))
        cands = rec["candidates"]
        out_rows.append({
            "sample_id": r["sample_id"], "gt": r["gt"],
            "own_belief": r["task_belief"], "swap_belief": b_other,
            "own_action": cands[own_t], "swap_b_action": cands[top1["swap_b"]],
            "p_gt_own": round(P["own"][-1], 4), "p_gt_swap_b": round(P["swap_b"][-1], 4),
            "flip_swap_b": int(top1["swap_b"] != own_t), "own_correct": int(own_t == gt_idx)})
        sw.update(done=i + 1, metrics={
            "flip_both": round(sum(F["swap_both"]) / (i + 1), 3),
            "flip_para": round(sum(F["para"]) / (i + 1), 3)})

    # 집계 + CI
    ci = {k: bootstrap_ci(v) for k, v in {**{f"p_{a}": P[a] for a in P},
                                          **{f"flip_{a}": F[a] for a in F}}.items()}
    causal = {  # (flip_swap_X − flip_para) paired CI
        "belief": diff_ci(F["swap_b"], F["para"]),
        "reasoning": diff_ci(F["swap_r"], F["para"]),
        "both": diff_ci(F["swap_both"], F["para"]),
    }
    util = diff_ci(P["own"], P["swap_both"])  # own p_gt − swap_both p_gt (구 지표, 하위호환)
    # belief-only utility U_g (v2 §4 G-CC3 필수): reasoning 고정, belief 만 swap
    u_g = diff_ci(P["own"], P["swap_b"])
    # directional D_g: Pr[p_gt(own) > p_gt(swap_b)]  (belief 이 GT 방향으로 유용한가)
    dg_flags = [int(P["own"][i] > P["swap_b"][i]) for i in range(n)]
    d_g = bootstrap_ci(dg_flags)
    # correct-switch: belief-swap 로 top1 이 바뀐 샘플에서의 평균 GT-확률 하락
    # (아무 flip 이 아니라 belief 가 GT 관련 정보를 나른다는 신호)
    sb_flip = [i for i in range(n) if F["swap_b"][i]]
    cs_drop = ([P["own"][i] - P["swap_b"][i] for i in sb_flip]) or [0.0]
    correct_switch = {
        "flip_rate_swap_b": round(sum(F["swap_b"]) / n, 4),
        "mean_pgt_drop_on_flip": round(sum(cs_drop) / len(cs_drop), 4),
        "n_flip": len(sb_flip)}

    # acc 직교성: 정답/오답 샘플별 both-flip 비율
    c_idx = [i for i in range(n) if correct[i]]
    w_idx = [i for i in range(n) if not correct[i]]
    orth = {
        "acc_own": round(sum(correct) / n, 4),
        "flip_both_on_correct": round(sum(F["swap_both"][i] for i in c_idx) / max(1, len(c_idx)), 4),
        "flip_both_on_wrong": round(sum(F["swap_both"][i] for i in w_idx) / max(1, len(w_idx)), 4),
    }

    g_s3a = causal["both"][1] > 0            # causal(both) CI 하한 > 0
    g_s3b = util[1] > 0                       # (구) own−swap_both 유용성 CI 하한 > 0
    g_cc1 = causal["belief"][1] > 0          # belief-only sensitivity CI 하한 > 0
    g_cc3 = u_g[1] > 0                         # belief-only utility U_g CI 하한 > 0 (필수)
    summary = {
        "arm": args.arm, "n": n,
        "ci_95": {k: {"point": v[0], "lo": v[1], "hi": v[2]} for k, v in ci.items()},
        "causal_sensitivity_ci": {k: {"point": v[0], "lo": v[1], "hi": v[2]} for k, v in causal.items()},
        "utility_own_minus_swapboth_ci": {"point": util[0], "lo": util[1], "hi": util[2]},
        "utility_belief_only_ci": {"point": u_g[0], "lo": u_g[1], "hi": u_g[2]},
        "directional_dg_ci": {"point": d_g[0], "lo": d_g[1], "hi": d_g[2]},
        "correct_switch": correct_switch,
        "acc_orthogonality": orth,
        "gate_S3a_causal_real": bool(g_s3a),
        "gate_S3b_utility_real": bool(g_s3b),
        "gate_CC1_belief_sensitivity": bool(g_cc1),
        "gate_CC3_belief_only_utility": bool(g_cc3),
        "verdict": ("PASS — spine 확정 (U_g)" if (g_cc1 and g_cc3) else
                    "PARTIAL — sensitivity-only" if g_cc1 else "FAIL — 라인 종료 검토"),
    }
    out = runs_root() / "eval" / f"{args.arm}.harden_s3.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (runs_root() / "eval" / f"{args.arm}.harden_s3.records.json").write_text(
        json.dumps(out_rows, ensure_ascii=False))
    sw.finish(metrics={"verdict": summary["verdict"], "causal_both": causal["both"][0]})
    write_marker(f"S3H_{args.arm.upper()}_DONE", summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    vlm.close_readers()


if __name__ == "__main__":
    main()
