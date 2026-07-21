#!/usr/bin/env python3
"""eval_harness_v2.py — 측정 하네스 v2 (핸드오프 §7 개선 1 · 최우선).

**왜 v2 인가** (핸드오프 §6 RC3):
  - subset 만 바꿔도 acc 가 0.264 → 0.302 로 움직였다 (Δ0.038). 하루 종일 주장한 효과 크기는 0.02–0.03.
  - n=500 이항 se = 0.020 → 두 arm 비교의 se_diff ≈ 0.028. **오늘의 "목표 ≥0.26 vs 실측 0.264"는
    애초에 검출 불가능한 크기였다.**
  - `eval_battery` 는 `do_sample=False`(greedy) 라 재현 3회가 바이트 동일했다 —
    그건 파이프라인 재현성의 확인이지 표본 잡음 강건성의 증거가 아니다.

**v2 가 하는 일** (지표 정의는 새로 만들지 않는다):
  1. heldout **전량** 1회 생성 (기본 `--limit 0`; 기존 배터리는 앞 500행만 봤다)
  2. 같은 생성 결과를 **2개 disjoint subset** 으로 갈라 나란히 보고
     → subset 흔들림이 곧 "이 측정계의 실측 바닥 잡음"이다
  3. acc(및 ③ causal_sensitivity)의 **부트스트랩 95% CI**
  4. **최소 검출 가능 효과(MDE)를 항상 출력** — 이보다 작은 임계값은 사전등록 금지

지표 정의는 `eval_battery.score_predictions` / `summarize_metrics` 를 **그대로 import** 한다.
③ 은 `eval_belief_swap` 의 `swap − control` 정의를 그대로 재집계한다. 숫자 비교 가능성 유지가
이 스크립트의 존재 이유이므로, 여기서 새 정의를 만들지 않는다.

CPU 전용 재분석 모드: `--from_records <*.records.jsonl>` 로 이미 있는 생성 결과를 다시 채점한다
(모델 로드 없음 — GPU 를 쓰지 않고 오늘까지의 실행을 전부 v2 기준으로 재해석할 수 있다).

예)
  # GPU: heldout 전량 재측정 (FAA 기준선)
  python scripts/step2/eval_harness_v2.py --jsonl $J1F --adapter $FAA \\
      --out $BAT/v2_faa_full.json --device cuda:0
  # CPU: 오늘 실행을 재해석 + ③ CI
  python scripts/step2/eval_harness_v2.py --jsonl $J1F \\
      --from_records $BAT/b0p12_gen_1f.records.jsonl \\
      --swap_records $BAT/swap_b0p12.records.jsonl --out $BAT/v2_b0p12.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_battery as EB  # noqa: E402  (지표 정의의 단일 출처)
from ego.common.run_provenance import write_run_config  # noqa: E402

# 유의수준 0.05(양측)·검정력 0.80 의 정규근사 상수 z_{0.975}+z_{0.80} = 1.960 + 0.842
Z_MDE = 2.802
Z_95 = 1.960


# ── subset 분할 ─────────────────────────────────────────────────────────────
def split_indices(records, mode: str, seed: int = 42):
    """disjoint 2분할. 반환 (name_a, idx_a, name_b, idx_b).

    contiguous : 앞절반/뒤절반 — 오늘 관측된 Δ0.038(첫 500행 vs 다음 500행)을 그대로 재현하는
                 **보수적(최악) 추정**. heldout 이 참가자 ID 순이면 subject shift 가 그대로 들어온다.
    interleave : 짝수/홀수 index — subject 분포가 균형이라 잡음 하한을 본다.
    hash       : sample_id 해시 패리티 — 순서에 의존하지 않는 재현 가능한 분할.
    """
    n = len(records)
    if mode == "interleave":
        return ("even", list(range(0, n, 2)), "odd", list(range(1, n, 2)))
    if mode == "hash":
        a, b = [], []
        for i, r in enumerate(records):
            h = hashlib.sha1(f'{seed}:{r["sample_id"]}'.encode()).hexdigest()
            (a if int(h[-1], 16) % 2 == 0 else b).append(i)
        return ("hash0", a, "hash1", b)
    half = n // 2
    return ("first_half", list(range(half)), "second_half", list(range(half, n)))


# ── 부트스트랩 ──────────────────────────────────────────────────────────────
def _resample_means(values, n_boot: int, seed: int) -> list[float]:
    """복원추출 재표집 평균 n_boot 개. numpy 가 있으면 벡터화, 없으면 순수 파이썬."""
    n = len(values)
    try:
        import numpy as np
        arr = np.asarray(values, dtype=float)
        rs = np.random.default_rng(seed)
        outs = []
        for start in range(0, n_boot, 1000):   # 청크 분할 — (n_boot × n) 인덱스 배열 회피
            k = min(1000, n_boot - start)
            outs.extend(arr[rs.integers(0, n, size=(k, n))].mean(axis=1).tolist())
        return outs
    except ImportError:
        rng = random.Random(seed)
        return [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """평균의 퍼센타일 부트스트랩 CI. values 는 0/1 지시자 리스트 (acc = mean(correct))."""
    n = len(values)
    if n == 0:
        return {"n": 0, "point": None, "lo": None, "hi": None, "se": None}
    point = sum(values) / n
    means = _resample_means(values, n_boot, seed)
    means.sort()
    lo = means[max(0, int(alpha / 2 * n_boot) - 1)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    mu = sum(means) / n_boot
    se = math.sqrt(sum((x - mu) ** 2 for x in means) / max(1, n_boot - 1))
    return {"n": n, "point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "se": round(se, 4), "n_boot": n_boot}


def bootstrap_paired_diff(pairs, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """(a_i, b_i) 쌍 재표집으로 mean(b) − mean(a) 의 CI. ③ = swap − control 에 사용."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "point": None, "lo": None, "hi": None, "se": None}
    point = sum(b - a for a, b in pairs) / n
    diffs = _resample_means([b - a for a, b in pairs], n_boot, seed)   # 쌍 단위 재표집
    diffs.sort()
    lo = diffs[max(0, int(alpha / 2 * n_boot) - 1)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    mu = sum(diffs) / n_boot
    se = math.sqrt(sum((x - mu) ** 2 for x in diffs) / max(1, n_boot - 1))
    return {"n": n, "point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "se": round(se, 4), "n_boot": n_boot}


def mde_block(p: float, n: int, subset_spread: float | None):
    """최소 검출 가능 효과. **사전등록 임계값은 반드시 이 값 위에서 정한다.**

    se_arm   : 한 arm 의 이항 se = sqrt(p(1-p)/n)
    se_diff  : 두 독립 arm(동일 n) 차이의 se = sqrt(2)·se_arm
    mde_80   : α=0.05(양측)·검정력 0.80 에서 검출 가능한 최소 차이 = 2.802·se_diff
    floor    : mde_80 과 **실측 subset 흔들림** 중 큰 값 — 통계 잡음만이 아니라 표본 구성
               흔들림도 넘어야 실재하는 효과다 (RC3: 오늘 Δ0.038 이 관측됐다)
    """
    if not n:
        return None
    se_arm = math.sqrt(max(0.0, p * (1 - p)) / n)
    se_diff = math.sqrt(2) * se_arm
    mde80 = Z_MDE * se_diff
    out = {"acc": round(p, 4), "n": n,
           "se_arm_binomial": round(se_arm, 4),
           "se_diff_two_arms": round(se_diff, 4),
           "mde_2se": round(2 * se_diff, 4),
           "mde_80power_a05": round(mde80, 4),
           "subset_spread_observed": (round(subset_spread, 4)
                                      if subset_spread is not None else None)}
    out["min_preregisterable_effect"] = round(
        max(mde80, subset_spread or 0.0), 4)
    return out


# ── ③ causal sensitivity 재집계 (eval_belief_swap 정의 그대로) ───────────────
def causal_from_swap_records(path: str, n_boot: int, seed: int):
    """swap records → control/swap action_change 와 causal_sensitivity(+CI).

    eval_belief_swap 은 각 cond 의 action_change 를 `changed / n` (n = usable trace 수)로
    정의한다. 여기서도 동일하게 sample_id 별 (control, swap) 쌍을 맞춰 재집계한다.
    """
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["sample_id"], {})[r["cond"]] = bool(r["changed"])
    pairs = [(float(d["control"]), float(d["swap"])) for d in by.values()
             if "control" in d and "swap" in d]
    if not pairs:
        return None
    ctrl = [a for a, _ in pairs]
    swp = [b for _, b in pairs]
    return {
        "records_src": path,
        "n_paired": len(pairs),
        "n_rows": len(rows),
        "control_action_change": round(sum(ctrl) / len(ctrl), 4),
        "swap_action_change": round(sum(swp) / len(swp), 4),
        # eval_belief_swap 과 동일하게 **각 비율을 먼저 4자리 반올림한 뒤** 뺀다 (수치 대조 보존)
        "causal_sensitivity": round(round(sum(swp) / len(swp), 4)
                                    - round(sum(ctrl) / len(ctrl), 4), 4),
        "causal_sensitivity_ci95": bootstrap_paired_diff(pairs, n_boot, seed),
        "control_ci95": bootstrap_ci(ctrl, n_boot, seed),
        "swap_ci95": bootstrap_ci(swp, n_boot, seed),
    }


# ── 생성 ────────────────────────────────────────────────────────────────────
def generate_preds(args, convs):
    """eval_battery 와 동일한 로드·생성 경로 (greedy · do_sample=False 유지)."""
    import torch
    from tqdm import tqdm
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
    preds = []
    for i in tqdm(range(0, len(convs), args.batch_size), desc="generate"):
        preds.extend(EB.generate_batch(model, processor, convs[i:i + args.batch_size],
                                       args.max_new_tokens,
                                       multi_image_dir=args.multi_image_dir))
    return preds


def preds_from_records(path: str, convs):
    """기존 *.records.jsonl 의 completion 을 sample_id 로 정렬해 재사용 (CPU 전용)."""
    recs = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["sample_id"]] = r["completion"]
    preds, keep = [], []
    for i, c in enumerate(convs):
        if c["sample_id"] in recs:
            preds.append(recs[c["sample_id"]])
            keep.append(i)
    return preds, keep


def main():
    ap = argparse.ArgumentParser(
        description="측정 하네스 v2 — heldout 전량 + 2 disjoint subset + bootstrap CI + MDE")
    ap.add_argument("--jsonl", required=True, help="heldout jsonl (기본: 전량 사용)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="0=heldout 전량(v2 기본). 0 이 아니면 앞 N행만 — v1 호환용이며 권장하지 않는다")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--multi_image_dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_memory", action="store_true", help="배터리 ①: memory_context 공란화")
    ap.add_argument("--action_only", action="store_true")
    ap.add_argument("--history_only", action="store_true", help="배터리 ②: 전 샘플 프레임 마스킹")
    ap.add_argument("--from_records", default=None,
                    help="CPU 전용 재분석: 기존 eval_battery *.records.jsonl 을 재채점(생성 없음)")
    ap.add_argument("--swap_records", default=None,
                    help="③ 재집계용 eval_belief_swap *.records.jsonl (있으면 CI 까지 출력)")
    ap.add_argument("--subset_mode", choices=["contiguous", "interleave", "hash"],
                    default="contiguous",
                    help="disjoint 2분할 방식. contiguous=보수적(오늘의 Δ0.038 재현 경로)")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--baseline_json", default=None,
                    help="비교 기준선(v2 산출 json 또는 eval_battery json). 있으면 Δ와 판정을 출력")
    ap.add_argument("--no_records_out", action="store_true", help="*.records.jsonl 을 쓰지 않는다")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 개선 0: 이 평가 실행의 출처도 산출물로 남긴다
    write_run_config(out.parent, args,
                     data_paths=[p for p in (args.jsonl, args.from_records,
                                             args.swap_records, args.adapter) if p],
                     filename=f"{out.stem}.run_config.json",
                     extra={"runner": "eval_harness_v2"})

    # 프롬프트 빌더 전역 플래그 — eval_battery 와 동일 규약
    EB.T.NO_MEMORY = args.no_memory
    if args.action_only:
        EB.T.ACTION_ONLY = True
    if args.history_only:
        EB.T.MASK_FRAME_PROB = 1.0
        EB.T.BLANK_IMAGE_PATH = EB.T._prepare_blank_image(str(out.parent))
    if args.multi_image_dir:
        EB.T.JOINT_FRAME_DESC_4 = (
            "1. Four first-person frames sampled over the last 4 seconds, given as four\n"
            "   separate images in order (4.0s ago, 2.7s ago, 1.3s ago, now).")

    convs, raws = EB.load_convs(args.jsonl, args.limit or None)
    total_rows = sum(1 for l in open(args.jsonl, encoding="utf-8") if l.strip())
    print(f"[load] heldout rows={total_rows} usable={len(convs)} "
          f"limit={args.limit or 'ALL'} no_memory={args.no_memory}")

    if args.from_records:
        preds, keep = preds_from_records(args.from_records, convs)
        convs = [convs[i] for i in keep]
        raws = [raws[i] for i in keep]
        print(f"[records] {len(preds)} completions 재사용 → {args.from_records} (GPU 미사용)")
    else:
        preds = generate_preds(args, convs)

    m, records = EB.score_predictions(preds, convs, raws)
    full = EB.summarize_metrics(m)

    # ── subset 병기 ────────────────────────────────────────────────────────
    na, ia, nb, ib = split_indices(records, args.subset_mode)
    subs = {}
    for name, idx in ((na, ia), (nb, ib)):
        sm, _ = EB.score_predictions([preds[i] for i in idx], [convs[i] for i in idx],
                                     [raws[i] for i in idx])
        subs[name] = EB.summarize_metrics(sm)
    spread = None
    if subs[na]["acc"] is not None and subs[nb]["acc"] is not None:
        spread = abs(subs[na]["acc"] - subs[nb]["acc"])
    # 분할 방식마다 흔들림이 다르다(실측: contiguous 0.008 vs interleave 0.080) —
    # 하나만 보면 바닥 잡음을 과소평가한다. 전 모드를 재집계해 **최댓값**을 MDE 바닥으로 쓴다.
    spread_all = {}
    for mode in ("contiguous", "interleave", "hash"):
        xa, xia, xb, xib = split_indices(records, mode)
        accs = []
        for idx in (xia, xib):
            if not idx:
                accs.append(None); continue
            accs.append(sum(corr_i for corr_i in
                            (1.0 if records[i]["correct"] else 0.0 for i in idx)) / len(idx))
        spread_all[mode] = (round(abs(accs[0] - accs[1]), 4)
                            if None not in accs else None)
    spread_floor = max([v for v in spread_all.values() if v is not None] or [0.0])

    # ── 부트스트랩 CI (acc = mean(correct) — summarize_metrics 의 acc 와 동일 정의) ──
    corr = [1.0 if r["correct"] else 0.0 for r in records]
    boot = {
        "acc_full": bootstrap_ci(corr, args.n_boot, args.boot_seed),
        f"acc_{na}": bootstrap_ci([corr[i] for i in ia], args.n_boot, args.boot_seed),
        f"acc_{nb}": bootstrap_ci([corr[i] for i in ib], args.n_boot, args.boot_seed),
        "g2_acc_full": bootstrap_ci([1.0 if r["correct"] else 0.0 for r in records if r["g2"]],
                                    args.n_boot, args.boot_seed),
        "wm_follow_full": bootstrap_ci([1.0 if r["wm_follow"] else 0.0 for r in records],
                                       args.n_boot, args.boot_seed),
    }
    mde = mde_block(full["acc"] or 0.0, full["n"], spread_floor)
    if mde:
        mde["subset_spread_by_mode"] = spread_all
        mde["subset_spread_primary"] = (round(spread, 4) if spread is not None else None)

    causal = causal_from_swap_records(args.swap_records, args.n_boot, args.boot_seed) \
        if args.swap_records else None

    baseline = None
    if args.baseline_json and Path(args.baseline_json).exists():
        bj = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        b_full = bj.get("full", bj)
        b_ci = ((bj.get("bootstrap") or {}).get("acc_full") or {})
        baseline = {"src": args.baseline_json, "acc": b_full.get("acc"), "n": b_full.get("n"),
                    "ci95": [b_ci.get("lo"), b_ci.get("hi")] if b_ci else None,
                    "delta_acc": (round((full["acc"] or 0) - (b_full.get("acc") or 0), 4)
                                  if b_full.get("acc") is not None else None)}
        if b_ci.get("hi") is not None and boot["acc_full"]["lo"] is not None:
            # 사전등록 판정(핸드오프 §8): "acc CI 하한 > 기준선 CI 상한"
            baseline["ci_separated"] = bool(boot["acc_full"]["lo"] > b_ci["hi"])
        if mde and baseline.get("delta_acc") is not None:
            baseline["exceeds_mde"] = bool(abs(baseline["delta_acc"])
                                           >= mde["min_preregisterable_effect"])

    summary = {
        "harness": "eval_harness_v2",
        "jsonl": args.jsonl, "heldout_rows_total": total_rows,
        "limit": args.limit or None, "model": args.model_name, "adapter": args.adapter,
        "from_records": args.from_records,
        "no_memory": args.no_memory, "history_only": args.history_only,
        "decoding": "greedy(do_sample=False)",
        "subset_mode": args.subset_mode,
        "full": full,
        "subsets": subs,
        "subset_acc_spread": (round(spread, 4) if spread is not None else None),
        "subset_acc_spread_by_mode": spread_all,
        "bootstrap": boot,
        "mde": mde,
        "causal": causal,
        "baseline": baseline,
        "max_new_tokens": args.max_new_tokens, "max_pixels": args.max_pixels,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_records_out:
        with out.with_suffix(".records.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 콘솔 리포트 ────────────────────────────────────────────────────────
    b = boot["acc_full"]
    print("\n=== eval_harness_v2 ===")
    print(f"heldout {args.jsonl}  rows={total_rows}  scored={full['n']}  "
          f"({'전량' if not args.limit else f'limit={args.limit}'})")
    print(f"acc(full)   = {full['acc']}  95% CI [{b['lo']}, {b['hi']}]  boot se={b['se']}")
    print(f"subset[{na}] n={subs[na]['n']:>5}  acc={subs[na]['acc']}  "
          f"CI [{boot[f'acc_{na}']['lo']}, {boot[f'acc_{na}']['hi']}]")
    print(f"subset[{nb}] n={subs[nb]['n']:>5}  acc={subs[nb]['acc']}  "
          f"CI [{boot[f'acc_{nb}']['lo']}, {boot[f'acc_{nb}']['hi']}]")
    print(f"subset Δacc = {summary['subset_acc_spread']} ({args.subset_mode})   "
          f"전 분할 모드: {spread_all}")
    print("            ▲ 측정계의 실측 바닥 잡음 — 이보다 작은 '개선'은 주장 불가")
    print(f"g2_acc      = {full['g2_acc']} (n={full['g2_n']})  "
          f"CI [{boot['g2_acc_full']['lo']}, {boot['g2_acc_full']['hi']}]")
    if mde:
        print(f"\n--- 최소 검출 가능 효과 (n={mde['n']}, p={mde['acc']}) ---")
        print(f"  이항 se(1 arm)          = {mde['se_arm_binomial']}")
        print(f"  se(두 arm 차)           = {mde['se_diff_two_arms']}")
        print(f"  MDE (α=.05, power=.80)  = {mde['mde_80power_a05']}")
        print(f"  실측 subset 흔들림(최대) = {mde['subset_spread_observed']} "
              f"{mde.get('subset_spread_by_mode')}")
        print(f"  ▶ 사전등록 가능한 최소 효과 = {mde['min_preregisterable_effect']}")
        print("    이보다 작은 임계값을 사전등록하면 결과와 무관하게 해석 불가다.")
    if causal:
        c = causal["causal_sensitivity_ci95"]
        print(f"\n--- ③ causal_sensitivity (n_paired={causal['n_paired']}) ---")
        print(f"  control={causal['control_action_change']}  swap={causal['swap_action_change']}")
        print(f"  ③ = {causal['causal_sensitivity']}  95% CI [{c['lo']}, {c['hi']}]  se={c['se']}")
        if c["lo"] is not None and c["lo"] <= 0 <= c["hi"]:
            print("  ▶ CI 가 0 을 포함한다 — 인과 효과가 있다고 말할 수 없다.")
    if baseline:
        print(f"\n--- vs 기준선 {baseline['src']} ---")
        print(f"  기준 acc={baseline['acc']} CI={baseline['ci95']}  Δ={baseline['delta_acc']}  "
              f"CI 분리={baseline.get('ci_separated')}  MDE 초과={baseline.get('exceeds_mde')}")
    print(f"\n[done] → {out}")


if __name__ == "__main__":
    main()
