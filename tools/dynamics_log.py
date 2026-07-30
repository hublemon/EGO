#!/usr/bin/env python3
"""dynamics_log.py — main.tex Figure 2(fig:dynamics) 형식의 궤적 로그 + 입력 교란 행렬.

Figure 2 의 읽는 방식을 그대로 로그로 만든다: **저장된 체크포인트를 재채점한 점들**을
누적 optimizer step 축에 놓고, 조건(Answer-Only / Prospection / EGO / replay-free)별로
같은 축을 비교한다. 보간하지 않는다 — 마커 사이에 값이 없다.

논문 어휘 ↔ 내부 arm 이름
  WM Top-1     : 정책을 거치지 않는 순위 모방 바닥 (records 의 wm_top1_correct)
  Base VLM     : base            (후학습 이전)
  Answer-Only  : cand_free / ans_*   (같은 정답·같은 예산, 후보 경계 없음)
  Prospection  : theta_ce / pro_*    (경계 안에서 판별하도록 학습)
  EGO          : sft_r15_c / retro_* (+ Retrospection)
  replay-free  : r00_*               (replay anchor 제거 branch)

축 3종 (Figure 2 패널)
  (a) Within-Boundary Accuracy — covered 셋 SelAcc, WM Top-1 바닥 병기
  (b) Prior retention          — WM top-1 이 맞은 케이스의 유지율 (G1)
  (c) Belief-action coupling   — flip(swap_belief) − flip(paraphrase), 별도 프로토콜
      ※ (c)는 (a)(b)와 추정대상이 다르다 — 논문 캡션의 경고를 그대로 승계한다.

추가 축 (논문 §Ablation 대응) — **입력 교란 행렬**
  history strip / no-image / 둘 다 / other-video donor history 4종 × 4조건,
  Δ = acc(full) − acc(perturbed) 를 video-cluster paired bootstrap 으로 통일 산출.
  (기존 perturb_verdict_*.json 은 sample bootstrap — CI 규약만 다르다.)

CLI:
  python3 tools/dynamics_log.py --curve_run runs/cesft_v2_fp_curve \
      --perturb_run runs/cesft_v2_fp_c --out runs/cesft_v2_fp_c/eval/dynamics_log.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_boot import (cluster_key, load_context_video_uids,  # noqa: E402
                         paired_cluster_delta, read_jsonl)
from strip_metrics import arm_strip, scored  # noqa: E402

# ── Figure 2 의 계열 정의: (논문 이름, 색 슬롯, [(x_step, ckpt 파일 stem), …]) ──
# x = 누적 optimizer step. stage-2(EGO·replay-free)는 논문과 같이 523 + s 로 놓는다.
STAGE1_END = 523
CURVE = {
    "Answer-Only": {"slot": "ref", "stage": 1, "points": [
        (100, "ans_s100"), (200, "ans_s200"), (300, "ans_s300"),
        (400, "ans_s400"), (500, "ans_s500"), (STAGE1_END, "ans_final")]},
    "Prospection": {"slot": "s1", "stage": 1, "points": [
        (100, "pro_s100"), (200, "pro_s200"), (300, "pro_s300"),
        (400, "pro_s400"), (500, "pro_s500"), (STAGE1_END, "pro_final")]},
    "EGO": {"slot": "s2", "stage": 2, "points": [
        (STAGE1_END + 100, "retro_s100"), (STAGE1_END + 150, "retro_s150"),
        (STAGE1_END + 200, "retro_s200"), (STAGE1_END + 250, "retro_s250"),
        (STAGE1_END + 300, "retro_s300"), (STAGE1_END + 347, "retro_final")]},
    "replay-free": {"slot": "dash", "stage": 2, "points": [
        (STAGE1_END + 100, "r00_s100"), (STAGE1_END + 200, "r00_s200"),
        (STAGE1_END + 347, "r00_final")]},
}

PERTURB = {  # 파일 접미사 → 논문에서 부르는 개입 이름
    "_nohist": "history 제거",
    "_othervideo": "다른 영상의 이력으로 교체",
    "_noimage": "프레임 제거",
    "_nohist_noimage": "이력·프레임 모두 제거",
}
COND = [("base", "Base VLM"), ("cand_free", "Answer-Only"),
        ("theta_ce", "Prospection"), ("sft_r15_c", "EGO")]


def pct(x, nd=1):
    return None if x is None else round(100 * x, nd)


def ckpt_axes(run: Path, stem: str) -> dict | None:
    """{stem}.json(배터리 요약) + harden_paired(있으면)에서 세 축을 뽑는다."""
    p = run / "eval" / f"{stem}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {
        "ckpt": stem,
        "n": d.get("n"),
        "wb_acc": pct(d.get("acc")),                  # (a)
        "wm_top1_floor": pct(d.get("L0_wm_top1")),
        "beats_floor": bool(d.get("beats_L0")),
        "retention": pct(d.get("G1_retention")),      # (b)
        "retention_n": d.get("G1_n"),
        "correction": pct(d.get("G2_correction")),
        "correction_n": d.get("G2_n"),
        "malformed": pct(d.get("malformed_rate")),
        "e2e_full_set": pct(d.get("acc_full_equiv")),
    }
    hp = run / "eval" / f"{stem}.harden_paired.json"
    if hp.is_file():                                   # (c) 별도 프로토콜
        h = json.loads(hp.read_text(encoding="utf-8"))
        sens = h.get("sensitivity", {})
        key = next((k for k in ("swap_b_shared", "swap_b") if k in sens), None)
        if key:
            s = sens[key]
            out["belief_coupling"] = {"point": pct(s.get("point")),
                                      "ci": [pct(s.get("lo")), pct(s.get("hi"))],
                                      "n": h.get("n"), "donor": h.get("donor_arm")}
    return out


def paired_vs(run: Path, arm_a: str, arm_b: str, metric: str,
              n_boot: int, seed: int) -> dict | None:
    """같은 covered 셋에서 arm_a − arm_b 의 짝지은 차이 (video-cluster bootstrap)."""
    ctx_map = load_context_video_uids(run)
    ra = {r["sample_id"]: r for r in read_jsonl(run / "eval" / f"{arm_a}.records.jsonl")}
    rb = {r["sample_id"]: r for r in read_jsonl(run / "eval" / f"{arm_b}.records.jsonl")}
    sids = sorted(s for s in ra if s in rb and ra[s].get("gt_in_support"))
    if not sids:
        return None
    by_cluster: dict[str, list] = {}
    for s in sids:
        wm = bool(ra[s].get("wm_top1_correct"))
        by_cluster.setdefault(cluster_key(s, ctx_map), []).append(
            (scored(ra[s]), wm, scored(rb[s]), wm))
    clusters = sorted(by_cluster)
    d = paired_cluster_delta(clusters, by_cluster, metric, metric, n_boot=n_boot, seed=seed)
    return {"metric": metric, "arm_a": arm_a, "arm_b": arm_b,
            "point_a": pct(d["point_a"]), "point_b": pct(d["point_b"]),
            "delta_pp": pct(d["point"], 2),
            "ci": [pct(d["lo"], 2), pct(d["hi"], 2)],
            "n_paired": len(sids), "n_clusters": len(clusters),
            "excludes_zero": bool(d["lo"] > 0 or d["hi"] < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve_run", default="runs/cesft_v2_fp_curve")
    ap.add_argument("--perturb_run", default="runs/cesft_v2_fp_c")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    curve, perturb = Path(args.curve_run), Path(args.perturb_run)
    res: dict = {
        "figure": "main.tex fig:dynamics 형식 — 저장 체크포인트 재채점, 마커 사이 보간 없음",
        "curve_run": str(curve), "perturb_run": str(perturb),
        "axes": {"a": "Within-Boundary Accuracy (covered SelAcc)",
                 "b": "Prior retention (WM top-1 정답 케이스)",
                 "c": "Belief-action coupling — 별도 프로토콜, (a)(b)와 추정대상 다름"},
        "series": {}, "reference": {}, "paired": {}, "perturbation": {},
    }

    # Base VLM — 학습 전 단일 점 (x=0)
    base = ckpt_axes(curve, "base")
    if base is None:                                    # curve 런에 요약이 없으면 records 로 계산
        rows = [r for r in read_jsonl(curve / "eval" / "base.records.jsonl")
                if r.get("gt_in_support")]
        n = len(rows) or 1
        wmc = [r for r in rows if r.get("wm_top1_correct")]
        base = {"ckpt": "base", "n": len(rows),
                "wb_acc": round(100 * sum(scored(r) for r in rows) / n, 1),
                "wm_top1_floor": round(100 * len(wmc) / n, 1),
                "retention": round(100 * sum(scored(r) for r in wmc) / max(1, len(wmc)), 1),
                "retention_n": len(wmc)}
    hp = curve / "eval" / "base.harden_paired.json"
    if hp.is_file():
        h = json.loads(hp.read_text(encoding="utf-8"))
        s = h.get("sensitivity", {}).get("swap_b_shared") or {}
        if s:
            base["belief_coupling"] = {"point": pct(s.get("point")),
                                       "ci": [pct(s.get("lo")), pct(s.get("hi"))],
                                       "n": h.get("n")}
    res["reference"]["Base VLM"] = {"x_step": 0, **base}

    for name, spec in CURVE.items():
        pts = []
        for x, stem in spec["points"]:
            a = ckpt_axes(curve, stem)
            if a:
                pts.append({"x_step": x, "stage": spec["stage"], **a})
        res["series"][name] = {"slot": spec["slot"], "stage": spec["stage"], "points": pts}
        if pts:
            print(f"[{name:12s}] {len(pts)} ckpt · WB-Acc "
                  f"{' → '.join(str(p['wb_acc']) for p in pts)}", flush=True)

    # 종점 짝지은 비교 (Figure 2 캡션이 인용하는 검정들)
    for label, (a, b, m) in {
        "EGO vs Answer-Only · WB-Acc": ("retro_final", "ans_final", "SelAcc"),
        "EGO vs Answer-Only · retention": ("retro_final", "ans_final", "G1"),
        "Prospection vs Answer-Only · WB-Acc": ("pro_final", "ans_final", "SelAcc"),
        "EGO vs Prospection · WB-Acc": ("retro_final", "pro_final", "SelAcc"),
        "EGO vs replay-free · WB-Acc": ("retro_final", "r00_final", "SelAcc"),
        "EGO vs replay-free · retention": ("retro_final", "r00_final", "G1"),
        # 캡션의 replay anchor 비교는 **같은 stage-2 step** 끼리다 (final 끼리가 아니다).
        "EGO vs replay-free @s100 · WB-Acc": ("retro_s100", "r00_s100", "SelAcc"),
        "EGO vs replay-free @s100 · retention": ("retro_s100", "r00_s100", "G1"),
        "EGO vs replay-free @s200 · WB-Acc": ("retro_s200", "r00_s200", "SelAcc"),
        "EGO vs replay-free @s200 · retention": ("retro_s200", "r00_s200", "G1"),
        "EGO vs WM floor · WB-Acc": ("retro_final", "retro_final", "WMtop1"),
    }.items():
        if a == b:      # 자기 WM 바닥 대비
            d = paired_vs(curve, a, b, "SelAcc", args.n_boot, args.seed)
            if d:
                floor = paired_vs(curve, a, b, "WMtop1", args.n_boot, args.seed)
                d = {**d, "metric": "SelAcc − WMtop1",
                     "delta_pp": round(d["point_a"] - floor["point_a"], 2),
                     "ci": None, "note": "점추정만 — 자기-WM 대비 CI 는 paired_boot G-ACC1 참조"}
        else:
            d = paired_vs(curve, a, b, m, args.n_boot, args.seed)
        if d:
            res["paired"][label] = d
            ci = f"[{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]" if d.get("ci") else "(CI 없음)"
            print(f"  {label:38s} {d['delta_pp']:+6.2f}pp {ci}", flush=True)

    # 입력 교란 행렬 — 전 조건·전 개입을 cluster CI 로 통일
    for arm, name in COND:
        res["perturbation"][name] = {"arm": arm, "modes": {}}
        for sfx, label in PERTURB.items():
            r = arm_strip(perturb, arm, args.n_boot, args.seed, "", sfx)
            if r.get("error"):
                res["perturbation"][name]["modes"][label] = {"error": r["error"]}
                continue
            sel = r["delta"]["SelAcc"]
            res["perturbation"][name]["modes"][label] = {
                "suffix": sfx, "full": sel["hist"], "perturbed": sel["strip"],
                "delta_pp": sel["delta_pp"], "ci": sel["ci"],
                "excludes_zero": bool(sel["ci"][0] > 0 or sel["ci"][1] < 0),
                "delta_GADR_pp": r["delta"]["GADR"]["delta_pp"],
                "delta_retention_pp": r["delta"]["G1"]["delta_pp"],
                "n_paired": r["n_paired"], "n_clusters": r["n_clusters"],
                "by_hlen": r["delta_selacc_by_hlen"],
            }
        got = {k: v for k, v in res["perturbation"][name]["modes"].items() if "delta_pp" in v}
        print(f"[perturb {name:12s}] " + " · ".join(
            f"{k} {v['delta_pp']:+.1f}[{v['ci'][0]:+.1f},{v['ci'][1]:+.1f}]" for k, v in got.items()),
            flush=True)

    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
        print(f"\n[dynamics_log] wrote {args.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
