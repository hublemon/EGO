#!/usr/bin/env python3
"""paired_boot.py — PAIRED VIDEO-CLUSTER BOOTSTRAP Δ with confidence intervals.

Statistical-gate utility for the EGO Step-2 "candidate-CE ↔ projected-SFT
combination" experiment (v2 methodology §2; combination handoff G-NH; Appendix-A
T-ACC).

The unit of resampling is the *video cluster* (video_uid), not the individual
sample: consecutive frames from one video are correlated, so a naive per-sample
bootstrap understates the variance. Each bootstrap iteration draws the LIST OF
CLUSTERS with replacement; the resampled sample set is the union (with
multiplicity) of all samples belonging to the drawn clusters. Both arms are
scored on that SAME resampled set, so the Δ = metric_A − metric_B is properly
paired.

Metrics (computed on the COVERED set = gt_in_support==True and not malformed):
    SelAcc = mean(correct)
    GADR   = mean(correct | wm_top1_correct==False)   (gain beyond imitation)
    G1     = mean(correct | wm_top1_correct==True)     (retention)
    WMtop1 = mean(wm_top1_correct)                      (imitation floor)

Gates:
    G-ACC1  : Δ = SelAcc(arm) − WMtop1(arm) ; pass if delta.lo > 0.
    G-DELTA : Δ = metric(arm_a) − metric(arm_b) ; pass if delta.lo > 0.
    G-NH    : non-harm (A=after, B=before) ; pass if
                  [SelAcc(A) − SelAcc(B)].lo >= -0.01  AND
                  [GADR(A)   − GADR(B)].point >= -0.02.

Pure stdlib, GPU-free, deterministic (seeded). Importable helpers:
    load_arm_records, metric_on, paired_cluster_delta, eval_gate.

CLI:
    python3 tools/paired_boot.py --run runs/cesft_v2 --arm_a <armA> \\
        [--arm_b <armB>] --gate {G-ACC1,G-DELTA,G-NH} \\
        [--metric SelAcc|GADR|G1] [--n_boot 2000] [--seed 0] --out <path.json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

METRICS = ("SelAcc", "GADR", "G1", "WMtop1")


# ----------------------------------------------------------------------------- IO
def read_jsonl(p: Path) -> list[dict]:
    """Read a .jsonl file into a list of dicts; missing file -> []; robust to junk."""
    if not p.is_file():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_context_video_uids(run: Path) -> dict[str, str]:
    """Map sample_id -> video_uid using runs/<run>/data/context_val.jsonl."""
    ctx = read_jsonl(run / "data" / "context_val.jsonl")
    m: dict[str, str] = {}
    for r in ctx:
        sid = r.get("sample_id")
        vid = r.get("video_uid")
        if sid is not None and vid:
            m[sid] = vid
    return m


def cluster_key(sample_id: str, ctx_map: dict[str, str]) -> str:
    """True video_uid (cluster key) for a sample_id.

    Prefer the context_val `video_uid` field; fall back to stripping the
    trailing `_{idx}` only when the sample_id is absent from context_val
    (sample_id looks like {video_uid}_{idx} but video_uid contains underscores).
    """
    if sample_id in ctx_map:
        return ctx_map[sample_id]
    return sample_id.rsplit("_", 1)[0]


def load_arm_records(run: Path, arm: str) -> dict[str, dict]:
    """Load one arm's covered records, keyed by sample_id.

    COVERED = gt_in_support==True and not malformed. Malformed / error rows and
    rows not in support are dropped. Returns {sample_id: {correct, wm_top1_correct}}.
    """
    rows = read_jsonl(run / "eval" / f"{arm}.records.jsonl")
    out: dict[str, dict] = {}
    for r in rows:
        sid = r.get("sample_id")
        if sid is None:
            continue
        if r.get("malformed"):
            continue
        if r.get("error") is not None:
            continue
        if not r.get("gt_in_support"):
            continue
        out[sid] = {
            "correct": bool(r.get("correct")),
            "wm_top1_correct": bool(r.get("wm_top1_correct")),
        }
    return out


# ------------------------------------------------------------------------- metrics
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def metric_on(name: str, correct: list[bool], wm: list[bool]) -> float:
    """Compute a metric over aligned per-sample arrays correct[] and wm_top1_correct[]."""
    if name == "SelAcc":
        return _mean([1.0 if c else 0.0 for c in correct])
    if name == "WMtop1":
        return _mean([1.0 if w else 0.0 for w in wm])
    if name == "GADR":
        vals = [1.0 if c else 0.0 for c, w in zip(correct, wm) if not w]
        return _mean(vals)
    if name == "G1":
        vals = [1.0 if c else 0.0 for c, w in zip(correct, wm) if w]
        return _mean(vals)
    raise ValueError(f"unknown metric {name!r}")


# ----------------------------------------------------------------- paired bootstrap
def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,100]) on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def paired_cluster_delta(
    clusters: list[str],
    by_cluster: dict[str, list[tuple[bool, bool, bool, bool]]],
    metric_a: str,
    metric_b: str,
    self_wm_b: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired video-cluster bootstrap of Δ = metric_A(arm A) − metric_B(arm B).

    by_cluster maps cluster -> list of per-sample tuples
        (correct_a, wm_a, correct_b, wm_b).
    metric_a is scored on arm A's (correct_a, wm_a); metric_b on arm B's
    (correct_b, wm_b). If self_wm_b is True, metric_b is scored against arm A's
    own wm_top1_correct (used for G-ACC1: SelAcc(θ) − WM-top1 of the SAME arm).

    Returns {point, lo, hi, point_a, point_b}. lo/hi are the 2.5/97.5 percentiles.
    """
    def eval_delta(cls: list[str]) -> tuple[float, float, float]:
        ca: list[bool] = []
        wa: list[bool] = []
        cb: list[bool] = []
        wb: list[bool] = []
        for c in cls:
            for (c_a, w_a, c_b, w_b) in by_cluster[c]:
                ca.append(c_a)
                wa.append(w_a)
                cb.append(c_b)
                wb.append(w_b)
        va = metric_on(metric_a, ca, wa)
        if self_wm_b:
            # arm B metric measured against arm A's own wm_top1_correct
            vb = metric_on(metric_b, wa, wa)
        else:
            vb = metric_on(metric_b, cb, wb)
        return va - vb, va, vb

    point, point_a, point_b = eval_delta(clusters)

    rng = random.Random(seed)
    n = len(clusters)
    deltas: list[float] = []
    if n > 0:
        for _ in range(n_boot):
            drawn = [clusters[rng.randrange(n)] for _ in range(n)]
            d, _, _ = eval_delta(drawn)
            deltas.append(d)
    deltas.sort()
    return {
        "point": point,
        "lo": _percentile(deltas, 2.5),
        "hi": _percentile(deltas, 97.5),
        "point_a": point_a,
        "point_b": point_b,
    }


def _build_paired(
    run: Path,
    arm_a: str,
    arm_b: str | None,
) -> tuple[dict[str, list], list[str], int, str | None]:
    """Inner-join arm_a (and arm_b if given) on sample_id, group by cluster.

    Returns (by_cluster, clusters, n_paired, error). When arm_b is None the arm B
    slots are filled with arm A's own values (for self-WM gates).
    """
    ctx_map = load_context_video_uids(run)
    rec_a = load_arm_records(run, arm_a)
    if not rec_a:
        return {}, [], 0, f"no covered records for arm_a={arm_a!r}"

    if arm_b is None:
        common = set(rec_a)
        rec_b = rec_a
    else:
        rec_b = load_arm_records(run, arm_b)
        if not rec_b:
            return {}, [], 0, f"no covered records for arm_b={arm_b!r}"
        common = set(rec_a) & set(rec_b)

    if not common:
        return {}, [], 0, "empty inner-join (no sample_id covered in both arms)"

    by_cluster: dict[str, list] = {}
    for sid in sorted(common):
        key = cluster_key(sid, ctx_map)
        a = rec_a[sid]
        b = rec_b[sid]
        by_cluster.setdefault(key, []).append(
            (a["correct"], a["wm_top1_correct"], b["correct"], b["wm_top1_correct"])
        )
    clusters = sorted(by_cluster)
    return by_cluster, clusters, len(common), None


# ------------------------------------------------------------------------- gates
def eval_gate(
    gate: str,
    run: Path,
    arm_a: str,
    arm_b: str | None = None,
    metric: str = "SelAcc",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Evaluate a statistical gate. Always returns a JSON-able dict.

    On any data problem the result carries `error` and `pass: False` (never raises)
    so a chain's marker logic controls flow.
    """
    base = {
        "run": str(run),
        "gate": gate,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "metric": metric,
        "n_paired": 0,
        "n_clusters": 0,
        "point_a": None,
        "point_b": None,
        "delta": {"point": None, "lo": None, "hi": None},
        "ci_low_gt0": False,
        "pass": False,
        "extra": {},
    }

    if gate == "G-ACC1":
        by_cluster, clusters, n_paired, err = _build_paired(run, arm_a, None)
        if err:
            base["error"] = err
            return base
        d = paired_cluster_delta(
            clusters, by_cluster, "SelAcc", "WMtop1",
            self_wm_b=True, n_boot=n_boot, seed=seed,
        )
        base["metric"] = "SelAcc-WMtop1"
        base["n_paired"] = n_paired
        base["n_clusters"] = len(clusters)
        base["point_a"] = d["point_a"]
        base["point_b"] = d["point_b"]
        base["delta"] = {"point": d["point"], "lo": d["lo"], "hi": d["hi"]}
        base["ci_low_gt0"] = d["lo"] > 0
        base["pass"] = d["lo"] > 0
        base["extra"] = {"selacc": d["point_a"], "wmtop1": d["point_b"]}
        return base

    if gate == "G-DELTA":
        if arm_b is None:
            base["error"] = "G-DELTA requires --arm_b"
            return base
        if metric not in ("SelAcc", "GADR", "G1", "WMtop1"):
            base["error"] = f"unsupported metric {metric!r} for G-DELTA"
            return base
        by_cluster, clusters, n_paired, err = _build_paired(run, arm_a, arm_b)
        if err:
            base["error"] = err
            return base
        d = paired_cluster_delta(
            clusters, by_cluster, metric, metric,
            n_boot=n_boot, seed=seed,
        )
        base["n_paired"] = n_paired
        base["n_clusters"] = len(clusters)
        base["point_a"] = d["point_a"]
        base["point_b"] = d["point_b"]
        base["delta"] = {"point": d["point"], "lo": d["lo"], "hi": d["hi"]}
        base["ci_low_gt0"] = d["lo"] > 0
        base["pass"] = d["lo"] > 0
        return base

    if gate == "G-NH":
        if arm_b is None:
            base["error"] = "G-NH requires --arm_b (A=after, B=before)"
            return base
        by_cluster, clusters, n_paired, err = _build_paired(run, arm_a, arm_b)
        if err:
            base["error"] = err
            return base
        sel = paired_cluster_delta(
            clusters, by_cluster, "SelAcc", "SelAcc", n_boot=n_boot, seed=seed,
        )
        gadr = paired_cluster_delta(
            clusters, by_cluster, "GADR", "GADR", n_boot=n_boot, seed=seed,
        )
        selacc_ok = sel["lo"] >= -0.01
        gadr_ok = gadr["point"] >= -0.02
        base["metric"] = "SelAcc&GADR"
        base["n_paired"] = n_paired
        base["n_clusters"] = len(clusters)
        base["point_a"] = sel["point_a"]
        base["point_b"] = sel["point_b"]
        # primary reported delta is SelAcc (the CI-gated one)
        base["delta"] = {"point": sel["point"], "lo": sel["lo"], "hi": sel["hi"]}
        base["ci_low_gt0"] = sel["lo"] > 0
        base["pass"] = bool(selacc_ok and gadr_ok)
        base["extra"] = {
            "selacc_delta": {
                "point": sel["point"], "lo": sel["lo"], "hi": sel["hi"],
                "point_a": sel["point_a"], "point_b": sel["point_b"],
            },
            "gadr_delta": {
                "point": gadr["point"], "lo": gadr["lo"], "hi": gadr["hi"],
                "point_a": gadr["point_a"], "point_b": gadr["point_b"],
            },
            "selacc_ok": selacc_ok,   # lo >= -0.01
            "gadr_ok": gadr_ok,       # point >= -0.02
        }
        return base

    base["error"] = f"unknown gate {gate!r}"
    return base


# --------------------------------------------------------------------------- CLI
def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


def _summary(res: dict) -> str:
    lines = []
    lines.append(f"gate={res['gate']}  run={res['run']}")
    lines.append(
        f"arm_a={res['arm_a']}  arm_b={res['arm_b']}  metric={res['metric']}"
    )
    if res.get("error"):
        lines.append(f"ERROR: {res['error']}")
        lines.append(f"pass={res['pass']}")
        return "\n".join(lines)
    lines.append(
        f"n_paired={res['n_paired']}  n_clusters={res['n_clusters']}"
    )
    lines.append(
        f"point_a={_fmt(res['point_a'])}  point_b={_fmt(res['point_b'])}"
    )
    d = res["delta"]
    lines.append(
        f"delta: point={_fmt(d['point'])}  "
        f"CI95=[{_fmt(d['lo'])}, {_fmt(d['hi'])}]  ci_low_gt0={res['ci_low_gt0']}"
    )
    if res["gate"] == "G-NH":
        ex = res["extra"]
        sd, gd = ex["selacc_delta"], ex["gadr_delta"]
        lines.append(
            f"  SelAcc Δ point={_fmt(sd['point'])} lo={_fmt(sd['lo'])} "
            f"(need lo>=-0.01) ok={ex['selacc_ok']}"
        )
        lines.append(
            f"  GADR   Δ point={_fmt(gd['point'])} "
            f"CI95=[{_fmt(gd['lo'])},{_fmt(gd['hi'])}] "
            f"(need point>=-0.02) ok={ex['gadr_ok']}"
        )
    lines.append(f"PASS={res['pass']}")
    return "\n".join(lines)


def _default_out(run: Path, gate: str, arm_a: str, arm_b: str | None) -> Path:
    name = f"paired_{gate}_{arm_a}"
    if arm_b is not None:
        name += f"_vs_{arm_b}"
    return run / "eval" / f"{name}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="paired_boot.py",
        description="Paired video-cluster bootstrap Δ with CIs for EGO Step-2 gates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run", required=True, help="run dir, e.g. runs/cesft_v2")
    ap.add_argument("--arm_a", required=True, help="arm A name (eval/<arm>.records.jsonl)")
    ap.add_argument("--arm_b", default=None, help="arm B name (required for G-DELTA/G-NH)")
    ap.add_argument(
        "--gate", required=True, choices=["G-ACC1", "G-DELTA", "G-NH"],
    )
    ap.add_argument(
        "--metric", default="SelAcc", choices=list(METRICS),
        help="metric for G-DELTA (default SelAcc)",
    )
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args(argv)

    run = Path(args.run)
    res = eval_gate(
        args.gate, run, args.arm_a, args.arm_b,
        metric=args.metric, n_boot=args.n_boot, seed=args.seed,
    )

    out = Path(args.out) if args.out else _default_out(
        run, args.gate, args.arm_a, args.arm_b
    )
    try:
        os.makedirs(out.parent, exist_ok=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        wrote = str(out)
    except OSError as e:
        wrote = f"<failed: {e}>"

    print(_summary(res))
    print(f"[written] {wrote}")
    # exit 0 always so the chain's marker logic controls flow
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
