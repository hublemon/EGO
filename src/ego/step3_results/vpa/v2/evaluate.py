"""VPA v2 채점 — SR / mAcc / mIoU + **video-cluster** 부트스트랩 CI, 그리고 무비용 baseline.

지표 정의는 원본 VPA(Patel ICCV2023)와 동일하며, VLaMP 공식 구현
(`training/metrics/{success_rate,mean_intersection_over_union}.py`)과 의미 등가임을 실사 확인했다:
  SR   : 예측 T-시퀀스가 **순서까지** 정답과 완전 일치한 샘플 비율
  mAcc : 위치별 1[pred_i == gt_i] 평균 (순서 민감, 부분점수 인정)
  mIoU : 예측 집합 vs 정답 집합의 IoU를 샘플별로 구해 평균 (순서 무시)

**부트스트랩은 video 클러스터 단위**로 한다. 영상당 평균 12명 내외의 샘플이 상관돼 있어
샘플 단위 재표집은 CI를 과소추정한다(cesft_v2 게이트와 동일한 관행).

내장 baseline (프레임 불필요, 비용 ~0):
  random          : 어휘 균등 샘플
  most_probable   : 전역 빈도 top-T
  most_probable_goal : goal(=scenario)별 빈도 top-T
  wm_top1_repeat  : WM top-1 을 T회 반복 (WM 은 다음 1개만 예측 가능 — 정직한 퇴화형)
  wm_topk_rank    : WM 후보를 점수 내림차순으로 T개 (WM prior 를 계획으로 읽는 변형)

사용:
  PYTHONPATH=src python -m ego.step3_results.vpa.v2.evaluate \
      --gt runs/vpa_v2/vpa_v2_T3.json --vocab runs/vpa_v2/vocab.json --baselines
  ... --pred runs/vpa_v2/preds/qwen_backbone_T3.json --run-name qwen_backbone
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ego.step3_results.vpa.v2 import common as C


# ── 지표 ────────────────────────────────────────────────────────────────────
def per_sample(gt: list[str], pred: list[str], T: int) -> tuple[int, int, float]:
    """(성공 0/1, 위치일치 수, IoU) — 예측이 짧으면 빈 문자열로 패딩(오답 처리)."""
    pred = (list(pred) + [""] * T)[:T]
    gt = gt[:T]
    correct = sum(1 for a, b in zip(gt, pred) if a == b and a != "")
    gs, ps = set(gt), {x for x in pred if x}
    union = gs | ps
    iou = len(gs & ps) / len(union) if union else 0.0
    return int(correct == T), correct, iou


def aggregate(rows: list[tuple], T: int) -> dict:
    if not rows:
        return {"SR": 0.0, "mAcc": 0.0, "mIoU": 0.0, "n": 0}
    n = len(rows)
    return {
        "SR": 100.0 * sum(r[0] for r in rows) / n,
        "mAcc": 100.0 * sum(r[1] for r in rows) / (n * T),
        "mIoU": 100.0 * sum(r[2] for r in rows) / n,
        "n": n,
    }


def cluster_bootstrap(rows: list[tuple], clusters: list[str], T: int,
                      n_boot: int = 1000, seed: int = 0) -> dict:
    """video 단위로 클러스터를 재표집 — 같은 영상 샘플의 상관을 반영한 보수적 CI."""
    by_cluster: dict[str, list[tuple]] = defaultdict(list)
    for r, c in zip(rows, clusters):
        by_cluster[c].append(r)
    keys = list(by_cluster)
    if len(keys) < 2:
        return {"SR": [None, None], "mAcc": [None, None], "mIoU": [None, None], "n_clusters": len(keys)}

    rng = np.random.default_rng(seed)
    draws = {"SR": [], "mAcc": [], "mIoU": []}
    for _ in range(n_boot):
        picked = rng.integers(0, len(keys), len(keys))
        sample_rows = [r for i in picked for r in by_cluster[keys[i]]]
        m = aggregate(sample_rows, T)
        for k in draws:
            draws[k].append(m[k])
    out = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}
    out["n_clusters"] = len(keys)
    return out


def score(samples: list[dict], preds: dict[str, list[str]], vocab: list[str], T: int,
          n_boot: int, seed: int) -> dict:
    vset, rows, clusters, per_records = set(vocab), [], [], []
    missing, oov = 0, 0
    for s in samples:
        sid = s["sample_id"]
        if sid not in preds:
            missing += 1
            continue
        mapped = []
        for lab in preds[sid][:T]:
            m = C.map_to_vocab(lab, vset, vocab)
            if m not in vset:
                oov += 1
            mapped.append(m)
        gt = [C.normalize_label(x) for x in s["future_actions"][:T]]
        r = per_sample(gt, mapped, T)
        rows.append(r)
        clusters.append(s["video_uid"])
        per_records.append({"sample_id": sid, "video_uid": s["video_uid"],
                            "gt": gt, "pred": mapped, "success": r[0], "correct": r[1],
                            "iou": round(r[2], 3)})
    m = aggregate(rows, T)
    m["ci95"] = cluster_bootstrap(rows, clusters, T, n_boot, seed)
    m["n_videos"] = len(set(clusters))
    m["n_missing_pred"] = missing
    m["n_oov_mapped"] = oov
    m["coverage"] = round(100.0 * len(rows) / max(1, len(samples)), 2)
    m["reportable"] = missing == 0
    return m, per_records


# ── baseline 예측 생성 ──────────────────────────────────────────────────────
BASELINE_NAMES = ("random", "most_probable", "most_probable_goal",
                  "wm_top1_repeat", "wm_topk_rank")


def make_baselines(samples: list[dict], vocab: list[str], T: int, seed: int,
                   names: list[str] | None = None) -> dict[str, dict]:
    """정답을 보지 않는 예측기들. 빈도 통계는 **평가셋 자체**에서 뽑지 않고
    관측된 history(=과거)에서만 뽑아 정보 누출을 피한다.

    names 를 주면 요청한 baseline 자체만 생성한다. 특히 WM-only ablation 실행에서
    history 기반 baseline 을 부수적으로 계산하지 않도록 하는 엄격한 실행 범위 가드다.
    """
    selected = tuple(names or BASELINE_NAMES)
    unknown = set(selected) - set(BASELINE_NAMES)
    if unknown:
        raise ValueError(f"unknown baselines: {sorted(unknown)}")

    glob = Counter()
    per_goal: dict[str, Counter] = defaultdict(Counter)
    if {"most_probable", "most_probable_goal"} & set(selected):
        for s in samples:
            for a in s["observed_actions"]:
                glob[a] += 1
                per_goal[s.get("scenario", "")][a] += 1

    rng = random.Random(seed)
    out: dict[str, dict] = {name: {} for name in selected}
    gtop = [a for a, _ in glob.most_common(T)] or vocab[:T]
    for s in samples:
        sid = s["sample_id"]
        if "random" in out:
            out["random"][sid] = [rng.choice(vocab) for _ in range(T)]
        if "most_probable" in out:
            out["most_probable"][sid] = list(gtop)
        if "most_probable_goal" in out:
            gt_top = [a for a, _ in per_goal[s.get("scenario", "")].most_common(T)] or gtop
            out["most_probable_goal"][sid] = (gt_top + gtop)[:T]
        if "wm_top1_repeat" in out or "wm_topk_rank" in out:
            cands, scores = s["wm_candidates"], s["wm_scores"]
            order = sorted(range(len(cands)), key=lambda i: -scores[i])
            if "wm_top1_repeat" in out:
                out["wm_top1_repeat"][sid] = [cands[order[0]]] * T
            if "wm_topk_rank" in out:
                out["wm_topk_rank"][sid] = [cands[i] for i in order[:T]]
    return out


def fmt(m: dict, key: str) -> str:
    lo, hi = m["ci95"][key]
    ci = f" [{lo:.1f}, {hi:.1f}]" if lo is not None else ""
    return f"{m[key]:.2f}{ci}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--vocab", default="runs/vpa_v2/vocab.json")
    p.add_argument("--pred", default=None, help="preds json {sample_id: [labels]}")
    p.add_argument("--run-name", default=None)
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--baseline-names", nargs="+",
                   choices=["random", "most_probable", "most_probable_goal",
                            "wm_top1_repeat", "wm_topk_rank"],
                   default=None,
                   help="--baselines 중 저장·채점할 arm만 선택. 미지정 시 전부.")
    p.add_argument("--subset", default=None, help="sample_id 화이트리스트 json (예: frontier_subset_T3.json)")
    p.add_argument("--out-dir", default="runs/vpa_v2/metrics")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    samples = C.load_json(args.gt)
    T = samples[0]["horizon"]
    vocab = C.load_json(args.vocab)["labels"]
    tag = ""
    if args.subset:
        keep = set(C.load_json(args.subset)["sample_ids"])
        samples = [s for s in samples if s["sample_id"] in keep]
        tag = "_" + Path(args.subset).stem
    print(f"[info] T={T} · {len(samples)} samples · {len({s['video_uid'] for s in samples})} videos"
          f"{' · subset ' + args.subset if args.subset else ''}")

    results = {}
    if args.baselines:
        baseline_preds = make_baselines(samples, vocab, T, args.seed, args.baseline_names)
        for name, preds in baseline_preds.items():
            m, _ = score(samples, preds, vocab, T, args.n_boot, args.seed)
            results[name] = m
    if args.pred:
        preds = C.load_json(args.pred)
        name = args.run_name or Path(args.pred).stem
        m, recs = score(samples, preds, vocab, T, args.n_boot, args.seed)
        results[name] = m
        C.dump_json(Path(args.out_dir) / f"records_{name}_T{T}{tag}.json", recs)

    print(f"\n{'arm':<22} {'n':>5} {'SR':>18} {'mAcc':>18} {'mIoU':>18}  cov%")
    for name, m in results.items():
        flag = "" if m["reportable"] else "  ⚠partial"
        print(f"{name:<22} {m['n']:>5} {fmt(m,'SR'):>18} {fmt(m,'mAcc'):>18} "
              f"{fmt(m,'mIoU'):>18}  {m['coverage']}{flag}")

    out = Path(args.out_dir) / f"metrics_T{T}{tag}.json"
    prev = C.load_json(out) if out.is_file() else {}
    prev.update(results)
    C.dump_json(out, prev)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
