"""두 arm의 **짝지은** 차이와 그 CI — video-cluster paired bootstrap.

arm별 CI를 각각 보고 겹치는지 보는 것은 잘못이다(difference of significance ≠ significance
of difference). 같은 표본에서 두 arm을 나란히 재표집해 **차이의 분포**를 직접 구한다.
cesft_v2 게이트(`paired_G-ACC1` 등)와 동일한 관행이며, 클러스터 단위라 영상 내 상관을 반영한다.

핵심 용도: **frames arm − blind arm** = 프레임 기여분. 이것이 VPA v2 재작성의 정당성을 재는 수치다.

사용:
  PYTHONPATH=src python -m ego.step3_results.vpa.v2.paired \
      --gt runs/vpa_v2/vpa_v2_T3.json --subset runs/vpa_v2/frames_subset_T3.json \
      --a runs/vpa_v2/preds/qwen_backbone_T3.json --a-name qwen3vl_frames \
      --b runs/vpa_v2/preds/qwen_blind_T3.json  --b-name qwen3vl_blind
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ego.step3_results.vpa.v2 import common as C
from ego.step3_results.vpa.v2.evaluate import aggregate, per_sample


def rows_for(samples: list[dict], preds: dict, vocab: list[str], T: int):
    """공통 표본에 대해 (sample_id, video_uid, row) 목록. 양쪽 모두 예측이 있는 것만."""
    vset = set(vocab)
    out = {}
    for s in samples:
        p = preds.get(s["sample_id"])
        if p is None:
            continue
        mapped = [C.map_to_vocab(x, vset, vocab) for x in p[:T]]
        gt = [C.normalize_label(x) for x in s["future_actions"][:T]]
        out[s["sample_id"]] = (s["video_uid"], per_sample(gt, mapped, T))
    return out


def paired_delta(a_rows: dict, b_rows: dict, T: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """A − B. 클러스터(video)를 복원추출하고, 뽑힌 클러스터의 **같은 샘플들**로 양쪽을 동시에 계산."""
    common = sorted(set(a_rows) & set(b_rows))
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for sid in common:
        by_cluster[a_rows[sid][0]].append(sid)
    keys = list(by_cluster)

    point_a = aggregate([a_rows[s][1] for s in common], T)
    point_b = aggregate([b_rows[s][1] for s in common], T)
    delta = {k: point_a[k] - point_b[k] for k in ("SR", "mAcc", "mIoU")}

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in delta}
    for _ in range(n_boot):
        picked = rng.integers(0, len(keys), len(keys))
        sids = [s for i in picked for s in by_cluster[keys[i]]]
        ma = aggregate([a_rows[s][1] for s in sids], T)
        mb = aggregate([b_rows[s][1] for s in sids], T)
        for k in draws:
            draws[k].append(ma[k] - mb[k])

    out = {"n_paired": len(common), "n_clusters": len(keys),
           "A": {k: point_a[k] for k in delta}, "B": {k: point_b[k] for k in delta},
           "delta": delta, "ci95": {}, "significant": {}}
    for k, v in draws.items():
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        out["ci95"][k] = [lo, hi]
        out["significant"][k] = bool(lo > 0 or hi < 0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--vocab", default="runs/vpa_v2/vocab.json")
    p.add_argument("--subset", default=None)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.add_argument("--out-dir", default="runs/vpa_v2/metrics")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    samples = C.load_json(args.gt)
    T = samples[0]["horizon"]
    if args.subset:
        keep = set(C.load_json(args.subset)["sample_ids"])
        samples = [s for s in samples if s["sample_id"] in keep]
    vocab = C.load_json(args.vocab)["labels"]

    a = rows_for(samples, C.load_json(args.a), vocab, T)
    b = rows_for(samples, C.load_json(args.b), vocab, T)
    res = paired_delta(a, b, T, args.n_boot)
    res["A_name"], res["B_name"], res["horizon"] = args.a_name, args.b_name, T

    print(f"paired {args.a_name} − {args.b_name} · n={res['n_paired']} · clusters={res['n_clusters']}\n")
    print(f"{'metric':>6}  {args.a_name:>16}  {args.b_name:>16}  {'Δ':>8}  {'CI95':>18}  sig")
    for k in ("SR", "mAcc", "mIoU"):
        lo, hi = res["ci95"][k]
        print(f"{k:>6}  {res['A'][k]:>16.2f}  {res['B'][k]:>16.2f}  {res['delta'][k]:>+8.2f}  "
              f"[{lo:>+6.2f},{hi:>+6.2f}]  {'YES' if res['significant'][k] else 'no'}")

    out = Path(args.out_dir) / f"paired_{args.a_name}_vs_{args.b_name}_T{T}.json"
    C.dump_json(out, res)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
