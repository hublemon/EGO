"""Step-1 probe checkpoint → Top-K support 덤프 + coverage 리포트 (Phase-0).

probe(AnticipationHead, V-JEPA2 feature 위 attentive probe)의 action head로
샘플별 Top-K (verb,noun) support를 만들고, 셔플·이름 복원·dev/heldout 분리까지
수행한다. GT는 coverage 측정과 라벨 기록에만 쓰고 후보 생성에는 관여하지 않는다.

출력:
  runs/retro3/data/support_{train,val}.jsonl   (SupportSample, history/future 비어 있음)
  runs/retro3/data/coverage.json               (cov@{1,5,10,15}, MRR, split별)
  runs/retro3/data/splits.json                 (val video-disjoint dev/heldout)

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.data.build_support \
      --config configs/step2_retrospection/goalstep_start_m1_lobs8.yaml [--split val] [--limit 64]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ego.step2_retrospection.contracts import TOP_K, SupportSample, assert_strict_contract
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, runs_root, write_marker

REPO = Path(__file__).resolve().parents[4]


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_name_maps(registry: dict, taxonomy: dict):
    """dense id → 이름. registry: raw('v'|'n' str) → dense, taxonomy: raw idx → 이름."""
    verbs, nouns = taxonomy["verbs"], taxonomy["nouns"]
    dense2verb = {d: verbs[int(r)] for r, d in registry["verb_classes"].items()}
    dense2noun = {d: nouns[int(r)] for r, d in registry["noun_classes"].items()}
    dense2action = {}
    for key, d in registry["action_classes"].items():
        rv, rn = key.split("|")
        dense2action[d] = (verbs[int(rv)], nouns[int(rn)])
    return dense2verb, dense2noun, dense2action


def make_split(video_uids: list[str], dev_ratio: float, seed: int) -> dict[str, str]:
    uids = sorted(set(video_uids))
    rng = random.Random(seed)
    rng.shuffle(uids)
    n_dev = max(1, int(len(uids) * dev_ratio))
    dev = set(uids[:n_dev])
    return {u: ("dev" if u in dev else "heldout") for u in uids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_start_m1_lobs8.yaml")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="스모크용 샘플 수 제한")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    export = Path(cfg["step1_export"]["root"])
    index_dir = Path(cfg["step1_export"]["index_dir"])
    cache_root = Path(cfg["shared_assets"]["feature_cache_dir"])
    tau_a = float(cfg["contract"]["tau_a"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # step1 코드 재사용 (step2_vlm_alignment 아님 — 허용)
    sys.path.insert(0, str(REPO / "src"))
    from ego.datasets.ego4d import z1_sample_id
    from ego.step1_action_anticipation.data.collator import anticipation_collate
    from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset
    from ego.step1_action_anticipation.models import AnticipationHead

    registry = json.loads((index_dir / "action_registry.json").read_text())
    taxonomy = json.loads((REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_verbnoun_taxonomy.json").read_text())
    dense2verb, dense2noun, dense2action = build_name_maps(registry, taxonomy)
    # parquet의 verb/noun_label은 raw(taxonomy) id
    raw_verb_name = lambda r: taxonomy["verbs"][int(r)]  # noqa: E731
    raw_noun_name = lambda r: taxonomy["nouns"][int(r)]  # noqa: E731

    state = torch.load(export / "best_action_top5.pt", map_location="cpu", weights_only=False)
    num_classes = state["num_classes"]

    out_dir = runs_root() / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    model = None
    coverage_report: dict = {"top_k": TOP_K}

    for split in splits:
        df = pd.read_parquet(index_dir / f"{split}.parquet")
        sample_ids = [z1_sample_id(row["clip_uid"], i) for i, row in df.iterrows()]
        if args.limit:
            df, sample_ids = df.iloc[: args.limit], sample_ids[: args.limit]
        ds = FeatureCacheDataset(sample_ids, cache_root / split)
        if len(ds) != len(sample_ids):
            raise RuntimeError(f"feature cache 불일치: index {len(sample_ids)} vs cache {len(ds)}")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=anticipation_collate, num_workers=4, pin_memory=True)

        if model is None:
            embed_dim = ds[0]["video"].shape[-1]
            model = AnticipationHead(
                num_verb_classes=num_classes["verb"], num_noun_classes=num_classes["noun"],
                num_action_classes=num_classes["action"], embed_dim=embed_dim,
                num_heads=16, depth=4).to(device)
            model.load_state_dict(state["model_state"])
            model.eval()

        # split 분리 (val만) — video-disjoint
        vid_split = {}
        if split == "val":
            vid_split = make_split(df["video_uid"].tolist(),
                                   float(cfg["splits"]["dev_ratio"]), int(cfg["splits"]["split_seed"]))
            (out_dir / "splits.json").write_text(json.dumps(vid_split, indent=1))

        out_path = out_dir / f"support_{split}.jsonl"
        out_path.unlink(missing_ok=True)
        sw = StatusWriter(f"S0_support_{split}", total=len(ds))
        n_cov = {1: 0, 5: 0, 10: 0, 15: 0}
        mrr_sum, n_done = 0.0, 0

        with torch.no_grad():
            row_iter = df.reset_index(drop=True).iterrows()
            for batch in loader:
                feats = batch["video"].to(device)
                logits = model(feats)["action"].float()
                probs = torch.softmax(logits, dim=-1)
                topv, topi = probs.topk(TOP_K, dim=-1)
                ranks_all = probs.argsort(dim=-1, descending=True)

                for b in range(feats.shape[0]):
                    _, row = next(row_iter)
                    sid = batch["sample_id"][b]
                    gt_action_id = int(row["action_label"])
                    # GT rank (1-based, full 293)
                    gt_rank = int((ranks_all[b] == gt_action_id).nonzero()[0, 0].item()) + 1
                    mrr_sum += 1.0 / gt_rank
                    for k in n_cov:
                        n_cov[k] += int(gt_rank <= k)

                    cand_ids = topi[b].tolist()
                    cand_probs = topv[b].tolist()
                    names = [f"{dense2action[i][0]} {dense2action[i][1]}" for i in cand_ids]
                    order = list(range(TOP_K))
                    random.Random(f"shuffle:{sid}").shuffle(order)  # 결정적 셔플
                    cands = [names[j] for j in order]
                    scores = [cand_probs[j] for j in order]

                    s = SupportSample(
                        sample_id=sid, video_uid=row["video_uid"], clip_uid=row["clip_uid"],
                        obs_start_sec=float(row["obs_start_sec"]), obs_end_sec=float(row["obs_end_sec"]),
                        target_start_sec=float(row["obs_end_sec"]) + tau_a,
                        gt_verb=raw_verb_name(row["verb_label"]), gt_noun=raw_noun_name(row["noun_label"]),
                        candidates=cands, wm_scores=scores)
                    assert_strict_contract(s, tau_a=tau_a)
                    rec = asdict(s)
                    rec["gt_rank"] = gt_rank
                    rec["scenario"] = row["scenario"]
                    rec["split"] = vid_split.get(row["video_uid"], split)
                    append_jsonl(out_path, rec)
                    n_done += 1
                sw.update(done=n_done, metrics={
                    "cov@10": round(n_cov[10] / n_done, 4), "cov@5": round(n_cov[5] / n_done, 4)})

        cov = {f"cov@{k}": round(v / n_done, 4) for k, v in n_cov.items()}
        cov["mrr"] = round(mrr_sum / n_done, 4)
        cov["n"] = n_done
        coverage_report[split] = cov
        sw.finish(metrics=cov, message=f"{split} done n={n_done}")
        print(f"[S0] {split}: {cov}")

    (out_dir / "coverage.json").write_text(json.dumps(coverage_report, indent=1))
    write_marker("S0_SUPPORT_DONE", coverage_report)


if __name__ == "__main__":
    main()
