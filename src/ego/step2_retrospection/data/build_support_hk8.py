"""Phase-1 history-context K=8 WM → Top-K support 덤프 (retro4, end−1s 계약).

build_support.py의 history-K8 대응판. 차이점 (2026-07-23 브리지 조사 확정):
  - sample_id는 parquet 컬럼을 그대로 사용 (strict-next 필터로 1,081/254행이
    빠져 있어 위치 기반 재계산 z1_sample_id(clip_uid, i)는 오답).
  - target_start_sec은 parquet 컬럼 사용 — horizon 가변(≥1.0s, 평균 12.8s).
    assert_strict_contract(tau_exact=False)로 최소 gap만 검증.
  - 추론은 raw feature cache가 아니라 파생 store([17,1024] summaries + frozen
    visual logits, sample_id 키)를 HistoryContextResidualHead에 넣고
    outputs["fused"]["action"]을 쓴다.

출력: build_support.py와 동일 스키마 (RETRO3_RUNS=runs/retro4 권장)
  {runs}/data/support_{train,val}.jsonl · coverage.json · splits.json

사용:
  RETRO3_RUNS=runs/retro4 PYTHONPATH=src python -m ego.step2_retrospection.data.build_support_hk8 \
      --config configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml [--split val] [--limit 64]
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import yaml

from ego.step2_retrospection.contracts import TOP_K, SupportSample, assert_strict_contract
from ego.step2_retrospection.data.build_support import build_name_maps, load_cfg, make_split
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, runs_root, write_marker

REPO = Path(__file__).resolve().parents[4]
K_HISTORY = 8


class DerivedStore:
    """goalstep_history_context_store 셰이드 로더 — sample_id → summary/visual_logits.

    jihun2 PreloadedHistoryStore의 최소 재구현 (참조 금지 정책 — 계약은
    manifest.json 기준: summaries fp16 [N,17,1024], visual_logits fp32 dict).
    """

    def __init__(self, root: Path, split: str):
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.num_classes = self.manifest["num_classes"]
        self.summaries: list[torch.Tensor] = []
        self.visual: dict[str, list[torch.Tensor]] = {}
        self.row: dict[str, tuple[int, int]] = {}  # sample_id → (shard, row)
        shards = sorted((root / split).glob("*.pt"))
        if not shards:
            raise RuntimeError(f"store 셰이드 없음: {root / split}")
        for si, p in enumerate(shards):
            d = torch.load(p, map_location="cpu", weights_only=True)
            self.summaries.append(d["summaries"])
            for h, t in d["visual_logits"].items():
                self.visual.setdefault(h, []).append(t)
            for ri, sid in enumerate(d["sample_ids"]):
                self.row[sid] = (si, ri)

    def summary(self, sid: str) -> torch.Tensor:
        si, ri = self.row[sid]
        return self.summaries[si][ri]

    def visual_logits(self, sid: str) -> dict[str, torch.Tensor]:
        si, ri = self.row[sid]
        return {h: self.visual[h][si][ri] for h in self.visual}


def build_model(state: dict, num_classes: dict, device):
    from ego.step1_action_anticipation.models.history_context_head import HistoryContextResidualHead
    arch = (state.get("metadata") or {}).get("architecture") or {}
    model = HistoryContextResidualHead(
        num_classes=num_classes,
        embed_dim=int(arch.get("embed_dim", 1024)),
        max_history=int(arch.get("max_history", K_HISTORY)),
        segment_pooler_heads=int(arch.get("segment_pooler_heads", 16)),
        transformer_heads=int(arch.get("transformer_heads", 16)),
        transformer_layers=int(arch.get("transformer_layers", 2)),
        transformer_mlp_ratio=float(arch.get("transformer_mlp_ratio", 4.0)),
        transformer_dropout=float(arch.get("transformer_dropout", 0.1)),
        segment_dropout=float(arch.get("segment_dropout", 0.3)),
        recency_scale_sec=float(arch.get("recency_scale_sec", 300.0)),
    ).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model


def batch_inputs(rows: pd.DataFrame, store: DerivedStore, device):
    """parquet 행 배치 → 모델 입력 텐서 (jihun2 HistoryContextDataset 계약 미러)."""
    B = len(rows)
    summaries = torch.zeros((B, 1 + K_HISTORY, 17, 1024), dtype=torch.float32)
    mask = torch.zeros((B, K_HISTORY), dtype=torch.bool)
    delta = torch.zeros((B, K_HISTORY), dtype=torch.float32)
    level = torch.full((B, K_HISTORY), -1, dtype=torch.long)
    vis = {h: torch.zeros((B, store.num_classes[h]), dtype=torch.float32)
           for h in ("verb", "noun", "action")}
    for b, (_, row) in enumerate(rows.iterrows()):
        sid = row["sample_id"]
        summaries[b, 0] = store.summary(sid).float()
        vl = store.visual_logits(sid)
        for h in vis:
            vis[h][b] = vl[h].float()
        for i in range(1, K_HISTORY + 1):
            if bool(row[f"history_{i}_mask"]):
                mask[b, i - 1] = True
                summaries[b, i] = store.summary(row[f"history_{i}_cache_sample_id"]).float()
                delta[b, i - 1] = float(row[f"history_{i}_delta_t_sec"])
                level[b, i - 1] = int(row[f"history_{i}_level_id"])
    return (summaries.to(device), mask.to(device), delta.to(device), level.to(device),
            {h: t.to(device) for h, t in vis.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    export = Path(cfg["step1_export"]["root"])
    index_dir = Path(cfg["step1_export"]["index_dir"])
    store_root = Path(cfg["shared_assets"]["derived_store_dir"])
    tau_a = float(cfg["contract"]["tau_a"])  # 최소 gap
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    registry = json.loads((index_dir / "action_registry.json").read_text())
    taxonomy = json.loads((REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_verbnoun_taxonomy.json").read_text())
    _, _, dense2action = build_name_maps(registry, taxonomy)
    raw_verb_name = lambda r: taxonomy["verbs"][int(r)]  # noqa: E731
    raw_noun_name = lambda r: taxonomy["nouns"][int(r)]  # noqa: E731

    state = torch.load(export / "best_action_top5.pt", map_location="cpu", weights_only=False)

    out_dir = runs_root() / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    model = None
    coverage_report: dict = {"top_k": TOP_K, "prior": "phase1_history_k8_fused"}

    for split in splits:
        df = pd.read_parquet(index_dir / f"{split}.parquet")
        if args.limit:
            df = df.iloc[: args.limit]
        store = DerivedStore(store_root, split)
        missing = [s for s in df["sample_id"] if s not in store.row]
        if missing:
            raise RuntimeError(f"store에 없는 sample_id {len(missing)}개 (예: {missing[:3]})")
        if model is None:
            model = build_model(state, store.num_classes, device)

        vid_split = {}
        if split == "val":
            vid_split = make_split(df["video_uid"].tolist(),
                                   float(cfg["splits"]["dev_ratio"]), int(cfg["splits"]["split_seed"]))
            (out_dir / "splits.json").write_text(json.dumps(vid_split, indent=1))

        out_path = out_dir / f"support_{split}.jsonl"
        out_path.unlink(missing_ok=True)
        sw = StatusWriter(f"S0_support_{split}", total=len(df))
        n_cov = {1: 0, 5: 0, 10: 0, 15: 0}
        mrr_sum, n_done = 0.0, 0

        with torch.no_grad():
            for start in range(0, len(df), args.batch_size):
                chunk = df.iloc[start:start + args.batch_size]
                summaries, mask, delta, level, vis = batch_inputs(chunk, store, device)
                logits = model(summaries, mask, delta, level, vis)["fused"]["action"].float()
                probs = torch.softmax(logits, dim=-1)
                topv, topi = probs.topk(TOP_K, dim=-1)
                ranks_all = probs.argsort(dim=-1, descending=True)

                for b, (_, row) in enumerate(chunk.iterrows()):
                    sid = row["sample_id"]
                    gt_action_id = int(row["action_label"])  # dense (조사에서 전행 검증)
                    gt_rank = int((ranks_all[b] == gt_action_id).nonzero()[0, 0].item()) + 1
                    mrr_sum += 1.0 / gt_rank
                    for k in n_cov:
                        n_cov[k] += int(gt_rank <= k)

                    names = [f"{dense2action[i][0]} {dense2action[i][1]}" for i in topi[b].tolist()]
                    cand_probs = topv[b].tolist()
                    order = list(range(TOP_K))
                    random.Random(f"shuffle:{sid}").shuffle(order)  # 결정적 셔플 (retro3 동일)

                    s = SupportSample(
                        sample_id=sid, video_uid=row["video_uid"], clip_uid=row["clip_uid"],
                        obs_start_sec=float(row["obs_start_sec"]), obs_end_sec=float(row["obs_end_sec"]),
                        target_start_sec=float(row["target_start_sec"]),
                        gt_verb=raw_verb_name(row["verb_label"]), gt_noun=raw_noun_name(row["noun_label"]),
                        candidates=[names[j] for j in order],
                        wm_scores=[cand_probs[j] for j in order],
                        boundary_flag=bool(row["boundary_flag"]))
                    assert_strict_contract(s, tau_a=tau_a, tau_exact=False)
                    rec = asdict(s)
                    rec["gt_rank"] = gt_rank
                    rec["scenario"] = row["scenario"]
                    rec["split"] = vid_split.get(row["video_uid"], split)
                    rec["target_horizon_sec"] = round(float(row["target_horizon_sec"]), 3)
                    rec["annotation_level"] = row["annotation_level"]
                    append_jsonl(out_path, rec)
                    n_done += 1
                sw.update(done=n_done, metrics={
                    "cov@10": round(n_cov[10] / n_done, 4), "cov@5": round(n_cov[5] / n_done, 4)})

        cov = {f"cov@{k}": round(v / n_done, 4) for k, v in n_cov.items()}
        cov["mrr"] = round(mrr_sum / n_done, 4)
        cov["n"] = n_done
        coverage_report[split] = cov
        sw.finish(metrics=cov, message=f"{split} done n={n_done}")
        print(f"[S0:hk8] {split}: {cov}")
        del store

    (out_dir / "coverage.json").write_text(json.dumps(coverage_report, indent=1))
    write_marker("S0_SUPPORT_DONE", coverage_report)


if __name__ == "__main__":
    main()
