"""VPA v2 데이터셋 빌더 — step2 검증셋(context_val.jsonl) → 프레임-조건 VPA 샘플.

표본 계약 (plan §4):
  · covered  : GT(다음 action) ∈ WM top-10 후보  — step2 평가 표본과 동일 모집단
  · horizon  : future action 이 T개 이상 존재
  · gap cap  : target_start - obs_end(원본) <= GAP_CAP_SEC (기본 5s)
  · 창 재정의: obs_end := target_start - 1s, obs_start := obs_end - 4s (경계-안전)

WM top-10 은 **표본 선정에만** 쓰고 예측 후보로 주지 않는다. 후보는 전체 어휘(293)이며,
이래야 VPA(전체 행동 공간에서의 계획)로서 원본과 정합한다 (plan §4 D1).

사용:
  PYTHONPATH=src python -m ego.step3_results.vpa.v2.build_dataset \
      --context runs/cesft_v2/data/context_val.jsonl --out-dir runs/vpa_v2
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ego.step3_results.vpa.v2 import common as C


def load_goals(goalstep_val: Path) -> dict[str, str]:
    """video_uid → goal_description (없으면 goal_category 정리본)."""
    data = json.loads(goalstep_val.read_text())
    out = {}
    for v in data.get("videos", []):
        desc = (v.get("goal_description") or "").strip()
        if not desc:
            cat = (v.get("goal_category") or "").strip()
            desc = cat.split(":", 1)[-1].replace("_", " ").strip().lower()
        out[v["video_uid"]] = desc
    return out


def action_str(a: dict) -> str:
    return C.normalize_label(f"{a['verb']} {a['noun']}")


def build(rows: list[dict], goals: dict[str, str], horizon: int, gap_cap: float,
          split: str = "heldout") -> tuple[list[dict], dict]:
    samples, drops = [], Counter()
    for r in rows:
        # split 계약: context_val 은 dev(학습 중 probe 용) + heldout(평가용)이 섞여 있다.
        # dev 는 step2 학습 루프가 probe_gen 으로 들여다본 표본이므로 평가에 쓰면 안 된다
        # (video-disjoint 라 누출은 경미하나, cesft_v2 보고 수치가 heldout 기준이라 비교도 깨진다).
        if split != "both" and r.get("split") != split:
            drops["wrong_split"] += 1
            continue
        gt = C.normalize_label(f"{r['gt_verb']} {r['gt_noun']}")
        if gt not in {C.normalize_label(c) for c in r["candidates"]}:
            drops["not_covered"] += 1
            continue
        future = r.get("future") or []
        if len(future) < horizon:
            drops["short_future"] += 1
            continue
        gap = r["target_start_sec"] - r["obs_end_sec"]
        if gap > gap_cap:
            drops["gap_over_cap"] += 1
            continue

        target_start = float(r["target_start_sec"])
        obs_start, obs_end = C.observation_window(target_start)
        if obs_end <= obs_start:  # 영상 시작부 클램프 — 관측이 비면 버린다
            drops["window_clamped"] += 1
            continue

        # history: 관측 종료 시점까지 완료된 action (누출 방지 — stop <= obs_end)
        observed = [action_str(a) for a in (r.get("history") or []) if a["stop_sec"] <= obs_end + 1e-3]
        future_labels = [action_str(a) for a in future[:horizon]]

        # 계약 assert — 미래 action 이 관측창을 침범하면 즉시 실패시킨다
        assert obs_end < target_start, f"{r['sample_id']}: obs_end >= target_start"
        assert all(a["start_sec"] >= target_start - 1e-3 for a in future[:horizon]), \
            f"{r['sample_id']}: future action starts before target_start"

        samples.append({
            "sample_id": r["sample_id"],
            "video_uid": r["video_uid"],
            "goal_text": goals.get(r["video_uid"], ""),
            "scenario": r.get("scenario", ""),
            "observed_actions": observed,
            "future_actions": future_labels,
            "horizon": horizon,
            "obs_start_sec": round(obs_start, 5),
            "obs_end_sec": round(obs_end, 5),
            "target_start_sec": round(target_start, 5),
            "gap_sec": round(gap, 3),
            "wm_candidates": [C.normalize_label(c) for c in r["candidates"]],
            "wm_scores": r["wm_scores"],
            "gt_next_action": gt,
        })
    stats = {
        "n_samples": len(samples),
        "n_videos": len({s["video_uid"] for s in samples}),
        "drops": dict(drops),
        "median_gap_sec": round(sorted(s["gap_sec"] for s in samples)[len(samples) // 2], 3) if samples else None,
    }
    return samples, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context", default="runs/cesft_v2/data/context_val.jsonl")
    p.add_argument("--goalstep-val", default="data/Ego4D/v2/annotations/goalstep_val.json")
    p.add_argument("--out-dir", default="runs/vpa_v2")
    p.add_argument("--gap-cap", type=float, default=C.GAP_CAP_SEC)
    p.add_argument("--split", choices=["heldout", "dev", "both"], default="heldout",
                   help="평가 split. 기본 heldout — dev 는 step2 학습 중 probe 대상이라 제외")
    p.add_argument("--horizons", type=int, nargs="+", default=list(C.DEFAULT_HORIZONS))
    p.add_argument("--frontier-subset", type=int, default=500,
                   help="frontier 비용 상한용 고정 부분집합 크기 (seed 고정, 재개 안전)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = C.read_jsonl(args.context)
    goals = load_goals(Path(args.goalstep_val))
    vocab = C.action_vocab()
    out_dir = Path(args.out_dir)
    C.dump_json(out_dir / "vocab.json", {"labels": vocab, "size": len(vocab), "lineage": "goalstep_step_labels.csv"})

    manifest = {
        "contract": {
            "obs_window_sec": C.OBS_WINDOW_SEC, "safety_gap_sec": C.SAFETY_GAP_SEC,
            "n_frames": C.N_FRAMES, "fps": C.N_FRAMES / C.OBS_WINDOW_SEC,
            "frame_short_side": C.FRAME_SHORT_SIDE, "gap_cap_sec": args.gap_cap,
            "note": "obs ends 1s BEFORE target action starts - no future leakage",
        },
        "source": {"context": args.context, "n_rows": len(rows), "split": args.split},
        "contamination": {
            "vpa_videos_subset_of_goalstep_val": True,
            "overlap_with_goalstep_train_videos": 0,
            "step2_sft_trained_on": "context_train.jsonl (goalstep train split)",
            "dev_split_excluded": args.split == "heldout",
            "why": "dev 는 step2 학습 루프의 probe_gen 대상이라 평가에서 제외한다",
        },
        "vocab_size": len(vocab), "horizons": {},
    }

    for T in args.horizons:
        samples, stats = build(rows, goals, T, args.gap_cap, args.split)
        C.dump_json(out_dir / f"vpa_v2_T{T}.json", samples)

        # frontier 부분집합: seed 고정 셔플의 앞 N개 — 재개·재현 시 항상 동일 집합
        rng = random.Random(args.seed)
        ids = sorted(s["sample_id"] for s in samples)
        rng.shuffle(ids)
        subset = sorted(ids[: args.frontier_subset])
        C.dump_json(out_dir / f"frontier_subset_T{T}.json",
                    {"sample_ids": subset, "n": len(subset), "seed": args.seed,
                     "n_videos": len({s["video_uid"] for s in samples if s["sample_id"] in set(subset)})})
        stats["frontier_subset_n"] = len(subset)
        manifest["horizons"][f"T{T}"] = stats
        print(f"[T={T}] {stats['n_samples']} samples / {stats['n_videos']} videos "
              f"| drops={stats['drops']} | frontier subset={len(subset)}")

    C.dump_json(out_dir / "manifest.json", manifest)
    print(f"\nwrote {out_dir}/vpa_v2_T*.json, vocab.json ({len(vocab)} labels), manifest.json")


if __name__ == "__main__":
    main()
