"""에피소드 빌더 — 한 영상(=한 goal)의 **연속된 결정지점 사슬**을 만든다.

원천은 step2 검증셋 `runs/cesft_v2/data/context_val.jsonl` (결정지점마다 WM top-10 후보).
여기서 closed loop 를 돌릴 수 있는 사슬만 남긴다:

  1. `split == heldout`      — dev 는 step2 학습 루프가 probe 로 들여다본 표본
  2. 로컬 mp4 존재            — 프레임을 뽑을 수 있어야 한다
  3. **level == step**        — GoalStep 은 step 안에 substep 이 중첩돼 두 레벨을 섞으면
                                구간이 겹쳐 "선형 plan" 이 성립하지 않는다. VPA 도 step 레벨.
  4. gap ≤ WM_ALIGN_GAP_SEC  — WM 후보가 산출된 원본 관측창이 target 직전이어야 한다.
                                (gap 이 크면 후보가 수십 초 전 관측에서 나온 낡은 것)
  5. target_start ≥ 5.5s      — 4초 관측창이 영상 시작에 잘리지 않게
  6. 에피소드 길이 ≥ MIN_STEPS — 사슬이 너무 짧으면 "동적 planning" 을 볼 수 없다

사용:
  PYTHONPATH=src python -m ego.step3_results.dynamic.build_episodes --out-dir runs/dynamic_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ego.step3_results.dynamic import common as C

REPO = Path(__file__).resolve().parents[4]
TAXONOMY = REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_verbnoun_taxonomy.json"
LABELS_CSV = REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_step_labels.csv"


def load_levels() -> dict[tuple, set]:
    """(video_uid, start_time, verb, noun) → {'step','substep'} — 타깃의 계층 판별용."""
    tax = json.loads(TAXONOMY.read_text())
    verbs, nouns = tax["verbs"], tax["nouns"]
    out: dict[tuple, set] = defaultdict(set)
    with open(LABELS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["video_uid"], round(float(row["start_time"]), 2),
                   verbs[int(row["verb_label"])], nouns[int(row["noun_label"])])
            out[key].add(row["level"])
    return out


def load_goals(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    out = {}
    for v in data.get("videos", []):
        desc = (v.get("goal_description") or "").strip()
        cat = (v.get("goal_category") or "").strip()
        if not desc:
            desc = cat.split(":", 1)[-1].replace("_", " ").strip().lower()
        summary = v.get("summary") or ""
        if isinstance(summary, list):  # 일부 영상은 문장 리스트로 들어있다
            summary = " ".join(str(x) for x in summary)
        out[v["video_uid"]] = {"goal_text": desc, "goal_category": cat, "summary": summary.strip()}
    return out


def build(rows: list[dict], levels, goals, video_root: Path, *, level: str, min_steps: int,
          max_steps: int, gap_cap: float) -> tuple[list[dict], dict]:
    local = {f.stem for f in video_root.glob("*.mp4")}
    drops = Counter()
    per_video: dict[str, list[dict]] = defaultdict(list)

    for r in rows:
        if r.get("split") != "heldout":
            drops["not_heldout"] += 1
            continue
        if r["video_uid"] not in local:
            drops["no_local_video"] += 1
            continue
        if r["target_start_sec"] - r["obs_end_sec"] > gap_cap:
            drops["wm_window_stale"] += 1
            continue
        if r["target_start_sec"] < C.MIN_TARGET_START_SEC:
            drops["too_early_in_video"] += 1
            continue
        lv = levels.get((r["video_uid"], round(r["target_start_sec"], 2), r["gt_verb"], r["gt_noun"]), set())
        if level != "any" and level not in lv:
            drops[f"not_{level}_level"] += 1
            continue
        per_video[r["video_uid"]].append(r)

    episodes = []
    for vid, rs in sorted(per_video.items()):
        rs.sort(key=lambda r: (r["target_start_sec"], int(r["sample_id"].rsplit("_", 1)[-1])))
        # 같은 시각에 중복 결정지점(중첩 주석)이 있으면 첫 것만 — 사슬은 시각당 하나
        seen: set = set()
        chain = []
        for r in rs:
            k = round(r["target_start_sec"], 2)
            if k in seen:
                drops["duplicate_target_time"] += 1
                continue
            seen.add(k)
            chain.append(r)
        if len(chain) < min_steps:
            drops["episode_too_short"] += len(chain)
            continue
        if len(chain) > max_steps:
            drops["episode_truncated_tail"] += len(chain) - max_steps
            chain = chain[:max_steps]

        steps = []
        for i, r in enumerate(chain):
            obs_start, obs_end = C.observation_window(float(r["target_start_sec"]))
            gt = f"{r['gt_verb']} {r['gt_noun']}"
            assert obs_end < r["target_start_sec"], f"{r['sample_id']}: 관측창이 타깃을 침범"
            steps.append({
                "sample_id": r["sample_id"], "step_idx": i,
                "target_start_sec": round(float(r["target_start_sec"]), 5),
                "obs_start_sec": round(obs_start, 5), "obs_end_sec": round(obs_end, 5),
                "orig_gap_sec": round(r["target_start_sec"] - r["obs_end_sec"], 3),
                "gt_action": gt,
                "candidates": list(r["candidates"]), "wm_scores": list(r["wm_scores"]),
                "gt_in_candidates": gt in r["candidates"], "gt_rank": r.get("gt_rank"),
            })
        g = goals.get(vid, {})
        episodes.append({
            "video_uid": vid, "goal_text": g.get("goal_text", ""),
            "goal_category": g.get("goal_category", ""), "summary": g.get("summary", ""),
            "scenario": chain[0].get("scenario", ""), "n_steps": len(steps),
            "span_sec": round(steps[-1]["target_start_sec"] - steps[0]["target_start_sec"], 1),
            "coverage": round(sum(s["gt_in_candidates"] for s in steps) / len(steps), 4),
            "steps": steps,
        })

    episodes.sort(key=lambda e: -e["n_steps"])
    n_steps = sum(e["n_steps"] for e in episodes)
    stats = {
        "n_episodes": len(episodes), "n_steps": n_steps,
        "n_covered_steps": sum(s["gt_in_candidates"] for e in episodes for s in e["steps"]),
        "coverage_rate": round(sum(s["gt_in_candidates"] for e in episodes for s in e["steps"])
                               / max(1, n_steps), 4),
        "steps_per_episode": {"min": min((e["n_steps"] for e in episodes), default=0),
                              "max": max((e["n_steps"] for e in episodes), default=0),
                              "mean": round(n_steps / max(1, len(episodes)), 1)},
        "n_goal_categories": len({e["goal_category"] for e in episodes}),
        "drops": dict(drops),
    }
    return episodes, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context", default="runs/cesft_v2/data/context_val.jsonl")
    p.add_argument("--goalstep-val", default="data/Ego4D/v2/annotations/goalstep_val.json")
    p.add_argument("--video-root", default="data/Ego4D/v2/goalstep_videos")
    p.add_argument("--out-dir", default="runs/dynamic_v1")
    p.add_argument("--level", choices=["step", "substep", "any"], default="step")
    p.add_argument("--min-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=40, help="에피소드당 상한 (비용 통제, 앞에서부터)")
    p.add_argument("--gap-cap", type=float, default=C.WM_ALIGN_GAP_SEC)
    p.add_argument("--max-episodes", type=int, default=None, help="스텝 수 많은 순으로 상위 N개만")
    args = p.parse_args()

    rows = C.read_jsonl(args.context)
    levels = load_levels()
    goals = load_goals(Path(args.goalstep_val))
    episodes, stats = build(rows, levels, goals, Path(args.video_root), level=args.level,
                            min_steps=args.min_steps, max_steps=args.max_steps, gap_cap=args.gap_cap)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
        stats["n_episodes"] = len(episodes)
        stats["n_steps"] = sum(e["n_steps"] for e in episodes)

    out = {
        "contract": {
            "obs_window_sec": C.OBS_WINDOW_SEC, "safety_gap_sec": C.SAFETY_GAP_SEC,
            "n_frames": C.N_FRAMES, "fps": C.N_FRAMES / C.OBS_WINDOW_SEC,
            "frame_short_side": C.FRAME_SHORT_SIDE,
            "wm_align_gap_cap_sec": args.gap_cap, "level": args.level,
            "min_steps": args.min_steps, "max_steps": args.max_steps,
            "note": "obs 는 target 시작 1초 전 종료 — 미래 누출 없음. history/belief 는 런타임에 "
                    "모델 자신의 출력으로 채운다 (GT 미사용).",
        },
        "source": {"context": args.context, "n_rows": len(rows), "split": "heldout"},
        "stats": stats,
        "episodes": episodes,
    }
    C.dump_json(Path(args.out_dir) / "episodes.json", out)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out_dir}/episodes.json")
    for e in episodes[:15]:
        print(f"  {e['video_uid'][:12]}  steps={e['n_steps']:3d}  cov={e['coverage']:.2f}  "
              f"{e['goal_category'][:34]:34s} | {e['goal_text'][:40]}")


if __name__ == "__main__":
    main()
