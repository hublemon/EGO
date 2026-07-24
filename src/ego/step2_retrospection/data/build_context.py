"""support jsonl에 history H<t / future F_t를 채운다 (Phase-1 첫 단계, CPU).

원천: src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_step_labels.csv
(step+substep 평탄화, raw taxonomy id) — 원본 goalstep json 재파싱 없이 인덱스와
동일한 계보의 action 시퀀스를 쓴다.

규칙:
  history: end_time <= decision_time(=obs_end) 인 완료 action 전부 (시간순)
  future:  start_time >= target_start - 1ms 인 action을 시간순으로 최대
           FUTURE_MAX_ACTIONS(5)개, start < target_start + FUTURE_MAX_SECONDS(24s)
  중복: (start,end,verb,noun) 완전 중복 제거 (step/substep 동일 구간)

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.data.build_context [--split val] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from ego.step2_retrospection.contracts import (
    FUTURE_MAX_ACTIONS, FUTURE_MAX_SECONDS, HistoryAction, SupportSample, assert_strict_contract,
)
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, read_jsonl, runs_root, write_marker

REPO = Path(__file__).resolve().parents[4]
LABELS_CSV = REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_step_labels.csv"
TAXONOMY = REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_verbnoun_taxonomy.json"
EPS = 1e-3


def load_timelines() -> dict[str, list[HistoryAction]]:
    tax = json.loads(TAXONOMY.read_text())
    verbs, nouns = tax["verbs"], tax["nouns"]
    per_video: dict[str, set] = defaultdict(set)
    with open(LABELS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_video[row["video_uid"]].add((
                float(row["start_time"]), float(row["end_time"]),
                verbs[int(row["verb_label"])], nouns[int(row["noun_label"])],
            ))
    out = {}
    for vid, rows in per_video.items():
        out[vid] = [HistoryAction(verb=v, noun=n, start_sec=s, stop_sec=e)
                    for s, e, v, n in sorted(rows)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tau_exact", type=int, default=1,
                    help="1=gap≡tau_a 강제(start−1s 계약) · 0=최소 gap만(end−1s 가변 horizon, retro4)")
    args = ap.parse_args()

    timelines = load_timelines()
    data_dir = runs_root() / "data"
    splits = ["train", "val"] if args.split == "both" else [args.split]
    stats_all = {}

    for split in splits:
        rows = read_jsonl(data_dir / f"support_{split}.jsonl")
        if args.limit:
            rows = rows[: args.limit]
        out_path = data_dir / f"context_{split}.jsonl"
        out_path.unlink(missing_ok=True)
        sw = StatusWriter(f"S1_context_{split}", total=len(rows))
        n_gt_first = n_future0 = 0
        hist_lens, fut_lens = [], []

        for i, rec in enumerate(rows):
            tl = timelines.get(rec["video_uid"], [])
            t = rec["obs_end_sec"]
            ts = rec["target_start_sec"]
            history = [a for a in tl if a.stop_sec <= t + EPS]
            future = [a for a in tl if a.start_sec >= ts - EPS
                      and a.start_sec < ts + FUTURE_MAX_SECONDS][:FUTURE_MAX_ACTIONS]
            # 계약 재검증 (EPS 경계는 시각 클램프로 흡수)
            history = [HistoryAction(a.verb, a.noun, a.start_sec, min(a.stop_sec, t)) for a in history]
            future = [HistoryAction(a.verb, a.noun, max(a.start_sec, ts), a.stop_sec) for a in future]

            s = SupportSample(
                sample_id=rec["sample_id"], video_uid=rec["video_uid"], clip_uid=rec["clip_uid"],
                obs_start_sec=rec["obs_start_sec"], obs_end_sec=rec["obs_end_sec"],
                target_start_sec=rec["target_start_sec"], gt_verb=rec["gt_verb"], gt_noun=rec["gt_noun"],
                candidates=rec["candidates"], wm_scores=rec["wm_scores"],
                history=history, future=future, boundary_flag=rec.get("boundary_flag", False))
            assert_strict_contract(s, tau_exact=bool(args.tau_exact))

            if not future:
                n_future0 += 1
            elif future[0].verb == s.gt_verb and future[0].noun == s.gt_noun:
                n_gt_first += 1
            hist_lens.append(len(history))
            fut_lens.append(len(future))

            out = asdict(s)
            out.update({k: rec[k] for k in ("gt_rank", "scenario", "split")})
            append_jsonl(out_path, out)
            sw.update(done=i + 1)

        n = len(rows)
        stats = {
            "n": n, "future_empty": n_future0,
            "gt_is_future0": round(n_gt_first / max(1, n - n_future0), 4),
            "hist_len_mean": round(sum(hist_lens) / max(1, n), 2),
            "fut_len_mean": round(sum(fut_lens) / max(1, n), 2),
        }
        stats_all[split] = stats
        sw.finish(metrics=stats, message=f"{split} done")
        print(f"[S1] {split}: {stats}")

    (data_dir / "context_stats.json").write_text(json.dumps(stats_all, indent=1))
    write_marker("S1_CONTEXT_DONE", stats_all)


if __name__ == "__main__":
    main()
