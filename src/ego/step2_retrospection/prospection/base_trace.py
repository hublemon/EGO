"""Prospection: Base Qwen zero-shot trace 생성 (y−). 학습 없음 (Handoff 1 §4 / 2 §4).

입력: runs/retro3/data/context_{split}.jsonl (+ GoalStep 영상 프레임)
출력: runs/retro3/data/base_trace_{split}.jsonl
  {sample_id, reasoning, task_belief, action_raw, action(매칭된 후보 or null),
   malformed, correct, ...}

- greedy decode (base trace는 프롬프트당 1개, 결정적)
- exact candidate matching으로 a∈D_t 보장, 실패는 malformed 태깅 (pair에서
  "너무 쉬운 negative" 처리)
- resume: 기존 출력의 sample_id는 건너뜀 (무인 체인 재시작 안전)

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.prospection.base_trace \
      --split val [--limit N] [--subset train_subset.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, read_jsonl, runs_root, write_marker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_start_m1_lobs8.yaml")
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", default=None, help="sample_id 목록 json (train 서브셋)")
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])

    data_dir = runs_root() / "data"
    rows = read_jsonl(data_dir / f"context_{args.split}.jsonl")
    if args.subset:
        keep = set(json.loads(Path(args.subset).read_text())["sample_ids"])
        rows = [r for r in rows if r["sample_id"] in keep]
    if args.limit:
        rows = rows[: args.limit]

    out_path = data_dir / f"base_trace_{args.split}.jsonl"
    done_ids = {r["sample_id"] for r in read_jsonl(out_path)}
    todo = [r for r in rows if r["sample_id"] not in done_ids]
    todo.sort(key=lambda r: (r["video_uid"], r["obs_start_sec"]))  # 리더 캐시 히트↑ (출력은 id 기반 resume이라 순서 무관)

    model, processor = vlm.load_model()
    sw = StatusWriter(f"S2_base_trace_{args.split}", total=len(rows))
    sw.update(done=len(done_ids), force=True)

    n_ok = n_mal = n_correct = 0
    # prefetch_chunks: 다음 chunk 프레임 추출(CPU)이 현재 chunk 생성(GPU)과 겹침
    for chunk, frames in vlm.prefetch_chunks(video_root, todo, args.batch_size):
        t0 = time.time()
        msgs, ok_recs = [], []
        for rec, (imgs, err) in zip(chunk, frames):
            if err is not None:  # 영상 깨짐 등 — 기록 후 계속 (무인 안전)
                append_jsonl(out_path, {"sample_id": rec["sample_id"], "split": rec["split"],
                                        "error": str(err)[:300], "malformed": True,
                                        "correct": False, "action": None})
                n_mal += 1
            else:
                msgs.append(vlm.build_messages(rec, imgs))
                ok_recs.append(rec)
        texts = vlm.generate_batch(model, processor, msgs, max_new_tokens=args.max_new_tokens) if msgs else []
        per_sample = (time.time() - t0) / max(1, len(chunk))
        for rec, text in zip(ok_recs, texts):
            parsed = vlm.parse_trace(text)
            matched = vlm.match_candidate(parsed["action"], rec["candidates"]) if parsed else None
            gt = f"{rec['gt_verb']} {rec['gt_noun']}"
            out = {
                "sample_id": rec["sample_id"], "split": rec["split"],
                "reasoning": parsed["reasoning"] if parsed else None,
                "task_belief": parsed["task_belief"] if parsed else None,
                "action_raw": parsed["action"] if parsed else None,
                "action": matched, "malformed": parsed is None or matched is None,
                "correct": matched == gt, "gt": gt, "raw_text": None if parsed else text[:2000],
                "gen_sec": round(per_sample, 2),
            }
            n_ok += int(not out["malformed"])
            n_mal += int(out["malformed"])
            n_correct += int(out["correct"])
            append_jsonl(out_path, out)
        sw.update(done=len(done_ids) + n_ok + n_mal, metrics={
            "acc_so_far": round(n_correct / max(1, n_ok + n_mal), 4),
            "malformed_rate": round(n_mal / max(1, n_ok + n_mal), 4),
            "sec_per_sample": round(per_sample, 2)})

    sw.finish(metrics={"n": len(rows), "acc": round(n_correct / max(1, len(todo) or 1), 4),
                       "malformed_rate": round(n_mal / max(1, len(todo) or 1), 4)})
    write_marker(f"S2_BASE_TRACE_{args.split.upper()}_DONE")
    print(f"[S2] {args.split}: new={len(todo)} acc={n_correct / max(1, len(todo)):.4f} mal={n_mal}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
