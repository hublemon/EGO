"""학습 중 trace 진화 관측 — 고정 probe 세트를 주기적으로 현재 정책으로 생성.

probe 세트: dev split 8샘플 고정 (G1 2 + G2 4 + 기타 2, seed 42) —
모든 run이 같은 세트를 쓰므로 step 간·arm 간 직접 비교 가능.
출력: runs/retro3/probe/{run_name}.jsonl — 대시보드 '트레이스 진화' 섹션의 원천.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from ego.step2_retrospection import vlm
from ego.step2_retrospection.runtime import append_jsonl, read_jsonl, runs_root

PROBE_N = {"G1": 2, "G2": 4, "other": 2}


def build_probe_set(seed: int = 42) -> list[dict]:
    """probe_set.json 생성(1회) 또는 로드 — 모든 run 공용."""
    path = runs_root() / "probe" / "probe_set.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return json.loads(path.read_text())["recs"]
    rows = [r for r in read_jsonl(runs_root() / "data" / "context_val.jsonl") if r["split"] == "dev"]
    rng = random.Random(seed)
    rng.shuffle(rows)
    buckets = {"G1": [], "G2": [], "other": []}
    for r in rows:
        gt = f"{r['gt_verb']} {r['gt_noun']}"
        wm_top1 = r["candidates"][max(range(len(r["wm_scores"])), key=lambda j: r["wm_scores"][j])]
        b = "G1" if wm_top1 == gt else ("G2" if gt in r["candidates"] else "other")
        if len(buckets[b]) < PROBE_N[b]:
            r = dict(r)
            r["_bucket"] = b
            buckets[b].append(r)
        if all(len(buckets[k]) >= PROBE_N[k] for k in PROBE_N):
            break
    recs = buckets["G1"] + buckets["G2"] + buckets["other"]
    path.write_text(json.dumps({"seed": seed, "recs": recs}, ensure_ascii=False))
    return recs


@torch.no_grad()
def run_probe(model, processor, video_root: Path, run_name: str, step: int,
              probe_recs: list[dict]) -> dict:
    """현재 정책으로 probe 8샘플 생성 → jsonl append. 반환: 요약(acc 등)."""
    was_training = model.training
    model.eval()
    try:
        msgs = []
        for rec in probe_recs:
            imgs = vlm.extract_frames(video_root, rec["video_uid"],
                                      rec["obs_start_sec"], rec["obs_end_sec"])
            msgs.append(vlm.build_messages(rec, imgs))
        texts = vlm.generate_batch(model, processor, msgs, max_new_tokens=320)
    finally:
        if was_training:
            model.train()

    out_path = runs_root() / "probe" / f"{run_name}.jsonl"
    n_correct = 0
    samples = []
    for rec, text in zip(probe_recs, texts):
        parsed = vlm.parse_trace(text)
        matched = vlm.match_candidate(parsed["action"], rec["candidates"]) if parsed else None
        gt = f"{rec['gt_verb']} {rec['gt_noun']}"
        n_correct += int(matched == gt)
        samples.append({
            "sample_id": rec["sample_id"], "bucket": rec.get("_bucket", "?"), "gt": gt,
            "action": matched, "correct": matched == gt,
            "task_belief": (parsed["task_belief"] if parsed else None),
            "reasoning_head": (parsed["reasoning"][:260] if parsed else None),
            "malformed": parsed is None or matched is None,
        })
    entry = {"run": run_name, "step": step, "ts": time.time(),
             "probe_acc": round(n_correct / len(probe_recs), 3), "samples": samples}
    append_jsonl(out_path, entry)
    return {"probe_acc": entry["probe_acc"]}
