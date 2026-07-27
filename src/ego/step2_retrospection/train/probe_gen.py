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

# 2026-07-25 (cesft_v2_fp): 8→32 확대 — 스텝별 1인칭율·침식 곡선을 관측 가능한 n으로.
# 비율 유지(G1:G2:other = 1:2:1). run dir 별 probe_set.json이라 기존 run과 충돌 없음.
PROBE_N = {"G1": 8, "G2": 16, "other": 8}


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
                                      rec["obs_start_sec"], rec["obs_end_sec"], rec=rec)
            msgs.append(vlm.build_messages(rec, imgs))
        # 2026-07-26 OOM 수정: probe 는 짧게 디코드하고 학습으로 돌아가 오래 논다.
        # 여기서 리더를 해제하지 않으면 상주 VideoReader 가 학습 내내 ~1 GB/s 로 호스트 RAM 을
        # 먹어 4분 만에 cgroup 한도(240G)에 도달한다. 프레임(PIL)은 이미 복사본이라 안전.
        vlm.close_readers()
        # 학습 중 호출 — optimizer state가 GPU를 점유하므로 8개씩 청크 생성 (32 일괄 금지).
        texts = []
        for i in range(0, len(msgs), 8):
            texts.extend(vlm.generate_batch(model, processor, msgs[i:i + 8],
                                            max_new_tokens=320))
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
            # 2026-07-25: 260자 절단 해제 — 텍스트 지표(1인칭·scene 등) 사후 재계산용 전문 저장.
            "reasoning_head": (parsed["reasoning"] if parsed else None),
            "malformed": parsed is None or matched is None,
        })
    entry = {"run": run_name, "step": step, "ts": time.time(),
             "probe_acc": round(n_correct / len(probe_recs), 3), "samples": samples}
    append_jsonl(out_path, entry)
    return {"probe_acc": entry["probe_acc"]}
