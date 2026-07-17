"""
⑤ assemble_train.py — 최종 grpo_dataset.jsonl 조립.

입력:
  - selected_train.jsonl   (sample meta + gt_label + task_goal)
  - predictions_train.jsonl (V-JEPA2 top5 verb/noun/action + likelihood)
  - frames/{sample_id}.jpg  (trigger frame)
  - memory_train.jsonl      (task_history + temporal_proximity)
출력:
  - data/grpo_dataset/grpo_dataset.jsonl  (GRPO_DATASET_SPEC.md 포맷)

조인 키: sample_id. prediction 이 없는 샘플(②에서 에러난 건)은 드롭.
gt_in_top5_verb/noun/action 플래그 계산 포함.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
GRPO_DIR = EGO_ROOT / "data/grpo_dataset"
SELECTED = GRPO_DIR / "selected_train.jsonl"
PRED = GRPO_DIR / "predictions_train.jsonl"
MEMORY = GRPO_DIR / "memory_train.jsonl"
FRAMES_ROOT = GRPO_DIR / "frames"
OUT = GRPO_DIR / "grpo_dataset.jsonl"


def load_jsonl_by_id(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["sample_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation"], default="train",
                    help="validation: *_heldout.jsonl 입출력 세트 사용")
    ap.add_argument("--selected", type=str, default=None)
    ap.add_argument("--pred", type=str, default=None)
    ap.add_argument("--memory", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--frames_manifest", type=str, default=None,
                    help="frames_manifest.jsonl — n_frames/offsets_sec (4f 시 필수)")
    args = ap.parse_args()

    sfx = "heldout" if args.split == "validation" else "train"
    selected_path = Path(args.selected) if args.selected else GRPO_DIR / f"selected_{sfx}.jsonl"
    pred_path = Path(args.pred) if args.pred else GRPO_DIR / f"predictions_{sfx}.jsonl"
    memory_path = Path(args.memory) if args.memory else GRPO_DIR / f"memory_{sfx}.jsonl"
    out_path = Path(args.out) if args.out else (
        GRPO_DIR / ("grpo_dataset_heldout.jsonl" if args.split == "validation" else "grpo_dataset.jsonl"))

    selected = load_jsonl_by_id(selected_path)
    preds = load_jsonl_by_id(pred_path)
    memory = load_jsonl_by_id(memory_path)
    fmanifest_path = Path(args.frames_manifest) if args.frames_manifest else GRPO_DIR / "frames_manifest.jsonl"
    fmeta = load_jsonl_by_id(fmanifest_path) if fmanifest_path.exists() else {}

    print(f"[load] selected={len(selected)} predictions={len(preds)} memory={len(memory)} "
          f"frame_meta={len(fmeta)}")

    rows = []
    n_drop_nopred = 0
    n_drop_noframe = 0
    for sid, s in selected.items():
        if sid not in preds:
            n_drop_nopred += 1
            continue
        frame_path = FRAMES_ROOT / f"{sid}.jpg"
        if not frame_path.exists():
            n_drop_noframe += 1
            continue

        p = preds[sid]
        gt = s["gt_label"]
        gt_vc = gt["verb_class"]
        gt_nc = gt["noun_class"]

        gt_in_verb = any(x["verb_class"] == gt_vc for x in p["top5_verb"])
        gt_in_noun = any(x["noun_class"] == gt_nc for x in p["top5_noun"])
        gt_in_action = any(
            x["verb_class"] == gt_vc and x["noun_class"] == gt_nc
            for x in p["top5_action"]
        )

        mem = memory.get(sid, {})
        rec = {
            "sample_id": sid,
            "split": args.split,
            "video_id": s["video_id"],
            "narration_id": s["narration_id"],
            "trigger_frame": s["trigger_frame"],
            "trigger_timestamp": s["trigger_timestamp"],
            "frame_path": str(frame_path.relative_to(EGO_ROOT)),
            "task_goal": s["task_goal"],
            "gt_label": {
                "action": gt["action"],
                "verb": gt["verb"],
                "noun": gt["noun"],
                "verb_class": gt_vc,
                "noun_class": gt_nc,
            },
            "wm_output": {
                "top5_verb": p["top5_verb"],
                "top5_noun": p["top5_noun"],
                "top5_action": p["top5_action"],
                "gt_in_top5_verb": gt_in_verb,
                "gt_in_top5_noun": gt_in_noun,
                "gt_in_top5_action": gt_in_action,
            },
            "memory_context": {
                "task_history": mem.get("task_history", []),
                "temporal_proximity": mem.get("temporal_proximity", {}),
                "frame_aligned_context": mem.get("frame_aligned_context", {}),   # L2-c
                "cutoff_rule": mem.get("cutoff_rule", "legacy"),
            },
            # 4f 프롬프트 라벨용 (train 프롬프트 빌더가 읽음)
            "frame_meta": {
                "n_frames": (fmeta.get(sid) or {}).get("n_frames", 1),
                "offsets_sec": (fmeta.get(sid) or {}).get("offsets_sec", [0.0]),
            },
            # ⚠ policy prompt 노출 금지 — convert 가 b0_meta 파일로 분리
            "future_gt_actions": mem.get("future_gt_actions", []),
        }
        rows.append(rec)

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] assembled {len(rows)} samples → {out_path}")
    print(f"  dropped (no prediction): {n_drop_nopred}")
    print(f"  dropped (no frame):      {n_drop_noframe}")


if __name__ == "__main__":
    main()
