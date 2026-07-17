"""merge_b0_samples.py — faa_traces + b0meta 를 sample_id 로 병합 (build_dpo_dataset 입력).

faa_traces_{split}.jsonl (generate_faa_traces):
  {sample_id, prompt, image_path, memory_context, candidates, faa_traces[...], faa_checkpoint_hash}
grpo_{split}_b0meta.jsonl (F0 convert 산출):
  {sample_id, gt_action_t{verb,noun,...}, future_gt_actions[...], trigger_frame, ...}

출력 b0_samples_{split}.jsonl — build_dpo_dataset.build_pairs 가 기대하는 스키마:
  {sample_id, prompt, image_path, memory_context, candidates, gt_action{verb,noun},
   future_gt_actions[...], faa_traces[str...], trigger_time, policy_history[...], faa_checkpoint_hash}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _by_id(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["sample_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faa_traces", required=True)
    ap.add_argument("--b0meta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    faa = _by_id(Path(args.faa_traces))
    meta = _by_id(Path(args.b0meta))

    rows, n_nometa = [], 0
    for sid, f in faa.items():
        m = meta.get(sid)
        if not m:
            n_nometa += 1
            continue
        gt = m.get("gt_action_t") or {}
        rows.append({
            "sample_id": sid,
            "prompt": f["prompt"],
            "image_path": f.get("image_path", ""),
            "memory_context": f.get("memory_context", ""),
            "candidates": f.get("candidates", []),
            "gt_action": {"verb": gt.get("verb"), "noun": gt.get("noun")},
            "future_gt_actions": m.get("future_gt_actions", []),
            "faa_traces": f.get("faa_traces", []),
            "trigger_time": m.get("trigger_timestamp"),
            "policy_history": [],   # 프롬프트에 이미 반영됨 (수치 history 는 leakage 검사용 옵션)
            "faa_checkpoint_hash": f.get("faa_checkpoint_hash", ""),
        })
    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[done] merged {len(rows)} samples → {args.out} (no-meta dropped: {n_nometa})")


if __name__ == "__main__":
    main()
