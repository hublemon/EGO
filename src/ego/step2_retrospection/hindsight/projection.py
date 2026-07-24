"""Hindsight 파이프라인 러너: Ψ(teacher) → Φ(projection) → 규칙 게이트 → chosen trace.

대상: train 서브셋 중 gt_in_support=True 샘플만 (Handoff 1 §8 — support failure 분리).
출력: runs/retro3/data/chosen_train.jsonl (y+ = r_proj/b_proj/a_GT, gate 결과 포함)
정책: drop-not-patch — 게이트 탈락 시 재작성하지 않고 사유와 함께 기록만.

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.hindsight.projection \
      --subset runs/retro3/data/train_subset.json [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.hindsight import quality_gate as qg
from ego.step2_retrospection.hindsight.teacher import parse_h, teacher_messages
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, read_jsonl, runs_root, write_marker

PROJ_SYSTEM = (
    "You write a decision-time rationale for an egocentric video. You see frames from the last "
    "8 seconds BEFORE a decision point, the person's completed actions, and a shuffled candidate "
    "list of next actions. You are also given (a) a hindsight analysis of the procedure inferred "
    "from what happened later, and (b) the action the person actually did next.\n"
    "Write what a careful observer could have concluded AT the decision point, using ONLY "
    "evidence visible in the frames and the completed-action history.\n"
    "STRICT RULES:\n"
    "1. Never mention or imply events after the decision point (no future actions, objects or "
    "outcomes that are not visible).\n"
    "2. Never say the next action is already happening or done.\n"
    "3. If the visible evidence is ambiguous, keep the belief at the level of the local "
    "procedural stage - do not overclaim the exact goal.\n"
    "4. <task_belief> must NOT contain the next action's verb-noun label or a direct paraphrase "
    "of it.\n"
    "5. <reasoning> should compare candidates against the evidence and conclude toward the "
    "actual next action naturally.\n"
    "Respond in EXACTLY this format:\n"
    "<reasoning>\n3-6 sentences.\n</reasoning>\n"
    "<task_belief>\nOne sentence about the current local procedure/subgoal.\n</task_belief>"
)


def proj_messages(rec: dict, h: dict, images) -> list[dict]:
    gt = f"{rec['gt_verb']} {rec['gt_noun']}"
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": (
        f"Completed actions so far (oldest to newest):\n{vlm.fmt_history(rec)}\n\n"
        f"Candidate next actions (shuffled):\n{vlm.fmt_candidates(rec['candidates'])}\n\n"
        f"Hindsight analysis (from later events - do NOT cite it as evidence):\n"
        f"{json.dumps(h, ensure_ascii=False)}\n\n"
        f"Actual next action (for your grounding only, rule 4 applies): {gt}\n\n"
        "Write the decision-time <reasoning> and <task_belief> now."
    )})
    return [{"role": "system", "content": [{"type": "text", "text": PROJ_SYSTEM}]},
            {"role": "user", "content": content}]


def parse_proj(text: str) -> dict | None:
    r = vlm.TAG_RE["reasoning"].search(text)
    b = vlm.TAG_RE["task_belief"].search(text)
    if not (r and b):
        return None
    return {"reasoning": r.group(1).strip(), "task_belief": b.group(1).strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_start_m1_lobs8.yaml")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    data_dir = runs_root() / "data"

    rows = read_jsonl(data_dir / "context_train.jsonl")
    if args.subset:
        keep = set(json.loads(Path(args.subset).read_text())["sample_ids"])
        rows = [r for r in rows if r["sample_id"] in keep]
    rows = [r for r in rows if f"{r['gt_verb']} {r['gt_noun']}" in r["candidates"]]  # GT∈support
    if args.limit:
        rows = rows[: args.limit]

    out_path = data_dir / "chosen_train.jsonl"
    done_ids = {r["sample_id"] for r in read_jsonl(out_path)}
    todo = [r for r in rows if r["sample_id"] not in done_ids]
    todo.sort(key=lambda r: (r["video_uid"], r["obs_start_sec"]))  # 리더 캐시 히트↑ (id 기반 resume이라 순서 무관)

    model, processor = vlm.load_model()
    sw = StatusWriter("S3_hindsight", total=len(rows))
    sw.update(done=len(done_ids), force=True)
    n_pass = n_drop = 0
    gate_fail: dict[str, int] = {}

    # prefetch_chunks: 다음 chunk 프레임 추출(CPU)이 Ψ/Φ 생성(GPU)과 겹침
    for chunk, frames in vlm.prefetch_chunks(video_root, todo, args.batch_size):
        t0 = time.time()
        # Ψ — 텍스트 전용 배치
        h_texts = vlm.generate_batch(model, processor, [teacher_messages(r) for r in chunk],
                                     max_new_tokens=192)
        hs = [parse_h(t) for t in h_texts]
        # Φ — 비전 배치 (teacher 파싱 실패는 드랍)
        msgs, ok = [], []
        for rec, h, (imgs, err) in zip(chunk, hs, frames):
            if h is None:
                append_jsonl(out_path, {"sample_id": rec["sample_id"], "gate": "drop",
                                        "reasons": ["teacher_parse"], "split": rec["split"]})
                n_drop += 1
                gate_fail["teacher_parse"] = gate_fail.get("teacher_parse", 0) + 1
                continue
            if err is not None:
                append_jsonl(out_path, {"sample_id": rec["sample_id"], "gate": "drop",
                                        "reasons": [f"video:{str(err)[:120]}"], "split": rec["split"]})
                n_drop += 1
                gate_fail["video"] = gate_fail.get("video", 0) + 1
                continue
            msgs.append(proj_messages(rec, h, imgs))
            ok.append((rec, h))
        p_texts = vlm.generate_batch(model, processor, msgs, max_new_tokens=320) if msgs else []

        for (rec, h), text in zip(ok, p_texts):
            parsed = parse_proj(text)
            gt = f"{rec['gt_verb']} {rec['gt_noun']}"
            if parsed is None:
                reasons = ["proj_parse"]
            else:
                reasons = qg.check_chosen(parsed["reasoning"], parsed["task_belief"], rec)
            if reasons:
                for x in reasons:
                    gate_fail[x] = gate_fail.get(x, 0) + 1
                append_jsonl(out_path, {"sample_id": rec["sample_id"], "gate": "drop",
                                        "reasons": reasons, "split": rec["split"]})
                n_drop += 1
            else:
                append_jsonl(out_path, {
                    "sample_id": rec["sample_id"], "split": rec["split"], "gate": "pass",
                    "reasoning": parsed["reasoning"], "task_belief": parsed["task_belief"],
                    "action": gt, "gt": gt, "h_t": h,
                    "trace": qg.serialize_trace(parsed["reasoning"], parsed["task_belief"], gt)})
                n_pass += 1
        sw.update(done=len(done_ids) + n_pass + n_drop, metrics={
            "pass_rate": round(n_pass / max(1, n_pass + n_drop), 4),
            "gate_fail": gate_fail,
            "sec_per_sample": round((time.time() - t0) / max(1, len(chunk)), 2)})

    sw.finish(metrics={"pass": n_pass, "drop": n_drop, "gate_fail": gate_fail,
                       "pass_rate": round(n_pass / max(1, n_pass + n_drop), 4)})
    write_marker("S3_HINDSIGHT_DONE", {"pass": n_pass, "drop": n_drop})
    print(f"[S3] pass={n_pass} drop={n_drop} fail={gate_fail}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
