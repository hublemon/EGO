"""자유생성(freegen) 평가 — 후보-비제시 레짐의 화법·경계 내재화 측정 (cesft_v2_fp 설계 §4-3 E4).

battery(후보 제시)와 달리 **후보 리스트를 주지 않고** 다음 행동을 자유 생성시킨다.
파일럿(EGO_jihun `v3_cf_freegen_eval.py`)의 cand-free 레짐과 같은 취지 — 측정 지표:
  in_support   : 생성 action ∈ WM Top-K (경계 내재화 — 후보를 안 보여줘도 경계 안에 드는가)
  gt_correct   : 생성 action == GT (strict, match_candidate 정규화 재사용)
  malformed    : 태그 파싱 실패
  reasoning/task_belief 전문 저장 → tools/trace_text_metrics.py 가 1인칭율 등 사후 계산.

presented 레짐의 텍스트 지표는 battery records(reasoning 전문 기저장)에서 재계산하므로
본 스크립트는 cand_free 레짐 전용이다. 프롬프트는 SYSTEM_PROMPT(1인칭 일원화본)와
구조 동일(태그·3-6문장 규칙 유지) — 후보 관련 문구만 제거. 전 arm 동일 템플릿이므로
arm 간 비교 유효 (crosscohort 판정 준수).

eval 셋: battery 와 같은 pick_eval_set(seed 42) 셔플의 앞 n개 → battery n=1,000 의
부분집합이 되어 per-sample paired 비교 가능.

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.eval.freegen --arm base [--adapter PATH] \
      --config configs/step2_retrospection/cesft_v2.yaml [--eval_n 500]
출력: runs/<run>/eval/freegen_{arm}_cand_free.records.jsonl + .json
마커: S_FREEGEN_{ARM}_CAND_FREE_DONE
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.eval.battery import load_arm, pick_eval_set
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, read_jsonl, runs_root, write_marker

# SYSTEM_PROMPT(1인칭)와 구조 동일 — 후보 문구만 제거. 문장 수·태그 규칙 유지(지표 비교 기반).
FREEGEN_SYSTEM = (
    "You are an embodied agent reasoning about your own ongoing activity from a first-person "
    "view. You see frames from the last 8 seconds of your first-person video and a list of "
    "actions you already COMPLETED. Each action is 'verb noun'. Decide the single next action "
    f"you do next ({vlm.NEXT_GAP_TEXT}).\n"
    "Respond in EXACTLY this format:\n"
    "<reasoning>\nReason from what you see and your completed-action history. "
    "3-6 sentences.\n</reasoning>\n"
    "<task_belief>\nOne sentence: the local procedure or subgoal you are currently in. "
    "Do NOT name the chosen next action verbatim.\n</task_belief>\n"
    "<action>\nverb noun\n</action>"
)


def freegen_messages(rec: dict, images) -> list[dict]:
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": (
        f"Your completed actions so far (oldest to newest):\n{vlm.fmt_history(rec)}\n\n"
        "What is your next action? Follow the required format."
    )})
    return [{"role": "system", "content": [{"type": "text", "text": FREEGEN_SYSTEM}]},
            {"role": "user", "content": content}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/cesft_v2.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split_name", default="heldout", choices=["heldout", "dev"])
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--covered_only", action="store_true", default=True)
    # 2026-07-27: 기본 320 에서는 trace 가 긴 arm 이 </action> 앞에서 잘려 malformed 로 집계된다
    # (sft_r15_c 17.8% vs theta_ce 2.4%, 정상 생성분 83.5 vs 55.1 단어). 조건 간 비교를 하려면
    # 전 조건이 **같은 예산**을 써야 하므로 인자로 노출한다.
    ap.add_argument("--max_new_tokens", type=int, default=320)
    # 2026-07-27 진단 결과 malformed 의 실체는 </action> 누락(vlm.TAG_RE_LENIENT 주석 참조)이다.
    # 켜면 **전 조건을 같은 규칙으로** 다시 돌려야 비교가 유효하다.
    ap.add_argument("--lenient", action="store_true",
                    help="닫는 태그 없이 EOS 로 끝난 마지막 태그를 받아들인다")
    ap.add_argument("--save_text", action="store_true",
                    help="생성 원문을 레코드에 저장 — 이후 파싱 규칙 변경을 재실행 없이 재채점")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    rows, pool_coverage = pick_eval_set(read_jsonl(runs_root() / "data" / "context_val.jsonl"),
                                        args.split_name, args.eval_n,
                                        covered_only=args.covered_only)

    out_dir = runs_root() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"freegen_{args.arm}_cand_free.records.jsonl"
    done = {r["sample_id"] for r in read_jsonl(rec_path)}
    todo = [r for r in rows if r["sample_id"] not in done]
    todo.sort(key=lambda r: (r["video_uid"], r["obs_start_sec"]))

    model, processor = load_arm(args.adapter)
    sw = StatusWriter(f"S_freegen_{args.arm}", total=len(rows))
    sw.update(done=len(done), force=True)

    n_seen = 0
    for chunk, frames in vlm.prefetch_chunks(video_root, todo, args.batch_size):
        t0 = time.time()
        msgs, ok = [], []
        for rec, (imgs, err) in zip(chunk, frames):
            if err is not None:
                append_jsonl(rec_path, {"sample_id": rec["sample_id"], "error": str(err)[:200],
                                        "malformed": True, "action": None})
            else:
                msgs.append(freegen_messages(rec, imgs))
                ok.append(rec)
        texts = (vlm.generate_batch(model, processor, msgs,
                                    max_new_tokens=args.max_new_tokens) if msgs else [])
        for rec, text in zip(ok, texts):
            parsed = vlm.parse_trace(text, lenient=args.lenient)
            strict = vlm.parse_trace(text)          # 관대 규칙이 몇 건을 구했는지 집계용
            gt = f"{rec['gt_verb']} {rec['gt_noun']}"
            action_raw = parsed["action"] if parsed else None
            # in_support: 자유생성 action이 (안 보여준) WM 후보 경계 안에 드는가.
            in_sup = vlm.match_candidate(action_raw, rec["candidates"]) if action_raw else None
            # gt_correct: 정규화 동일성 — match_candidate 를 GT 단일 리스트에 재사용.
            gt_hit = vlm.match_candidate(action_raw, [gt]) if action_raw else None
            append_jsonl(rec_path, {
                "sample_id": rec["sample_id"], "action": action_raw, "gt": gt,
                "malformed": parsed is None,
                "in_support": in_sup is not None,
                "gt_correct": gt_hit is not None,
                "gt_in_support": gt in rec["candidates"],
                "task_belief": parsed["task_belief"] if parsed else None,
                "reasoning": parsed["reasoning"] if parsed else None,
                "malformed_strict": strict is None,
                "recovered": parsed is not None and strict is None,
                **({"text": text} if args.save_text else {})})
        n_seen += len(chunk)
        sw.update(done=len(done) + n_seen,
                  metrics={"sec_per_sample": round((time.time() - t0) / max(1, len(chunk)), 2)})

    recs = [r for r in read_jsonl(rec_path) if r["sample_id"] in {x["sample_id"] for x in rows}]
    n = len(recs)
    summary = {
        "arm": args.arm, "mode": "cand_free", "n": n,
        "in_support": round(sum(r.get("in_support", False) for r in recs) / max(1, n), 4),
        "gt_correct": round(sum(r.get("gt_correct", False) for r in recs) / max(1, n), 4),
        "malformed": round(sum(r.get("malformed", False) for r in recs) / max(1, n), 4),
        "malformed_strict": round(sum(r.get("malformed_strict", r.get("malformed", False))
                                      for r in recs) / max(1, n), 4),
        "recovered": round(sum(r.get("recovered", False) for r in recs) / max(1, n), 4),
        "lenient": args.lenient,
        "max_new_tokens": args.max_new_tokens,
        "coverage_topk": round(sum(r.get("gt_in_support", False) for r in recs) / max(1, n), 4),
        "pool_coverage": round(pool_coverage, 4),
        "adapter": args.adapter or "base",
    }
    (out_dir / f"freegen_{args.arm}_cand_free.json").write_text(json.dumps(summary, indent=1))
    write_marker(f"S_FREEGEN_{args.arm.upper()}_CAND_FREE_DONE", summary)
    sw.finish(metrics=summary)
    print(f"[freegen:{args.arm}] {json.dumps(summary)}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
