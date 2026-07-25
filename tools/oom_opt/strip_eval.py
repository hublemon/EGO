#!/usr/bin/env python3
"""θ_CE strip-eval — history 인과를 같은 체크포인트에서 paired 로 측정 (B_nohist 대체, 핸드오프 §2).

별도 no_history arm 을 4h 학습하는 대신, **이미 학습된 θ_CE 어댑터** 하나로
같은 eval 셋을 두 조건에서 추론한다:
  - with-history  : eval_theta_ce 배터리 산출물(theta_ce.records.jsonl) 재사용
  - history-strip : 동일 θ_CE, 프롬프트의 history 만 '(history withheld)' 로 치환 (WM 후보 불변)
per-sample_id paired 차이라 "다른 체크포인트" 교란이 없다 (EGO_jihun --no_memory 패턴).

판정: Δacc = acc(hist) − acc(strip), 전체 + history_length 층화, bootstrap CI.
  H8(긴 history)에서 Δ>0 이면 θ_CE 가 history 를 인과적으로 사용.

**GPU 잡** — orchestrator 와 동시 실행 금지. post_theta_ce_hook.sh 가
CESFT_V2_CHAIN_DONE 이후(=GPU 여유) 호출한다.

출력: runs/cesft_v2/eval/theta_ce_nohist.records.jsonl · strip_verdict.json
마커: S_STRIP_THETA_CE_DONE
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.eval.battery import pick_eval_set
from ego.step2_retrospection.runtime import (StatusWriter, append_jsonl, read_jsonl,
                                             runs_root, write_marker)

ADAPT = "outputs/step2_retrospection/cesft_v2/theta_ce/adapter"


def user_prompt_nohist(rec: dict) -> str:
    """battery user_prompt 와 동일 — history 텍스트만 공란 문구로 치환 (WM 후보 불변)."""
    return (f"Completed actions so far (oldest to newest):\n(history withheld)\n\n"
            f"Candidate next actions (shuffled):\n{vlm.fmt_candidates(rec['candidates'])}\n\n"
            "Which candidate is the next action? Follow the required format.")


def build_messages_nohist(rec: dict, images) -> list[dict]:
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": user_prompt_nohist(rec)})
    return [{"role": "system", "content": [{"type": "text", "text": vlm.SYSTEM_PROMPT}]},
            {"role": "user", "content": content}]


def boot_ci(diffs: list[int], iters=5000, seed=123):
    rng = random.Random(seed)
    n = len(diffs)
    if not n:
        return dict(delta=0.0, ci=[0.0, 0.0], n=0)
    base = sum(diffs) / n
    bs = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)]
    bs.sort()
    return dict(delta=round(100 * base, 2),
                ci=[round(100 * bs[int(0.025 * iters)], 2), round(100 * bs[int(0.975 * iters)], 2)],
                n=n)


def hbin(h):
    if h is None:
        return "?"
    if h == 0:
        return "H0"
    if h <= 3:
        return "H1-3"
    if h <= 7:
        return "H4-7"
    return "H8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/cesft_v2.yaml")
    ap.add_argument("--adapter", default=ADAPT)
    ap.add_argument("--eval_n", type=int, default=int(os.environ.get("EVAL_N", "1000")))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--split_name", default="heldout")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])

    # 배터리와 동일 서브셋 (covered_only 는 overrides.json 따름)
    covered_only = False
    ovp = runs_root() / "overrides.json"
    if ovp.is_file():
        covered_only = bool(json.loads(ovp.read_text()).get("eval_covered_only"))
    ctx = read_jsonl(runs_root() / "data" / "context_val.jsonl")
    rows, pool_cov = pick_eval_set(ctx, args.split_name, args.eval_n, covered_only=covered_only)
    hlen = {r["sample_id"]: len(r.get("history", [])) for r in ctx}
    vid = {r["sample_id"]: r.get("video_uid") for r in ctx}

    rec_path = runs_root() / "eval" / "theta_ce_nohist.records.jsonl"
    done = {r["sample_id"] for r in read_jsonl(rec_path)}
    todo = [r for r in rows if r["sample_id"] not in done]
    todo.sort(key=lambda r: (r["video_uid"], r["obs_start_sec"]))

    from ego.step2_retrospection.eval.battery import load_arm
    model, processor = load_arm(args.adapter)
    sw = StatusWriter("S_STRIP_theta_ce", total=len(rows))
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
                msgs.append(build_messages_nohist(rec, imgs))
                ok.append(rec)
        texts = vlm.generate_batch(model, processor, msgs) if msgs else []
        for rec, text in zip(ok, texts):
            parsed = vlm.parse_trace(text)
            matched = vlm.match_candidate(parsed["action"], rec["candidates"]) if parsed else None
            gt = f"{rec['gt_verb']} {rec['gt_noun']}"
            append_jsonl(rec_path, {
                "sample_id": rec["sample_id"], "action": matched, "gt": gt,
                "malformed": parsed is None or matched is None,
                "correct": matched == gt, "gt_in_support": gt in rec["candidates"],
                "history_length": len(rec.get("history", []))})
        n_seen += len(chunk)
        sw.update(done=len(done) + n_seen,
                  metrics={"sec_per_sample": round((time.time() - t0) / max(1, len(chunk)), 2)})

    # ── paired 분석: hist(theta_ce 배터리) vs strip ──
    sid_set = {r["sample_id"] for r in rows}
    hist = {r["sample_id"]: r for r in read_jsonl(runs_root() / "eval" / "theta_ce.records.jsonl")
            if r["sample_id"] in sid_set}
    strip = {r["sample_id"]: r for r in read_jsonl(rec_path) if r["sample_id"] in sid_set}
    paired = [s for s in hist if s in strip]

    def ok_h(s):
        return int(bool(hist[s].get("correct")) and not hist[s].get("malformed", False))

    def ok_s(s):
        return int(bool(strip[s].get("correct")) and not strip[s].get("malformed", False))

    all_diff = [ok_h(s) - ok_s(s) for s in paired]
    by_bin = defaultdict(list)
    for s in paired:
        by_bin[hbin(hlen.get(s))].append(ok_h(s) - ok_s(s))

    verdict = {
        "n_paired": len(paired),
        "acc_hist": round(sum(ok_h(s) for s in paired) / max(1, len(paired)), 4),
        "acc_strip": round(sum(ok_s(s) for s in paired) / max(1, len(paired)), 4),
        "delta_acc_all": boot_ci(all_diff),
        "delta_acc_by_hlen": {b: boot_ci(by_bin[b]) for b in ("H0", "H1-3", "H4-7", "H8") if by_bin.get(b)},
        "gate_history_causal_H8": bool(by_bin.get("H8") and boot_ci(by_bin["H8"])["ci"][0] > 0),
        "note": "same θ_CE, history-only intervention (WM 후보 불변). B_nohist 학습-arm 대체.",
    }
    (runs_root() / "eval" / "strip_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1))
    sw.finish(metrics={"delta_acc_all": verdict["delta_acc_all"]["delta"],
                       "H8_causal": verdict["gate_history_causal_H8"]})
    write_marker("S_STRIP_THETA_CE_DONE", verdict)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    vlm.close_readers()


if __name__ == "__main__":
    main()
