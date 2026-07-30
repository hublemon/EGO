#!/usr/bin/env python3
"""perturb_eval — 정책 입력 교란 평가 (image / history 축). strip_eval.py 의 일반화.

`strip_eval.py` 가 "배터리와 동일하되 history 텍스트만 치환"이었던 것을,
**후보 집합은 절대 건드리지 않은 채** image 축까지 확장한다. WM 경로(후보·wm_scores·top1)는
모든 모드에서 불변이므로 개입은 오직 정책 경로에만 걸린다 — 리뷰어 A1/A4 가 요구한 식별이다.

모드:
  noimage         프레임 8장을 blank 로 대체 + system prompt 마스킹. history 유지.
  nohist_noimage  위 + history 를 '(history withheld)' 로 치환 (2x2 의 교호작용 셀).
  othervideo      프레임 유지(원 표본), history 만 **다른 비디오**의 history 로 치환.
                  길이 매칭 + video-disjoint + seed 고정. "입력 제거"가 아니라 "의미만 파괴"라
                  OOD 반론과 의미 사용 주장을 분리한다.
  (nohist 는 strip_eval.py 가 이미 담당한다 — 중복 구현하지 않는다.)

설계 결정:
  · 프레임 수는 항상 8 로 유지한다. 개수를 바꾸면 프롬프트 포맷 shift 가 생겨
    "OOD 취약성" 반론을 자초한다 (defense plan v2 §6.3).
  · noimage 계열은 **decord 에 아예 진입하지 않는다** — blank 이미지를 합성하므로 프레임
    추출이 불필요하다. 07-25 OOM 의 원인이던 리더 상주가 구조적으로 발생하지 않고,
    실행 시간도 크게 준다.
  · othervideo 는 원 표본의 프레임을 그대로 쓰므로 프레임 캐시 히트가 100% 다.
    (도너의 프레임을 쓰는 othervideo_image 는 캐시를 무력화해 10분짜리가 40~60분이 된다.)
  · 출력 스키마는 strip_eval 과 동일 — paired_boot.py / strip_metrics.py 가 그대로 읽는다.

출력: runs/<run>/eval/{arm}_{mode}.records.jsonl · perturb_verdict_{arm}_{mode}.json
마커: S_PERTURB_{ARM}_{MODE}_DONE   (멱등 — 마커가 있으면 즉시 종료)

사용:
  PYTHONPATH=src RETRO3_RUNS=runs/cesft_v2_fp python tools/oom_opt/perturb_eval.py \
      --config configs/step2_retrospection/cesft_v2_fp.yaml \
      --arm theta_ce --adapter outputs/.../theta_ce/adapter --mode noimage --eval_n 1000
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
from ego.step2_retrospection.eval.battery import load_arm, pick_eval_set
from ego.step2_retrospection.runtime import (StatusWriter, append_jsonl, read_jsonl,
                                             runs_root, write_marker)

MODES = ("noimage", "nohist_noimage", "othervideo")
HIST_WITHHELD = "(history withheld)"

# 프레임을 가릴 때만 교체되는 system prompt. 나머지 문장·형식 요구는 원본과 글자 단위로 같다.
SYSTEM_PROMPT_NOIMAGE = vlm.SYSTEM_PROMPT.replace(
    "You see frames from the last 8 seconds of your first-person video, a list of actions "
    "you already COMPLETED,",
    "The video frames are withheld in this trial; you see only a list of actions "
    "you already COMPLETED,")


def _blank_frames(n: int = vlm.N_FRAMES):
    """관측창을 가린 중립 프레임 — 크기·개수는 실제 프레임과 동일 규약(짧은 변 336)."""
    from PIL import Image
    return [Image.new("RGB", (vlm.FRAME_SHORT_SIDE, vlm.FRAME_SHORT_SIDE), (128, 128, 128))
            for _ in range(n)]


def _hist_text(rec: dict, mode: str, donor: dict | None) -> str:
    if mode == "nohist_noimage":
        return HIST_WITHHELD
    if mode == "othervideo":
        return vlm.fmt_history(donor) if donor is not None else HIST_WITHHELD
    return vlm.fmt_history(rec)


def build_messages(rec: dict, images, mode: str, donor: dict | None) -> list[dict]:
    sys_p = SYSTEM_PROMPT_NOIMAGE if mode in ("noimage", "nohist_noimage") else vlm.SYSTEM_PROMPT
    text = (f"Your completed actions so far (oldest to newest):\n{_hist_text(rec, mode, donor)}\n\n"
            f"Candidate next actions (shuffled):\n{vlm.fmt_candidates(rec['candidates'])}\n\n"
            "Which candidate is the next action? Follow the required format.")
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": [{"type": "text", "text": sys_p}]},
            {"role": "user", "content": content}]


def pick_donors(rows: list[dict], seed: int = 42) -> dict:
    """video-disjoint · history 길이 매칭 도너. 결정적(정렬 후 고정 seed)."""
    pool = sorted(rows, key=lambda r: r["sample_id"])
    by_len = defaultdict(list)
    for r in pool:
        by_len[len(r.get("history", []))].append(r)
    lens = sorted(by_len)
    rng = random.Random(seed)
    out = {}
    for r in pool:
        L = len(r.get("history", []))
        # 같은 길이부터, 없으면 길이 차 오름차순으로 다른 비디오 도너를 찾는다
        for cand_len in sorted(lens, key=lambda x: (abs(x - L), x)):
            cands = [d for d in by_len[cand_len] if d.get("video_uid") != r.get("video_uid")]
            if cands:
                out[r["sample_id"]] = cands[rng.randrange(len(cands))]
                break
    return out


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
    ap.add_argument("--config", default="configs/step2_retrospection/cesft_v2_fp.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--eval_n", type=int, default=int(os.environ.get("EVAL_N", "1000")))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--split_name", default="heldout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--covered_only", action="store_true")
    args = ap.parse_args()

    tag = f"{args.arm}_{args.mode}"
    marker = f"S_PERTURB_{args.arm.upper()}_{args.mode.upper()}_DONE"
    if (runs_root() / "markers" / marker).is_file():
        print(f"[perturb] {marker} 존재 — SKIP")
        return

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])

    covered_only = args.covered_only
    ovp = runs_root() / "overrides.json"
    if not covered_only and ovp.is_file():
        covered_only = bool(json.loads(ovp.read_text()).get("eval_covered_only"))
    ctx = read_jsonl(runs_root() / "data" / "context_val.jsonl")
    rows, pool_cov = pick_eval_set(ctx, args.split_name, args.eval_n, covered_only=covered_only)
    hlen = {r["sample_id"]: len(r.get("history", [])) for r in ctx}

    donors = pick_donors(rows, args.seed) if args.mode == "othervideo" else {}
    no_frames = args.mode in ("noimage", "nohist_noimage")

    rec_path = runs_root() / "eval" / f"{tag}.records.jsonl"
    done = {r["sample_id"] for r in read_jsonl(rec_path)}
    todo = [r for r in rows if r["sample_id"] not in done]
    todo.sort(key=lambda r: (r["video_uid"], r["obs_start_sec"]))

    model, processor = load_arm(args.adapter or None)
    sw = StatusWriter(f"S_PERTURB_{tag}", total=len(rows))
    sw.update(done=len(done), force=True)

    def emit(rec, text):
        parsed = vlm.parse_trace(text)
        matched = vlm.match_candidate(parsed["action"], rec["candidates"]) if parsed else None
        gt = f"{rec['gt_verb']} {rec['gt_noun']}"
        wm_top1 = rec["candidates"][max(range(len(rec["wm_scores"])),
                                        key=lambda j: rec["wm_scores"][j])]
        append_jsonl(rec_path, {
            "sample_id": rec["sample_id"], "action": matched, "gt": gt,
            "malformed": parsed is None or matched is None,
            "correct": matched == gt, "gt_in_support": gt in rec["candidates"],
            "history_length": len(rec.get("history", [])),
            "wm_top1": wm_top1, "wm_top1_correct": wm_top1 == gt,
            "task_belief": parsed["task_belief"] if parsed else None,
            "reasoning": parsed["reasoning"] if parsed else None,
            "perturb_mode": args.mode,
            "donor_sample_id": donors.get(rec["sample_id"], {}).get("sample_id")})

    n_seen = 0
    try:
        if no_frames:
            # decord 미진입 경로 — 프레임 추출이 없으므로 리더 상주도 없다 (OOM 구조적 회피).
            blanks = _blank_frames()
            for i0 in range(0, len(todo), args.batch_size):
                chunk = todo[i0:i0 + args.batch_size]
                t0 = time.time()
                msgs = [build_messages(r, blanks, args.mode, None) for r in chunk]
                texts = vlm.generate_batch(model, processor, msgs)
                for rec, text in zip(chunk, texts):
                    emit(rec, text)
                n_seen += len(chunk)
                sw.update(done=len(done) + n_seen,
                          metrics={"sec_per_sample": round((time.time() - t0) / max(1, len(chunk)), 2),
                                   "mode": args.mode})
        else:
            for chunk, frames in vlm.prefetch_chunks(video_root, todo, args.batch_size):
                t0 = time.time()
                msgs, ok = [], []
                for rec, (imgs, err) in zip(chunk, frames):
                    if err is not None:
                        append_jsonl(rec_path, {"sample_id": rec["sample_id"],
                                                "error": str(err)[:200], "malformed": True,
                                                "action": None, "perturb_mode": args.mode})
                    else:
                        msgs.append(build_messages(rec, imgs, args.mode,
                                                   donors.get(rec["sample_id"])))
                        ok.append(rec)
                texts = vlm.generate_batch(model, processor, msgs) if msgs else []
                for rec, text in zip(ok, texts):
                    emit(rec, text)
                n_seen += len(chunk)
                sw.update(done=len(done) + n_seen,
                          metrics={"sec_per_sample": round((time.time() - t0) / max(1, len(chunk)), 2),
                                   "mode": args.mode})
    finally:
        vlm.close_readers()   # 07-25 OOM 2차 방어 — 예외 경로에서도 반드시 반납한다

    # ── paired 분석: 원조건(배터리 산출물) vs 교란조건 ──
    sid_set = {r["sample_id"] for r in rows}
    base_recs = {r["sample_id"]: r
                 for r in read_jsonl(runs_root() / "eval" / f"{args.arm}.records.jsonl")
                 if r["sample_id"] in sid_set}
    pert = {r["sample_id"]: r for r in read_jsonl(rec_path) if r["sample_id"] in sid_set}
    paired = [s for s in base_recs if s in pert]

    def ok_of(d, s):
        return int(bool(d[s].get("correct")) and not d[s].get("malformed", False))

    all_diff = [ok_of(base_recs, s) - ok_of(pert, s) for s in paired]
    by_bin = defaultdict(list)
    for s in paired:
        by_bin[hbin(hlen.get(s))].append(ok_of(base_recs, s) - ok_of(pert, s))

    verdict = {
        "arm": args.arm, "mode": args.mode, "covered_only": covered_only,
        "n_paired": len(paired),
        "acc_full": round(sum(ok_of(base_recs, s) for s in paired) / max(1, len(paired)), 4),
        "acc_perturb": round(sum(ok_of(pert, s) for s in paired) / max(1, len(paired)), 4),
        "delta_acc_all": boot_ci(all_diff),
        "delta_acc_by_hlen": {k: boot_ci(v) for k, v in sorted(by_bin.items())},
        "pool_coverage": round(pool_cov, 4),
        "note": ("정책 경로만 개입 — WM 후보·wm_scores·top1 불변. 프레임 수 8 유지. "
                 "출력 스키마는 strip_eval 과 동일."),
    }
    out = runs_root() / "eval" / f"perturb_verdict_{tag}.json"
    out.write_text(json.dumps(verdict, indent=1, ensure_ascii=False))
    write_marker(marker, {"n": len(paired), "delta": verdict["delta_acc_all"]["delta"]})
    print(json.dumps(verdict, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
