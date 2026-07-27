#!/usr/bin/env python3
"""strip-eval — history 인과를 같은 체크포인트에서 paired 로 측정 (B_nohist 대체, 핸드오프 §2).

별도 no_history arm 을 4h 학습하는 대신, **이미 학습된 어댑터** 하나로
같은 eval 셋을 두 조건에서 추론한다:
  - with-history  : 배터리 산출물({arm}.records.jsonl) 재사용
  - history-strip : 동일 체크포인트, 프롬프트의 history 만 '(history withheld)' 로 치환
                    (WM 후보 불변)
per-sample_id paired 차이라 "다른 체크포인트" 교란이 없다 (EGO_jihun --no_memory 패턴).

판정: Δacc = acc(hist) − acc(strip), 전체 + history_length 층화, bootstrap CI.
  H8(긴 history)에서 Δ>0 이면 해당 체크포인트가 history 를 인과적으로 사용.
covered 모집단·video-cluster CI 로 다시 보려면 `tools/strip_metrics.py`
(대시보드 규약, GPU 불필요)를 쓴다 — 본 스크립트의 verdict 는 full-set·sample bootstrap.

**GPU 잡** — orchestrator 와 동시 실행 금지. post_theta_ce_hook.sh 가
CESFT_V2_CHAIN_DONE 이후(=GPU 여유) 호출한다.

arm 별 실행 (--arm/--adapter/--covered_only):
  theta_ce (기본)      : 어댑터 outputs/.../theta_ce/adapter
  base                 : --arm base --adapter "" (어댑터 없음 = 소재 VLM)
  sft_r15 / sft_r0 …   : --arm sft_r15 --adapter outputs/.../sft_r15/adapter
배터리가 covered-only 로 평가된 arm(base 등)과 짝을 맞추려면 --covered_only 를 켠다.

출력: runs/cesft_v2/eval/{arm}_nohist.records.jsonl · strip_verdict[_{arm}].json
마커: S_STRIP_{ARM}_DONE  (theta_ce 는 기존 이름 S_STRIP_THETA_CE_DONE 유지)
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

# ── 프롬프트 레짐 ────────────────────────────────────────────────────────────────
# 2026-07-27 발견: cesft_v2 배터리(07-24/25)는 **3인칭 프롬프트**로 돌았고, 그 뒤 1인칭
# 일원화(rerun handoff §4)로 vlm.SYSTEM_PROMPT 와 user_prompt 헤더가 바뀌었다. strip 조건을
# 현행 코드로 돌리면 Δ 가 "history 제거"가 아니라 "프롬프트 교체"를 재게 된다 — 실측 확인:
# 같은 base·history 포함인데 1인칭율 0.0%→89.9%, 장면 묘사 56.3→35.3, 선택 일치 51.7%.
# 따라서 hist 조건이 어느 레짐이냐에 따라 strip 도 같은 레짐으로 돌려야 한다.
#   legacy3p : 07-24/25 배터리·대시보드·논문 수치와 짝이 맞는 레짐 (기본값)
#   current  : 현행 vlm.py (1인칭). hist 조건도 같은 레짐으로 새로 돌린 경우에만 유효.
# 원문 출처(독립 사본 2곳 md5 일치): backups/EGO_paper_backup_20260725/06_code/
# src_step2_retrospection/vlm.py · EGO_jihun2/src/ego/step2_retrospection/vlm.py
LEGACY_SYSTEM_PROMPT = (
    "You are an egocentric activity assistant. You see frames from the last 8 seconds of a "
    "first-person video, a list of actions the person already COMPLETED, and a shuffled list "
    "of candidate next actions. Each action is 'verb noun'. Exactly ONE candidate is what the "
    f"person does next ({vlm.NEXT_GAP_TEXT}).\n"
    "Respond in EXACTLY this format:\n"
    "<reasoning>\nCompare the candidates against the visual scene and the completed-action "
    "history. 3-6 sentences.\n</reasoning>\n"
    "<task_belief>\nOne sentence: the local procedure or subgoal the person is currently in. "
    "Do NOT name the chosen next action verbatim.\n</task_belief>\n"
    "<action>\nverb noun\n</action>\n"
    "The <action> line must copy one candidate EXACTLY as written."
)
LEGACY_HIST_HEADER = "Completed actions so far (oldest to newest):"
CURRENT_HIST_HEADER = "Your completed actions so far (oldest to newest):"


def user_prompt_nohist(rec: dict, regime: str) -> str:
    """battery user_prompt 와 동일 — history 텍스트만 공란 문구로 치환 (WM 후보 불변)."""
    header = LEGACY_HIST_HEADER if regime == "legacy3p" else CURRENT_HIST_HEADER
    return (f"{header}\n(history withheld)\n\n"
            f"Candidate next actions (shuffled):\n{vlm.fmt_candidates(rec['candidates'])}\n\n"
            "Which candidate is the next action? Follow the required format.")


def build_messages_hist(rec: dict, images, regime: str) -> list[dict]:
    """with-history 조건 — 레짐에 맞는 system/user 헤더로 배터리 프롬프트를 재구성.

    아카이브된 배터리 records 를 쓰지 않고 **같은 세션에서** hist 조건을 다시 돌리기 위한 경로.
    2026-07-27 실측: greedy 인데도 batch 구성이 다르면 선택이 31.7% 바뀌고 acc 가
    +5.8pp [0.0, 11.7] 움직인다(동일 프롬프트·동일 체크포인트). 즉 "아카이브 hist vs 신규 strip"
    비교는 배치 잡음을 개입 효과로 오인할 수 있다 → 두 조건을 같은 세션·같은 batch 로 돌린다.
    """
    header = LEGACY_HIST_HEADER if regime == "legacy3p" else CURRENT_HIST_HEADER
    system = LEGACY_SYSTEM_PROMPT if regime == "legacy3p" else vlm.SYSTEM_PROMPT
    text = (f"{header}\n{vlm.fmt_history(rec)}\n\n"
            f"Candidate next actions (shuffled):\n{vlm.fmt_candidates(rec['candidates'])}\n\n"
            "Which candidate is the next action? Follow the required format.")
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content}]


def build_messages_nohist(rec: dict, images, regime: str) -> list[dict]:
    system = LEGACY_SYSTEM_PROMPT if regime == "legacy3p" else vlm.SYSTEM_PROMPT
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": user_prompt_nohist(rec, regime)})
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
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
    ap.add_argument("--arm", default="theta_ce",
                    help="hist 조건 배터리 산출물 이름 — {arm}.records.jsonl 과 짝을 맞춘다")
    ap.add_argument("--covered_only", action="store_true",
                    help="GT∈Top-10 만 평가 (base 처럼 covered-only 로 평가된 arm 과 짝 맞추기)")
    ap.add_argument("--prompt_regime", default="legacy3p", choices=["legacy3p", "current"],
                    help="hist 조건과 같은 프롬프트 레짐을 쓸 것 — 기본 legacy3p(07-24/25 배터리)")
    ap.add_argument("--condition", default="strip", choices=["strip", "hist"],
                    help="strip=history 제거 · hist=history 포함(같은 세션 대조군 재실행)")
    ap.add_argument("--tag", default="",
                    help="산출 파일 접미사 — 세션이 섞이지 않게 한 쌍을 같은 tag 로 묶는다")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])

    # 배터리와 동일 서브셋 (CLI --covered_only 우선, 없으면 overrides.json)
    covered_only = args.covered_only
    ovp = runs_root() / "overrides.json"
    if not covered_only and ovp.is_file():
        covered_only = bool(json.loads(ovp.read_text()).get("eval_covered_only"))
    ctx = read_jsonl(runs_root() / "data" / "context_val.jsonl")
    rows, pool_cov = pick_eval_set(ctx, args.split_name, args.eval_n, covered_only=covered_only)
    hlen = {r["sample_id"]: len(r.get("history", [])) for r in ctx}
    vid = {r["sample_id"]: r.get("video_uid") for r in ctx}

    # 레짐·조건·세션(tag)별로 파일을 분리한다 — 섞이면 Δ 가 다른 요인을 재게 된다.
    cond = "nohist" if args.condition == "strip" else "hist"
    regime_sfx = "" if args.prompt_regime == "legacy3p" else "_fp"
    rec_path = runs_root() / "eval" / f"{args.arm}_{cond}{regime_sfx}{args.tag}.records.jsonl"
    done = {r["sample_id"] for r in read_jsonl(rec_path)}
    todo = [r for r in rows if r["sample_id"] not in done]
    todo.sort(key=lambda r: (r["video_uid"], r["obs_start_sec"]))

    from ego.step2_retrospection.eval.battery import load_arm
    model, processor = load_arm(args.adapter or None)
    sw = StatusWriter(f"S_STRIP_{args.arm}_{cond}{args.tag}", total=len(rows))
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
                msgs.append(build_messages_nohist(rec, imgs, args.prompt_regime)
                            if args.condition == "strip"
                            else build_messages_hist(rec, imgs, args.prompt_regime))
                ok.append(rec)
        texts = vlm.generate_batch(model, processor, msgs) if msgs else []
        for rec, text in zip(ok, texts):
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
                # WM 후보·top1 은 개입과 무관(불변) — 조건부 지표(GADR/G1) 자체 계산용.
                "wm_top1": wm_top1, "wm_top1_correct": wm_top1 == gt,
                # strip 조건의 트레이스 언어 지표(장면 묘사·배제 언명 등) 재계산용.
                "task_belief": parsed["task_belief"] if parsed else None,
                "reasoning": parsed["reasoning"] if parsed else None})
        n_seen += len(chunk)
        sw.update(done=len(done) + n_seen,
                  metrics={"sec_per_sample": round((time.time() - t0) / max(1, len(chunk)), 2)})

    if args.condition == "hist":
        sw.finish(metrics={"condition": "hist", "n": len(rows)})
        write_marker(f"S_STRIP_{args.arm.upper()}_HIST{args.tag.upper()}_DONE",
                     {"arm": args.arm, "regime": args.prompt_regime, "n": len(rows)})
        print(json.dumps({"arm": args.arm, "condition": "hist",
                          "prompt_regime": args.prompt_regime,
        "hist_source": hist_src.name,
        "tag": args.tag, "n": len(rows)}, ensure_ascii=False))
        vlm.close_readers()
        return

    # ── paired 분석: hist vs strip (같은 tag 의 hist 재실행이 있으면 그쪽을 쓴다) ──
    sid_set = {r["sample_id"] for r in rows}
    hist_own = runs_root() / "eval" / f"{args.arm}_hist{regime_sfx}{args.tag}.records.jsonl"
    hist_src = hist_own if hist_own.is_file() else runs_root() / "eval" / f"{args.arm}.records.jsonl"
    hist = {r["sample_id"]: r for r in read_jsonl(hist_src) if r["sample_id"] in sid_set}
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
        "arm": args.arm,
        "prompt_regime": args.prompt_regime,
        "hist_source": hist_src.name,
        "tag": args.tag,
        "covered_only": covered_only,
        "n_paired": len(paired),
        "acc_hist": round(sum(ok_h(s) for s in paired) / max(1, len(paired)), 4),
        "acc_strip": round(sum(ok_s(s) for s in paired) / max(1, len(paired)), 4),
        "delta_acc_all": boot_ci(all_diff),
        "delta_acc_by_hlen": {b: boot_ci(by_bin[b]) for b in ("H0", "H1-3", "H4-7", "H8") if by_bin.get(b)},
        "gate_history_causal_H8": bool(by_bin.get("H8") and boot_ci(by_bin["H8"])["ci"][0] > 0),
        "note": f"same {args.arm}, history-only intervention (WM 후보 불변). "
                "B_nohist 학습-arm 대체.",
    }
    # theta_ce full-set(2026-07-24) 산출물을 덮지 않도록 arm·모집단별로 파일을 분리.
    vname = (f"strip_verdict{'' if args.arm == 'theta_ce' else '_' + args.arm}"
             f"{'_covered' if covered_only else ''}"
             f"{'' if args.prompt_regime == 'legacy3p' else '_fp'}{args.tag}.json")
    (runs_root() / "eval" / vname).write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1))
    sw.finish(metrics={"delta_acc_all": verdict["delta_acc_all"]["delta"],
                       "H8_causal": verdict["gate_history_causal_H8"]})
    write_marker(f"S_STRIP_{args.arm.upper()}_DONE", verdict)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    vlm.close_readers()


if __name__ == "__main__":
    main()
