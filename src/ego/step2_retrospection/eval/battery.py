"""평가 배터리 — 생성 평가 + G1/G2 분해 + L0 강제 (Handoff 1 §14 / 2 §17).

arm 하나(base 또는 base+adapter)를 heldout(고정 seed 서브셋)에서 평가한다.
- SelAcc@K(생성→후보 매칭), malformed rate
- L0 = wm_scores argmax 추종 (모든 결과에 강제 병기)
- G1 = wm top1 == GT 부분집합 정확도(retention), G2 = GT∈support & wm top1 != GT(correction)
출력: runs/retro3/eval/{arm}.json + {arm}.records.jsonl

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.eval.battery --arm base [--adapter PATH] \
      [--eval_n 1000] [--split_name heldout]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, read_jsonl, runs_root, write_marker


def load_arm(adapter: str | None):
    model, processor = vlm.load_model()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    return model, processor


def pick_eval_set(rows, split_name: str, n: int, seed: int = 42, covered_only: bool = False,
                  top_k: int = 10):
    """covered_only: GT∈Top-K(gt_rank<=top_k) 샘플만 — 후보 매칭 구조상 uncovered는 전 arm 강제 0점.
    반환: (eval rows, pool_coverage) — pool_coverage는 필터 전 split 전체의 cov@K (full-set 환산용).

    top_k<10: K ablation (main.tex tab:kablation). 후보를 wm_scores 상위 K 로 절단한 **복사본**을
    돌려주므로 원본 context_val 은 불변이다. gt_rank 는 WM 전체 분포 기준이라 gt_rank<=K 가
    곧 top-K 커버리지 조건이며, 절단된 후보 집합과 정합적이다."""
    rows = [r for r in rows if r["split"] == split_name]
    pool_coverage = sum(r["gt_rank"] <= top_k for r in rows) / max(1, len(rows))
    if covered_only:
        rows = [r for r in rows if r["gt_rank"] <= top_k]
    if top_k < 10:
        cut = []
        for r in rows:
            order = sorted(range(len(r["wm_scores"])), key=lambda j: -r["wm_scores"][j])[:top_k]
            r = dict(r)
            r["candidates"] = [r["candidates"][j] for j in order]
            r["wm_scores"] = [r["wm_scores"][j] for j in order]
            cut.append(r)
        rows = cut
    rng = random.Random(seed)
    rng.shuffle(rows)
    return sorted(rows[:n], key=lambda r: r["sample_id"]), pool_coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_start_m1_lobs8.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split_name", default="heldout", choices=["heldout", "dev"])
    ap.add_argument("--eval_n", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--covered_only", action="store_true",
                    help="GT∈Top-10 샘플만 평가 (조건부 지표 — full-set acc/L0는 pool_coverage 곱으로 병기)")
    # 2026-07-27: malformed 의 실체가 </action> 누락임을 확인(vlm.TAG_RE_LENIENT 주석).
    # 기본값은 strict — 이미 산출된 셀과 섞이면 안 된다. 켜는 실행은 전 arm 을 같은 규칙으로.
    ap.add_argument("--lenient", action="store_true",
                    help="닫는 태그 없이 EOS 로 끝난 마지막 태그를 받아들인다")
    ap.add_argument("--save_text", action="store_true",
                    help="생성 원문을 레코드에 저장 — 파싱 규칙 변경을 재실행 없이 재채점")
    ap.add_argument("--top_k", type=int, default=10,
                    help="후보를 wm_scores 상위 K 로 절단 (K ablation). 기본 10 = 절단 없음")
    args = ap.parse_args()

    # 실행 중 체인에 CLI 수정 없이 값을 주입하는 통로 (runs/retro3/overrides.json).
    # 2026-07-22 사용자 확정: covered-only 평가 — uncovered는 구조적 0점이라 arm 비교력 희석.
    ov_path = runs_root() / "overrides.json"
    if ov_path.is_file():
        ov = json.loads(ov_path.read_text())
        if ov.get("eval_covered_only") and not args.covered_only:
            print("[override] covered_only on (overrides.json)")
            args.covered_only = True

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    rows, pool_coverage = pick_eval_set(read_jsonl(runs_root() / "data" / "context_val.jsonl"),
                                        args.split_name, args.eval_n,
                                        covered_only=args.covered_only, top_k=args.top_k)

    out_dir = runs_root() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"{args.arm}.records.jsonl"
    done = {r["sample_id"] for r in read_jsonl(rec_path)}
    todo = [r for r in rows if r["sample_id"] not in done]
    todo.sort(key=lambda r: (r["video_uid"], r["obs_start_sec"]))  # 리더 캐시 히트↑ (집계는 전체 재로드라 순서 무관)

    model, processor = load_arm(args.adapter)
    sw = StatusWriter(f"S7_eval_{args.arm}", total=len(rows))
    sw.update(done=len(done), force=True)

    n_seen = 0
    # prefetch_chunks: 다음 chunk 프레임 추출(CPU)이 현재 chunk 생성(GPU)과 겹침
    for chunk, frames in vlm.prefetch_chunks(video_root, todo, args.batch_size):
        t0 = time.time()
        msgs, ok = [], []
        for rec, (imgs, err) in zip(chunk, frames):
            if err is not None:
                append_jsonl(rec_path, {"sample_id": rec["sample_id"], "error": str(err)[:200],
                                        "malformed": True, "action": None})
            else:
                msgs.append(vlm.build_messages(rec, imgs))
                ok.append(rec)
        texts = vlm.generate_batch(model, processor, msgs) if msgs else []
        for rec, text in zip(ok, texts):
            parsed = vlm.parse_trace(text, lenient=args.lenient)
            strict = vlm.parse_trace(text)          # 관대 규칙 구제 건수 집계용
            matched = vlm.match_candidate(parsed["action"], rec["candidates"]) if parsed else None
            gt = f"{rec['gt_verb']} {rec['gt_noun']}"
            wm_top1 = rec["candidates"][max(range(len(rec["wm_scores"])),
                                            key=lambda j: rec["wm_scores"][j])]
            append_jsonl(rec_path, {
                "sample_id": rec["sample_id"], "action": matched, "gt": gt,
                "malformed": parsed is None or matched is None,
                "correct": matched == gt,
                "gt_in_support": gt in rec["candidates"],
                "wm_top1": wm_top1, "wm_top1_correct": wm_top1 == gt,
                "task_belief": parsed["task_belief"] if parsed else None,
                "reasoning": parsed["reasoning"] if parsed else None,
                "malformed_strict": strict is None,
                "recovered": parsed is not None and strict is None,
                **({"text": text} if args.save_text else {})})
        n_seen += len(chunk)
        sw.update(done=len(done) + n_seen,
                  metrics={"sec_per_sample": round((time.time() - t0) / max(1, len(chunk)), 2)})

    # 집계 — L0 행 강제
    recs = [r for r in read_jsonl(rec_path) if r["sample_id"] in {x["sample_id"] for x in rows}]
    n = len(recs)
    acc = sum(r.get("correct", False) for r in recs) / max(1, n)
    l0 = sum(r.get("wm_top1_correct", False) for r in recs) / max(1, n)
    g1 = [r for r in recs if r.get("wm_top1_correct")]
    g2 = [r for r in recs if r.get("gt_in_support") and not r.get("wm_top1_correct")]
    summary = {
        "arm": args.arm, "split": args.split_name, "n": n,
        "acc": round(acc, 4),
        "L0_wm_top1": round(l0, 4),
        "beats_L0": acc > l0,
        "G1_retention": round(sum(r["correct"] for r in g1) / max(1, len(g1)), 4),
        "G1_n": len(g1),
        "G2_correction": round(sum(r["correct"] for r in g2) / max(1, len(g2)), 4),
        "G2_n": len(g2),
        "malformed_rate": round(sum(r.get("malformed", False) for r in recs) / max(1, n), 4),
        "recovered_rate": round(sum(r.get("recovered", False) for r in recs) / max(1, n), 4),
        "lenient": args.lenient,
        "coverage_at_k": round(sum(r.get("gt_in_support", False) for r in recs) / max(1, n), 4),
        # 조건부 지표 표기 + full-set 환산: uncovered는 후보 매칭 구조상 correct/L0 모두 0이므로
        # full-set acc = acc(covered) × pool_coverage 가 정확한 환산 (malformed는 covered-set 지표).
        "covered_only": args.covered_only,
        "top_k": args.top_k,                       # K ablation 출처 표기 (기본 10 = 절단 없음)
        "pool_coverage": round(pool_coverage, 4),
        "acc_full_equiv": round(acc * pool_coverage, 4) if args.covered_only else round(acc, 4),
        "L0_full_equiv": round(l0 * pool_coverage, 4) if args.covered_only else round(l0, 4),
    }
    (out_dir / f"{args.arm}.json").write_text(json.dumps(summary, indent=1))
    write_marker(f"S7_EVAL_{args.arm.upper()}_DONE", summary)
    sw.finish(metrics=summary)
    print(f"[S7:{args.arm}] {json.dumps(summary)}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
