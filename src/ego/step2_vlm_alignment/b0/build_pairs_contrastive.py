"""build_pairs_contrastive.py — B0 P1+P2: 최소대조 preference pair 생성.

진단: DPO 는 chosen/rejected 를 가르는 '가장 쉬운 차이'를 배운다. MVP/R1 은 chosen=teacher
(frozen base VLM), rejected=FAA 라 **문체**가 가장 쉬운 차이였고, span 분해가 이를 확인했다
(belief +0.802 vs action +0.014). → 쌍은 가르치려는 것만 달라야 한다.

P1 (자기대조):   같은 FAA 의 롤아웃끼리 — GT 맞춘 trace ≻ 틀린 trace
                 (같은 모델·같은 분포 → 문체 상쇄, 남는 차이는 '정답에 닿은 추론'뿐)
P2 (최소대조):   같은 (reasoning, belief) + GT action ≻ 같은 (reasoning, belief) + 다른 후보
                 (앞부분이 글자까지 동일 → gradient 가 action 선택에만 집중)

⚠ 포맷 지름길 차단: P2 는 chosen/rejected 를 **둘 다** build_full_trace 로 정규 직렬화한다.
  한쪽만 재조립하면 모델이 action 이 아니라 포맷 차이를 학습한다. P2 의 chosen 은 따라서
  '모델 자신의 내용을 정규화한 것'이며(내용 불변), P1 은 양쪽 모두 원문 verbatim 이다.
  rejected 의 오답 action 은 반드시 **그 프롬프트의 후보 5개 중**에서 뽑는다(후보-밖 지름길 차단).

출력: pairs jsonl (+ pair_type 태그: p1_self / p2_minimal) + stats json.
전부 오답인 프롬프트(FAA 가 한 번도 못 맞춘 어려운 샘플)는 --out_hard 로 목록만 남기고,
gated teacher(build_dpo_dataset_r1)가 별도로 채운다.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .trace_utils import build_full_trace, canonical_action, parse_full_trace
from .validate_dpo_dataset import check_prompt_leakage


def _load(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def sample_group(ex_topk: list[dict], gt: dict) -> str:
    """원 topk 순서 기준 G1(top1==GT) / G2(GT∈top5, top1≠GT) / OUT."""
    keys = [canonical_action(a.get("verb"), a.get("noun")) for a in (ex_topk or [])[:5]]
    g = canonical_action(gt.get("verb"), gt.get("noun"))
    if g not in keys:
        return "OUT"
    return "G1" if keys and keys[0] == g else "G2"


def _mk_record(sample, chosen_raw, rejected_raw, pair_type, gt, meta_extra):
    """DPO record + 누설검사용 _leak_check (저장 시 제거)."""
    rec = {
        "record_id": meta_extra["record_id"],
        "prompt": sample["prompt"],
        "image_path": sample.get("image_path", ""),
        "chosen": chosen_raw,
        "rejected": rejected_raw,
        "metadata": {
            "pair_type": pair_type,
            "group": meta_extra.get("group"),
            "faa_checkpoint_hash": sample.get("faa_checkpoint_hash", ""),
            "builder_version": "contrastive_p12_v1",
        },
        "_leak_check": {
            "gt_action": gt,
            "gt_action_str": f"{gt['verb']} {gt['noun']}",
            "future_gt_actions": sample.get("future_gt_actions", []),
            "policy_history": sample.get("policy_history", []),
            "trigger_time": sample.get("trigger_time"),
        },
    }
    return rec


def build(samples, topk_by_id, max_p1=4, max_p2=2, seed=42):
    rng = random.Random(seed)
    pairs, hard = [], []
    st = {"n_samples": 0, "no_valid_parse": 0, "all_correct": 0, "all_wrong": 0,
          "mixed": 0, "p1_pairs": 0, "p2_pairs": 0,
          "by_group": {g: {"seen": 0, "mixed": 0, "pairs": 0} for g in ("G1", "G2", "OUT")}}

    for s in samples:
        st["n_samples"] += 1
        gt = s["gt_action"]
        gt_key = canonical_action(gt.get("verb"), gt.get("noun"))
        cands = s.get("candidates", [])
        grp = sample_group(topk_by_id.get(s["sample_id"], []), gt)
        st["by_group"][grp]["seen"] += 1

        # 8 롤아웃 파싱 → 정답군 / 오답군 (파싱 실패·중복 제거)
        seen_raw, correct, wrong = set(), [], []
        for t in s.get("faa_traces", []):
            tr = parse_full_trace(t)
            if not tr.is_complete():
                continue
            key = tr.raw.strip()
            if key in seen_raw:
                continue
            seen_raw.add(key)
            (correct if canonical_action(tr.verb, tr.noun) == gt_key else wrong).append(tr)

        if not correct and not wrong:
            st["no_valid_parse"] += 1
            continue
        if not correct:
            st["all_wrong"] += 1
            hard.append(s["sample_id"])       # teacher 가 채울 구간
            continue
        if not wrong:
            st["all_correct"] += 1
            continue                          # 대조가 없으니 학습 신호 없음
        st["mixed"] += 1
        st["by_group"][grp]["mixed"] += 1

        # ── P1: 자기대조 (양쪽 모두 원문 verbatim — 문체 대칭) ──────────────
        combos = [(c, w) for c in correct for w in wrong]
        rng.shuffle(combos)
        for i, (c, w) in enumerate(combos[:max_p1]):
            rec = _mk_record(s, c.raw, w.raw, "p1_self", gt,
                             {"record_id": f'{s["sample_id"]}:p1:{i}', "group": grp})
            pairs.append(rec); st["p1_pairs"] += 1; st["by_group"][grp]["pairs"] += 1

        # ── P2: 최소대조 (양쪽 모두 정규 직렬화 — 포맷 대칭, action 만 상이) ──
        others = [(str(x.get("verb")), str(x.get("noun"))) for x in cands
                  if canonical_action(x.get("verb"), x.get("noun")) != gt_key]
        if others:
            rng.shuffle(correct)
            for i, c in enumerate(correct[:max_p2]):
                ov, on = others[i % len(others)]
                chosen = build_full_trace(c.reasoning, c.belief, c.verb, c.noun)
                rejected = build_full_trace(c.reasoning, c.belief, ov, on)
                if chosen.strip() == rejected.strip():
                    continue
                rec = _mk_record(s, chosen, rejected, "p2_minimal", gt,
                                 {"record_id": f'{s["sample_id"]}:p2:{i}', "group": grp})
                pairs.append(rec); st["p2_pairs"] += 1; st["by_group"][grp]["pairs"] += 1

    return pairs, hard, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="b0_samples jsonl (8 롤아웃 병합본)")
    ap.add_argument("--train_jsonl", required=True, help="G1/G2 판정용 원 topk")
    ap.add_argument("--out_pairs", required=True)
    ap.add_argument("--out_hard", required=True, help="전부-오답 sample_id 목록(teacher 몫)")
    ap.add_argument("--out_stats", required=True)
    ap.add_argument("--max_p1", type=int, default=4, help="프롬프트당 P1 쌍 상한(편중 방지)")
    ap.add_argument("--max_p2", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    samples = _load(args.samples)[: args.limit or None]
    topk_by_id = {str(r.get("frame_id", "")): (r.get("topk_actions") or [])
                  for r in _load(args.train_jsonl)}
    pairs, hard, st = build(samples, topk_by_id, args.max_p1, args.max_p2)

    # 누설 검사 (프롬프트에 GT/미래가 노출되지 않았는가) — 위반 시 제외하고 집계
    kept, dropped = [], 0
    for rec in pairs:
        errs = check_prompt_leakage(rec)
        if errs:
            dropped += 1
            continue
        kept.append({k: v for k, v in rec.items() if k != "_leak_check"})
    st["leak_dropped"] = dropped
    st["final_pairs"] = len(kept)

    Path(args.out_pairs).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + ("\n" if kept else ""),
        encoding="utf-8")
    Path(args.out_hard).write_text(json.dumps(hard, ensure_ascii=False))
    Path(args.out_stats).write_text(json.dumps(st, ensure_ascii=False, indent=2))
    print(f"[done] pairs={len(kept)} (P1 {st['p1_pairs']} / P2 {st['p2_pairs']}) "
          f"hard={len(hard)} leak_dropped={dropped}")
    print(json.dumps({k: st[k] for k in ("n_samples", "mixed", "all_correct", "all_wrong",
                                         "no_valid_parse")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
