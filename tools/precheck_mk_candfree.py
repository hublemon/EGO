"""Precheck (c): candidate-free support mass M_K — S2 전제(gap) 확인.

질문: 후보 목록을 프롬프트에서 치우면, 모델이 WM top-K 집합에 여전히 질량을 얹나?
  - candidate-free M_K가 이미 높다 → gap 없음 → S2 무의미 (fusion F로 충분?)
  - 낮다 → 메울 여지 있음 → S2 진행 근거

정의 (review §8.3, restricted-set 정규화):
  각 예시에서 S = WM top-K ∪ {GT} ∪ freq-matched negatives 를 점수화.
  action a의 length-norm logp를 candidate-free 프롬프트 prefix 아래 teacher-force로 계산,
  S 위에서 softmax → M_K = Σ_{a∈K} p(a).  대조: 후보 제시(candidate-presented) 동일 측정.

  reference:
   - random-K baseline: |K|/|S| (질량이 균등하면 이 값) — M_K가 이보다 높아야 "집합 선호"
   - GT-in-K mass: GT가 K에 있을 때 GT에 얹힌 질량 (판별력 GADR 근사)

사용: RETRO3_RUNS=runs/retro4 PYTHONPATH=src python3 tools/precheck_mk_candfree.py \
        --config configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml [--n 300] [--adapter PATH]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.runtime import read_jsonl, runs_root

CANDFREE_USER = (
    "Completed actions so far (oldest to newest):\n{hist}\n\n"
    "Predict the single next action the person does. Respond in the required "
    "format; the <action> line must be a 'verb noun' pair."
)


def candfree_user(rec: dict) -> str:
    return CANDFREE_USER.format(hist=vlm.fmt_history(rec))


def messages_text(rec: dict, candidate_free: bool) -> list[dict]:
    """텍스트 전용 메시지 — candidate_free면 후보 목록 제거 (intervention.py와 동일 계보,
    이미지 없이 스코어링해 Qwen3-VL 배치-RoPE 충돌 회피)."""
    user = candfree_user(rec) if candidate_free else vlm.user_prompt(rec)
    return [{"role": "system", "content": [{"type": "text", "text": vlm.SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user}]}]


@torch.no_grad()
def score_set(model, processor, messages, actions: list[str], device):
    """restricted-set S(=actions) 각 원소의 length-norm logp → softmax 확률 벡터 (텍스트 전용)."""
    tok = processor.tokenizer
    prefix = "<|im_start|>assistant\n<action>\n"  # 직접 action 점수화 (reasoning 무관 순수 선호)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
    base = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    aid = [tok(a, add_special_tokens=False)["input_ids"] for a in actions]
    maxc = max(len(c) for c in aid)
    B, L = len(aid), len(base) + maxc
    input_ids = base.new_full((B, L), tok.eos_token_id)
    attn = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, cids in enumerate(aid):
        seq = torch.cat([base, torch.tensor(cids, device=device)])
        input_ids[i, :len(seq)] = seq
        attn[i, :len(seq)] = 1
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = input_ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    scores = torch.stack([tok_lp[i, len(base) - 1: len(base) - 1 + len(c)].mean()
                          for i, c in enumerate(aid)])
    return torch.softmax(scores, dim=0).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_neg", type=int, default=10, help="freq-matched negative 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    device = "cuda"
    rows = read_jsonl(runs_root() / "data" / "context_val.jsonl")
    rows = [r for r in rows if r["split"] == "heldout"]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]

    # 전역 action 빈도 (freq-matched negative 풀)
    freq = Counter()
    for r in read_jsonl(runs_root() / "data" / "support_val.jsonl"):
        freq.update(r["candidates"])
    pool = list(freq)
    weights = [freq[a] for a in pool]

    model, processor = vlm.load_model()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    # 결정적 대조: 자기 WM집합(own) vs 다른 예시 WM집합(other)의 비겹침 부분에 얹힌 질량.
    #   own_cf ≈ other_cf  → base에 instance-specific candidate-free 신호 없음 → S2 헤드룸 有
    #   own_cf ≫ other_cf  → base가 이미 집합을 구별 → S2 불필요 (fusion 검토)
    all_sets = [r["candidates"] for r in read_jsonl(runs_root() / "data" / "support_val.jsonl")]
    agg = {"own_cf": [], "other_cf": [], "own_cp": [], "other_cp": []}
    for i, rec in enumerate(rows):
        K = set(rec["candidates"])
        other = set(rng.choice(all_sets))
        own_only = sorted(K - other)      # 자기집합 고유
        other_only = sorted(other - K)    # 다른집합 고유 (대조 negative)
        if not own_only or not other_only:
            continue
        S = own_only + other_only
        own_idx = list(range(len(own_only)))
        other_idx = list(range(len(own_only), len(S)))
        p_cf = score_set(model, processor, messages_text(rec, candidate_free=True), S, device)
        p_cp = score_set(model, processor, messages_text(rec, candidate_free=False), S, device)
        # 비겹침 부분 내에서 own이 차지한 상대 질량 (0.5=구별 못함, 1=완전 구별)
        agg["own_cf"].append(float(p_cf[own_idx].sum()))
        agg["other_cf"].append(float(p_cf[other_idx].sum()))
        agg["own_cp"].append(float(p_cp[own_idx].sum()))
        agg["other_cp"].append(float(p_cp[other_idx].sum()))
        if (i + 1) % 50 == 0:
            cf = sum(agg["own_cf"]) / len(agg["own_cf"])
            print(f"  {i+1}/{len(rows)}  own_cf={cf:.3f} other_cf={sum(agg['other_cf'])/len(agg['other_cf']):.3f}")

    mean = lambda xs: round(sum(xs) / max(1, len(xs)), 4)  # noqa: E731
    disc_cf = mean(agg["own_cf"]) - mean(agg["other_cf"])   # candidate-free 구별력
    disc_cp = mean(agg["own_cp"]) - mean(agg["other_cp"])   # candidate-presented 구별력 (상한 참조)
    report = {
        "n": len(agg["own_cf"]), "adapter": args.adapter or "base",
        "own_mass_candidate_free": mean(agg["own_cf"]),
        "other_mass_candidate_free": mean(agg["other_cf"]),
        "discrimination_candidate_free": round(disc_cf, 4),
        "own_mass_candidate_presented": mean(agg["own_cp"]),
        "other_mass_candidate_presented": mean(agg["other_cp"]),
        "discrimination_candidate_presented": round(disc_cp, 4),
        "headroom_cp_minus_cf": round(disc_cp - disc_cf, 4),
    }
    if disc_cf < 0.1:
        report["verdict_hint"] = (f"candidate-free 구별력 {disc_cf} ≈ 0 — base는 자기/타 집합을 못 가림 "
                                  f"→ S2가 채울 instance-specific 헤드룸 큼 (진행 근거)")
    elif disc_cf > 0.4:
        report["verdict_hint"] = (f"candidate-free 구별력 {disc_cf} 이미 높음 — base가 집합을 구별 "
                                  f"→ S2 헤드룸 작음, fusion F 우선 검토")
    else:
        report["verdict_hint"] = (f"candidate-free 구별력 {disc_cf}, presented 상한 {disc_cp} — "
                                  f"헤드룸 {report['headroom_cp_minus_cf']} 존재, S2 여지 有")
    out = runs_root() / "eval" / f"precheck_mk_{report['adapter'].replace('/', '_')[:40]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\n[precheck] written: {out}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
