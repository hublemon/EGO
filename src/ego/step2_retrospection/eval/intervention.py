"""Belief 개입 테스트 ③ — 유일한 인과 성공 판정 (Handoff 1 §11 / 2 §14).

각 샘플의 **모델 생성 belief**(battery records)를 조건으로 Top-K 후보 action-span
logp를 채점하고, belief 변형별 p(a_GT) 변화를 잰다:
  gen        생성된 belief 그대로
  empty      belief 블록 제거
  incompat   다른 샘플의 belief로 교체 (swap)
  paraphrase 같은 belief의 동의어 재작성 (모델 자신이 생성 — 통제군)

성공 부등식: p(gen) > p(empty) > p(incompat), 보고는 swap−paraphrase 차감 포함.
LLM judge 없음.

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.eval.intervention --arm base \
      [--adapter PATH] [--n 300]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.eval.battery import load_arm
from ego.step2_retrospection.runtime import StatusWriter, read_jsonl, runs_root, write_marker


@torch.no_grad()
def candidate_probs(model, processor, rec, reasoning: str, belief: str | None, device="cuda"):
    """prefix(프롬프트+reasoning[+belief]) 아래 후보별 length-norm logp → softmax."""
    tok = processor.tokenizer
    prefix = "<|im_start|>assistant\n<reasoning>\n" + reasoning.strip() + "\n</reasoning>\n\n"
    if belief is not None:
        prefix += "<task_belief>\n" + belief.strip() + "\n</task_belief>\n\n"
    prefix += "<action>\n"
    text = processor.apply_chat_template(
        vlm.build_messages(rec, []), tokenize=False, add_generation_prompt=True) + prefix
    base = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    cand_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in rec["candidates"]]
    maxc = max(len(c) for c in cand_ids)
    B, L = len(cand_ids), len(base) + maxc
    input_ids = torch.full((B, L), tok.eos_token_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, cids in enumerate(cand_ids):
        seq = torch.cat([base, torch.tensor(cids)])
        input_ids[i, :len(seq)] = seq
        attn[i, :len(seq)] = 1
    logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = input_ids[:, 1:].to(device)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    scores = torch.stack([tok_lp[i, len(base) - 1: len(base) - 1 + len(c)].mean()
                          for i, c in enumerate(cand_ids)])
    return torch.softmax(scores, dim=0).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_start_m1_lobs8.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    yaml.safe_load(open(args.config, encoding="utf-8"))
    ctx = {r["sample_id"]: r for r in read_jsonl(runs_root() / "data" / "context_val.jsonl")}
    recs = [r for r in read_jsonl(runs_root() / "eval" / f"{args.arm}.records.jsonl")
            if not r.get("malformed") and r.get("task_belief") and r.get("gt_in_support")]
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    recs = recs[: args.n]

    model, processor = load_arm(args.adapter)
    device = "cuda"
    sw = StatusWriter(f"S7_intervention_{args.arm}", total=len(recs))

    # paraphrase 통제군: 모델 자신이 belief를 재작성 (판정 아님 — judge 정책과 무관)
    para_msgs = [[{"role": "user", "content": [{"type": "text", "text":
        f'Paraphrase this sentence, keeping the exact same meaning: "{r["task_belief"]}"\n'
        "Reply with the paraphrase only."}]}] for r in recs]
    paras = []
    for i0 in range(0, len(para_msgs), 16):
        paras.extend(vlm.generate_batch(model, processor, para_msgs[i0:i0 + 16], max_new_tokens=60))

    agg = {k: [] for k in ("gen", "empty", "incompat", "paraphrase")}
    flips = {"swap": 0, "para": 0}
    out_rows = []
    for i, r in enumerate(recs):
        rec = ctx[r["sample_id"]]
        gt_idx = rec["candidates"].index(r["gt"])
        other = recs[(i + 7) % len(recs)]["task_belief"]  # incompat: 다른 샘플 belief
        variants = {"gen": r["task_belief"], "empty": None,
                    "incompat": other, "paraphrase": paras[i].strip().strip('"')}
        ps, top1 = {}, {}
        for name, b in variants.items():
            p = candidate_probs(model, processor, rec, r["reasoning"], b, device)
            ps[name] = float(p[gt_idx])
            top1[name] = int(p.argmax())
        for k, v in ps.items():
            agg[k].append(v)
        flips["swap"] += int(top1["incompat"] != top1["gen"])
        flips["para"] += int(top1["paraphrase"] != top1["gen"])
        out_rows.append({"sample_id": r["sample_id"], **{f"p_{k}": round(v, 4) for k, v in ps.items()}})
        sw.update(done=i + 1, metrics={k: round(sum(v) / len(v), 4) for k, v in agg.items() if v})

    mean = {k: sum(v) / max(1, len(v)) for k, v in agg.items()}
    n = len(recs)
    summary = {
        "arm": args.arm, "n": n,
        "p_gt": {k: round(v, 4) for k, v in mean.items()},
        "inequality_gen_gt_empty": mean["gen"] > mean["empty"],
        "inequality_empty_gt_incompat": mean["empty"] > mean["incompat"],
        "margin_gen_minus_incompat": round(mean["gen"] - mean["incompat"], 4),
        "flip_swap": round(flips["swap"] / max(1, n), 4),
        "flip_paraphrase_control": round(flips["para"] / max(1, n), 4),
        "causal_sensitivity": round((flips["swap"] - flips["para"]) / max(1, n), 4),
    }
    out_dir = runs_root() / "eval"
    (out_dir / f"{args.arm}.intervention.json").write_text(json.dumps(summary, indent=1))
    (out_dir / f"{args.arm}.intervention.records.json").write_text(json.dumps(out_rows))
    sw.finish(metrics=summary)
    write_marker(f"S7_INTERVENTION_{args.arm.upper()}_DONE", summary)
    print(f"[S7:③:{args.arm}] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
