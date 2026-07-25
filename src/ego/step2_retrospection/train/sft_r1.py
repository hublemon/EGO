"""R1: Projected-trace distillation (span-normalized SFT, Handoff 1 §9) + 선택적 R2 aux (§10).

학습 데이터: chosen_train.jsonl (gate pass) ⋈ context_train.jsonl (프롬프트 재구성)
loss: λ_b·L_b + λ_r·L_r + λ_a·L_a  (λ = 1.0/0.5/0.25)
R2 (--lambda_ba > 0): projected belief prefix 아래 Top-K 후보 정규화 CE를 가산.

출력: outputs/step2_retrospection/{run_name}/adapter (LoRA), train_log.jsonl
사용:
  PYTHONPATH=src python -m ego.step2_retrospection.train.sft_r1 \
      --run_name r1_sft --epochs 2 [--lambda_ba 0.5] [--limit N] [--warmup_subset N]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, read_jsonl, runs_root, write_marker
from ego.step2_retrospection.train import common as C


def lora_wrap(model, r: int = 16):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=r * 2, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def r2_candidate_loss(model, processor, rec, chosen_fields, device):
    """belief-conditioned candidate CE (Handoff 1 §10): 후보 K개 action-span logp
    를 한 배치 forward로 계산 (프롬프트+r_proj+b_proj prefix 공유)."""
    tok = processor.tokenizer
    prefix = ("<|im_start|>assistant\n<reasoning>\n" + chosen_fields["reasoning"].strip() +
              "\n</reasoning>\n\n<task_belief>\n" + chosen_fields["task_belief"].strip() +
              "\n</task_belief>\n\n<action>\n")
    text = processor.apply_chat_template(
        vlm.build_messages(rec, []), tokenize=False, add_generation_prompt=True) + prefix
    base = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    cand_ids = [tok(c + "\n</action>", add_special_tokens=False)["input_ids"] for c in rec["candidates"]]
    maxc = max(len(c) for c in cand_ids)
    B = len(cand_ids)
    L = len(base) + maxc
    input_ids = torch.full((B, L), tok.eos_token_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, cids in enumerate(cand_ids):
        seq = torch.cat([base, torch.tensor(cids)])
        input_ids[i, :len(seq)] = seq
        attn[i, :len(seq)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = input_ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    scores = []
    for i, cids in enumerate(cand_ids):
        s, e = len(base) - 1, len(base) - 1 + len(cids)
        scores.append(tok_lp[i, s:e].mean())
    scores = torch.stack(scores)  # (15,) length-normalized
    gt_idx = rec["candidates"].index(f"{rec['gt_verb']} {rec['gt_noun']}")
    return -torch.log_softmax(scores, dim=0)[gt_idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_start_m1_lobs8.yaml")
    ap.add_argument("--run_name", default="r1_sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lambda_ba", type=float, default=0.0, help=">0 이면 R2 aux 활성")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup_subset", type=int, default=None,
                    help="D2 warm-up용: 앞 N개만 학습")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_every", type=int, default=100,
                    help="N step마다 고정 probe 8샘플 생성 (trace 진화 관측). 0=끔")
    args = ap.parse_args()

    # 실행 중 체인에 CLI 수정 없이 값을 주입하는 통로 (runs/retro3/overrides.json).
    # 2026-07-22 사용자 확정: SFT 2 epoch → 1 (시간 단축, −2.8h).
    ov_path = runs_root() / "overrides.json"
    if ov_path.is_file():
        ov = json.loads(ov_path.read_text())
        if "sft_epochs" in ov and args.run_name == "r1_sft":
            print(f"[override] epochs {args.epochs} -> {ov['sft_epochs']}")
            args.epochs = int(ov["sft_epochs"])

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    device = "cuda"
    data_dir = runs_root() / "data"

    ctx = {r["sample_id"]: r for r in read_jsonl(data_dir / "context_train.jsonl")}
    chosen = [r for r in read_jsonl(data_dir / "chosen_train.jsonl") if r.get("gate") == "pass"]
    random.Random(args.seed).shuffle(chosen)
    if args.warmup_subset:
        chosen = chosen[: args.warmup_subset]
    if args.limit:
        chosen = chosen[: args.limit]

    model, processor = vlm.load_model()
    model = lora_wrap(model)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = math.ceil(len(chosen) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    out_dir = Path(cfg["experiment"]["output_dir"]).parent / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    sw = StatusWriter(f"S6_{args.run_name}", total=len(chosen) * args.epochs)

    probe_recs = None
    if args.probe_every:
        from ego.step2_retrospection.train.probe_gen import build_probe_set, run_probe
        probe_recs = build_probe_set()

    step = n_seen = 0
    last_probe = -1
    ema = {"loss": None, "reasoning": None, "task_belief": None, "action": None}
    opt.zero_grad()
    for epoch in range(args.epochs):
        for rec_c in chosen:
            rec = ctx.get(rec_c["sample_id"])
            if rec is None:
                continue
            t0 = time.time()
            try:
                imgs = vlm.extract_frames(video_root, rec["video_uid"],
                                          rec["obs_start_sec"], rec["obs_end_sec"])
                p_in = C.build_prompt_inputs(processor, vlm.build_messages(rec, imgs), device)
                comp_ids, field_of = C.encode_completion(processor.tokenizer, {
                    "reasoning": rec_c["reasoning"], "task_belief": rec_c["task_belief"],
                    "action": rec_c["action"]})
                inputs, p_len = C.cat_completion(p_in, comp_ids, device)
                loss, per_field = C.span_ce_loss(model, inputs, p_len, field_of)
                if args.lambda_ba > 0:
                    loss = loss + args.lambda_ba * r2_candidate_loss(
                        model, processor, rec, rec_c, device)
                (loss / args.accum).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt.zero_grad()
                append_jsonl(log_path, {"skip_oom": rec["sample_id"]})
                continue
            n_seen += 1
            if n_seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
            lv = float(loss.detach())
            for k, v in [("loss", lv), *per_field.items()]:
                ema[k] = v if ema[k] is None else 0.98 * ema[k] + 0.02 * v
            if n_seen % 10 == 0:
                append_jsonl(log_path, {"step": step, "seen": n_seen, "epoch": epoch,
                                        "loss": round(lv, 4), **per_field,
                                        "sec": round(time.time() - t0, 2)})
            sw.update(done=n_seen, metrics={
                "step": step, "epoch": epoch,
                "loss_ema": round(ema["loss"], 4) if ema["loss"] else None,
                "belief_ce_ema": round(ema["task_belief"], 4) if ema["task_belief"] else None,
                "action_ce_ema": round(ema["action"], 4) if ema["action"] else None,
                "lr": sched.get_last_lr()[0]})
            if probe_recs and step != last_probe and (step % args.probe_every == 0 or n_seen == 1):
                last_probe = step
                pm = run_probe(model, processor, video_root, args.run_name, step, probe_recs)
                sw.update(metrics={"probe_acc": pm["probe_acc"]}, force=True)

    model.save_pretrained(out_dir / "adapter")
    sw.finish(metrics={"steps": step, "final_loss_ema": ema["loss"]})
    write_marker(f"S6_{args.run_name.upper()}_DONE", {"steps": step})
    print(f"[S6:{args.run_name}] steps={step} loss_ema={ema['loss']}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
