"""Field-balanced DPO (Handoff 2 §9–§11). D1(직행) / D2(--init_adapter로 SFT warm-up 이어받기).

pair: pairs_train.jsonl. policy = LoRA, ref = 같은 모델의 adapter-disabled 경로
(추가 메모리 0). loss = -logσ(β·(Δ_θ^FB − Δ_ref^FB)) · w_i.

가드레일 G3 (사전 등록): belief margin EMA가 오르는데 action margin EMA가 바닥이면
문체-학습 신호 → G3_STYLE_ABORT 마커 쓰고 중단.

사용:
  PYTHONPATH=src python -m ego.step2_retrospection.train.dpo_fb --run_name dpo_d1 [--init_adapter PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.runtime import StatusWriter, append_jsonl, read_jsonl, runs_root, write_marker
from ego.step2_retrospection.train import common as C
from ego.step2_retrospection.train.sft_r1 import lora_wrap

# 2026-07-23 사용자 확정: belief-only margin 폭주(문체-학습) 관측 후 우선순위 역전 —
# action이 손실을 지배하게. (이전: belief 1.0 / reasoning 0.5 / action 0.25)
FIELD_W = {"task_belief": 0.5, "reasoning": 0.5, "action": 1.0}


def fb_score(model, processor, rec, fields, images, device):
    """field별 length-norm logp 텐서 dict 반환 (loss_mode에 따라 합산 방식이 다름)."""
    p_in = C.build_prompt_inputs(processor, vlm.build_messages(rec, images), device)
    comp_ids, field_of = C.encode_completion(processor.tokenizer, fields)
    inputs, p_len = C.cat_completion(p_in, comp_ids, device)
    fps = C.field_logps(model, inputs, p_len, field_of)
    return {f: fps[f][0] for f in C.FIELDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_start_m1_lobs8.yaml")
    ap.add_argument("--run_name", default="dpo_d1")
    ap.add_argument("--init_adapter", default=None, help="D2: SFT warm-up adapter 경로")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_pairs", type=int, default=200, help="미달 시 즉시 실패 (G2 사전 등록)")
    ap.add_argument("--pairs_file", default=None,
                    help="pair jsonl 경로 (기본 pairs_train.jsonl — ablation: pairs_train_{all,sem}.jsonl)")
    ap.add_argument("--g3_mode", choices=["abort", "warn"], default="abort",
                    help="문체-학습 가드: abort=중단(기본) · warn=마커만 쓰고 계속 "
                         "(DPO-all ablation은 실패 관측이 목적이므로 warn)")
    ap.add_argument("--loss_mode", choices=["fieldwise", "sum"], default="fieldwise",
                    help="fieldwise(기본)=field별 독립 -logσ(β·Δ_f) 가중합 — belief가 "
                         "action margin을 보상 못 함 (2026-07-24 G3 5연속 abort 대응). "
                         "sum=기존 가중합 margin 하나 (ablation 재현용)")
    ap.add_argument("--margin_clip", type=float, default=3.0,
                    help="목표 margin 상한: β·Δ_f를 이 값에서 clamp(detach 잔차) — 이미 "
                         "충분히 벌어진 pair는 gradient 0. 무한 등배로 인한 policy collapse "
                         "차단 (2026-07-24 dpo_d1_fix margin +70 폭주·생성 공백 대응). 0=끔")
    ap.add_argument("--probe_every", type=int, default=50,
                    help="N step마다 고정 probe 8샘플 생성 (trace 진화 관측). 0=끔")
    args = ap.parse_args()

    # 실행 중 체인에 CLI 수정 없이 값을 주입하는 통로 (runs/retro3/overrides.json).
    # 2026-07-22 사용자 확정: DPO 2 epoch — ablation 공정성 위해 3-arm(rule/all/sem) 공통 적용.
    ov_path = runs_root() / "overrides.json"
    if ov_path.is_file():
        ov = json.loads(ov_path.read_text())
        if "dpo_epochs" in ov:
            print(f"[override] epochs {args.epochs} -> {ov['dpo_epochs']}")
            args.epochs = int(ov["dpo_epochs"])

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    device = "cuda"
    data_dir = runs_root() / "data"

    ctx = {r["sample_id"]: r for r in read_jsonl(data_dir / "context_train.jsonl")}
    pairs = read_jsonl(Path(args.pairs_file) if args.pairs_file else data_dir / "pairs_train.jsonl")
    random.Random(args.seed).shuffle(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    if len(pairs) < args.min_pairs:
        write_marker("G2_PAIRS_INSUFFICIENT", {"n": len(pairs), "min": args.min_pairs})
        raise SystemExit(f"[G2] pairs {len(pairs)} < {args.min_pairs} — D1 폐기, D2로 (§12)")

    model, processor = vlm.load_model()
    model = lora_wrap(model)
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(Path(args.init_adapter) / "adapter_model.safetensors")
        set_peft_model_state_dict(model, sd)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = math.ceil(len(pairs) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    out_dir = Path(cfg["experiment"]["output_dir"]).parent / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    sw = StatusWriter(f"S6_{args.run_name}", total=len(pairs) * args.epochs)

    probe_recs = None
    if args.probe_every:
        from ego.step2_retrospection.train.probe_gen import build_probe_set, run_probe
        probe_recs = build_probe_set()

    step = n_seen = 0
    last_probe = -1
    ema = {"loss": None, "m_belief": None, "m_action": None, "m_reason": None}
    opt.zero_grad()
    for epoch in range(args.epochs):
        for pr in pairs:
            rec = ctx.get(pr["sample_id"])
            if rec is None:
                continue
            t0 = time.time()
            try:
                imgs = vlm.extract_frames(video_root, rec["video_uid"],
                                          rec["obs_start_sec"], rec["obs_end_sec"])
                # policy
                f_cw = fb_score(model, processor, rec, pr["chosen_fields"], imgs, device)
                f_rw = fb_score(model, processor, rec, pr["rejected_fields"], imgs, device)
                # ref (adapter off, no grad)
                with torch.no_grad(), model.disable_adapter():
                    f_cr = fb_score(model, processor, rec, pr["chosen_fields"], imgs, device)
                    f_rr = fb_score(model, processor, rec, pr["rejected_fields"], imgs, device)
                d_f = {f: (f_cw[f] - f_rw[f]) - (f_cr[f] - f_rr[f]) for f in C.FIELDS}

                def clipped(z):
                    # β·Δ가 목표 상한을 넘으면 초과분을 detach — logσ는 계속 계산하되
                    # gradient는 상한에서 멈춰 무한 margin 확대(→collapse)를 막는다.
                    if args.margin_clip and args.margin_clip > 0:
                        hi = args.margin_clip
                        return torch.clamp(z, max=hi) + (z - torch.clamp(z, max=hi)).detach()
                    return z

                if args.loss_mode == "fieldwise":
                    # field별 독립 sigmoid — 각 field margin이 스스로 올라야만 loss 감소.
                    # AO pair는 belief/reason Δ≡0 (동일 텍스트, 기울기 상쇄) → action 항만 학습.
                    loss = sum(FIELD_W[f] * -F.logsigmoid(clipped(args.beta * d_f[f])) for f in C.FIELDS)
                else:  # sum — 기존 가중합 margin (ablation 재현용)
                    loss = -F.logsigmoid(clipped(args.beta * sum(FIELD_W[f] * d_f[f] for f in C.FIELDS)))
                loss = loss * pr.get("weight", 1.0)
                (loss / args.accum).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt.zero_grad()
                append_jsonl(log_path, {"skip_oom": pr["sample_id"]})
                continue
            n_seen += 1
            if n_seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
            # field별 policy margin (chosen − rejected) — G3 가드레일 소재
            m = {"m_belief": float((f_cw["task_belief"] - f_rw["task_belief"]).detach()),
                 "m_action": float((f_cw["action"] - f_rw["action"]).detach()),
                 "m_reason": float((f_cw["reasoning"] - f_rw["reasoning"]).detach()),
                 "loss": float(loss.detach())}
            for k, v in m.items():
                ema[k] = v if ema[k] is None else 0.98 * ema[k] + 0.02 * v
            if n_seen % 10 == 0:
                append_jsonl(log_path, {"step": step, "seen": n_seen, **{k: round(v, 4) for k, v in m.items()},
                                        "sec": round(time.time() - t0, 2)})
            sw.update(done=n_seen, metrics={
                "step": step, "loss_ema": round(ema["loss"], 4),
                "belief_margin_ema": round(ema["m_belief"], 4),
                "action_margin_ema": round(ema["m_action"], 4)})
            if probe_recs and step != last_probe and (step % args.probe_every == 0 or n_seen == 1):
                last_probe = step
                pm = run_probe(model, processor, video_root, args.run_name, step, probe_recs)
                sw.update(metrics={"probe_acc": pm["probe_acc"]}, force=True)
            # G3: 문체-학습 가드 (충분히 본 뒤에만)
            # 2026-07-23 확장: action margin '정체'뿐 아니라 '음수 드리프트'(belief만 오르며
            # action이 −0.05 아래로 밀리는 경우)도 발동 — D1 초기 belief +2.7/action −0.31 관측 대응.
            if n_seen >= 400 and ema["m_belief"] > 0.25 and (
                    abs(ema["m_action"]) < 0.01 or ema["m_action"] < -0.05):
                if args.g3_mode == "abort":
                    write_marker("G3_STYLE_ABORT", {"run": args.run_name, "seen": n_seen,
                                                    "belief_margin": ema["m_belief"],
                                                    "action_margin": ema["m_action"]})
                    model.save_pretrained(out_dir / "adapter_aborted")
                    sw.fail("G3 style-learning guard tripped")
                    raise SystemExit("[G3] belief margin만 상승 — 문체 학습 신호로 중단")
                if not (runs_root() / "markers" / f"G3_STYLE_WARN_{args.run_name}").exists():
                    write_marker(f"G3_STYLE_WARN_{args.run_name}",
                                 {"seen": n_seen, "belief_margin": ema["m_belief"],
                                  "action_margin": ema["m_action"]})

    model.save_pretrained(out_dir / "adapter")
    sw.finish(metrics={"steps": step, **{k: round(v, 4) for k, v in ema.items() if v is not None}})
    write_marker(f"S6_{args.run_name.upper()}_DONE", {"steps": step})
    print(f"[S6:{args.run_name}] steps={step} ema={json.dumps({k: round(v, 4) for k, v in ema.items() if v is not None})}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
