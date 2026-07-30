"""Stage-2 Projected Retrospection SFT — CE-replay 앵커드 (조합 §2① / v2 §1 Stage-2).

θ_CE 초기화(--init_adapter) 위에 projected-trace SFT 를 하되, candidate-CE 샘플을
**배치 혼합 ρ** 로 인터리브해 CE 판별 채널을 앵커한다 (순차 적층의 덮어쓰기 방지 —
조합 handoff §0/§2①). loss field 가중은 belief 1.0 / reasoning 0.5 / action 0.25
(common.FIELD_WEIGHTS 기본값 그대로).

혼합 방식 (조합 §2 방식 A — 합산손실 아님, gradient 충돌 회피):
  --ce_replay_rho ρ : 매 micro-step 확률 ρ 로 CE 샘플, (1−ρ) 로 SFT 샘플.
  --alternate       : 확률 대신 결정적 교대 (ALRIGHT, 조합 §2③). 매 round(1/ρ) 마다 CE 1회.

CE 스트림은 select_ce.selection_ce_step (wm_cand, video-grounded) 재사용.

출력: outputs/step2_retrospection/cesft_v2/<run_name>/adapter, train_log.jsonl
사용:
  RETRO3_RUNS=runs/cesft_v2 PYTHONPATH=src python -m \
    ego.step2_retrospection.train.sft_v2 --run_name sft_r15 \
    --init_adapter outputs/step2_retrospection/cesft_v2/theta_ce/adapter \
    --ce_replay_rho 0.15 --epochs 1
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
from ego.step2_retrospection.runtime import (StatusWriter, append_jsonl, read_jsonl,
                                             runs_root, write_marker)
from ego.step2_retrospection.train import ckpt
from ego.step2_retrospection.train import common as C
from ego.step2_retrospection.train import select_ce as SCE


def lora_wrap(model, r: int = 16):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=r * 2, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def sft_step(model, processor, rec, rec_c, video_root, device):
    """projected-trace span-normalized SFT loss (sft_r1 과 동일 경로)."""
    imgs = vlm.extract_frames(video_root, rec["video_uid"],
                              rec["obs_start_sec"], rec["obs_end_sec"], rec=rec)
    p_in = C.build_prompt_inputs(processor, vlm.build_messages(rec, imgs), device)
    comp_ids, field_of = C.encode_completion(processor.tokenizer, {
        "reasoning": rec_c["reasoning"], "task_belief": rec_c["task_belief"],
        "action": rec_c["action"]})
    inputs, p_len = C.cat_completion(p_in, comp_ids, device)
    return C.span_ce_loss(model, inputs, p_len, field_of)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml")
    ap.add_argument("--run_name", default="sft_r15")
    ap.add_argument("--init_adapter", default=None, help="θ_CE warm-start (조합 reference-anchor)")
    ap.add_argument("--ce_replay_rho", type=float, default=0.15, help="CE 혼합 비율 (0=순차 적층)")
    ap.add_argument("--alternate", action="store_true", help="확률 대신 결정적 교대(ALRIGHT)")
    ap.add_argument("--ce_tau", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_every", type=int, default=100)
    ap.add_argument("--subset_file", default="train_subset.json",
                    help="CE-replay 풀 제한 파일 (data/ 기준). 'none' = full covered 풀 전량")
    ap.add_argument("--ckpt_every", type=int, default=100,
                    help="opt.step 주기 중간 체크포인트 (0=끄기)")
    ap.add_argument("--resume", default="auto", choices=["auto", "off"],
                    help="auto: ckpt/ 가 있으면 그 지점부터 이어 학습")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    SCE._VIDEO_ROOT = video_root  # select_ce.selection_ce_step 가 참조
    device = "cuda"
    data_dir = runs_root() / "data"

    ctx = {r["sample_id"]: r for r in read_jsonl(data_dir / "context_train.jsonl")}
    chosen = [r for r in read_jsonl(data_dir / "chosen_train.jsonl") if r.get("gate") == "pass"]
    random.Random(args.seed).shuffle(chosen)
    if args.limit:
        chosen = chosen[: args.limit]

    # CE-replay 풀 = covered train (θ_CE 와 동일 분포). --subset_file none 이면 full 풀 전량.
    sub_ids: set[str] = set()
    if args.subset_file and args.subset_file.lower() != "none":
        sp = data_dir / args.subset_file
        if sp.is_file():
            sub_ids = set(json.loads(sp.read_text()).get("sample_ids", []))
    ce_pool = [r for r in ctx.values()
               if r.get("gt_rank", 99) <= 10 and (not sub_ids or r["sample_id"] in sub_ids)]

    model, processor = vlm.load_model()
    model = lora_wrap(model)
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(str(Path(args.init_adapter) / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, sd)
        print(f"[sft_v2] warm-start from {args.init_adapter}")
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = math.ceil(len(chosen) * args.epochs / args.accum / max(1e-6, (1 - args.ce_replay_rho)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total))

    out_dir = Path(cfg["experiment"]["output_dir"]).parent / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    sw = StatusWriter(f"S6_{args.run_name}", total=len(chosen) * args.epochs)

    probe_recs = None
    if args.probe_every:
        try:
            from ego.step2_retrospection.train.probe_gen import build_probe_set
            probe_recs = build_probe_set()
        except Exception as e:
            print(f"[sft_v2] probe disabled: {e}")

    rng = random.Random(args.seed)
    step = n_sft = micro = 0
    last_probe = -1
    ema = {"loss": None, "reasoning": None, "task_belief": None, "action": None, "sel_ce": None}
    opt.zero_grad()
    # 비디오별 그룹 정렬 — decord 리더 캐시 히트 (샘플마다 500MB 재디코드 방지)
    def vgroup(rows):
        by_v: dict[str, list] = {}
        for r in rows:
            by_v.setdefault(ctx.get(r["sample_id"], {}).get("video_uid", "?"), []).append(r)
        vids = list(by_v); random.Random(args.seed).shuffle(vids)
        return [r for v in vids for r in by_v[v]]
    sft_queue = vgroup(list(chosen)) * args.epochs
    period = max(2, round(1 / max(1e-6, args.ce_replay_rho))) if args.ce_replay_rho > 0 else 0

    # ── resume: LoRA·optimizer·scheduler·rng·큐 위치 복원 (§ckpt.py) ──
    # 큐는 pop() 으로 뒤에서 소비되므로, 이미 소비한 n_sft 개를 잘라내면 정확히 이어붙는다.
    if args.resume == "auto":
        st = ckpt.load_ckpt(out_dir, model, opt, sched)
        if st:
            step, n_sft, micro = st.get("step", 0), st.get("n_sft", 0), st.get("micro", 0)
            ema.update({k: v for k, v in (st.get("ema") or {}).items() if k in ema})
            ckpt.restore_rng(st, rng)
            if n_sft:
                sft_queue = sft_queue[: max(0, len(sft_queue) - n_sft)]
            sw.update(done=n_sft, force=True)

    while sft_queue:
        # 이 micro-step 이 CE replay 인가?
        do_ce = False
        if args.ce_replay_rho > 0:
            do_ce = (micro % period == period - 1) if args.alternate else (rng.random() < args.ce_replay_rho)
        micro += 1
        t0 = time.time()
        try:
            if do_ce and ce_pool:
                rec = rng.choice(ce_pool)
                ce_imgs = vlm.extract_frames(video_root, rec["video_uid"],
                                             rec["obs_start_sec"], rec["obs_end_sec"], rec=rec)
                loss, m = SCE.selection_ce_step(model, processor, rec, ce_imgs, device, args.ce_tau,
                                                "wm_cand", [], rng)
                per = {"sel_ce": m.get("sel_ce")}
                tag = "ce"
            else:
                rec_c = sft_queue.pop()
                rec = ctx.get(rec_c["sample_id"])
                if rec is None:
                    continue
                loss, per = sft_step(model, processor, rec, rec_c, video_root, device)
                n_sft += 1
                tag = "sft"
            (loss / args.accum).backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            opt.zero_grad()
            append_jsonl(log_path, {"skip_oom": rec.get("sample_id") if 'rec' in dir() else "?"})
            continue
        if micro % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if args.ckpt_every and step % args.ckpt_every == 0:
                ckpt.save_ckpt(out_dir, model, opt, sched,
                               {"step": step, "n_sft": n_sft, "micro": micro,
                                "ema": dict(ema), "rng": rng.getstate(),
                                "rho": args.ce_replay_rho, "run_name": args.run_name})
        lv = float(loss.detach())
        for k, v in [("loss", lv), *per.items()]:
            if v is not None:
                ema[k] = v if ema[k] is None else 0.98 * ema[k] + 0.02 * v
        if micro % 10 == 0:
            append_jsonl(log_path, {"step": step, "seen": n_sft, "micro": micro, "tag": tag,
                                    "loss": round(lv, 4), **{k: v for k, v in per.items() if v is not None},
                                    "sec": round(time.time() - t0, 2)})
        sw.update(done=n_sft, metrics={
            "step": step, "rho": args.ce_replay_rho,
            "loss_ema": round(ema["loss"], 4) if ema["loss"] else None,
            "belief_ce_ema": round(ema["task_belief"], 4) if ema["task_belief"] else None,
            "action_ce_ema": round(ema["action"], 4) if ema["action"] else None,
            "sel_ce_ema": round(ema["sel_ce"], 4) if ema["sel_ce"] else None,
            "lr": sched.get_last_lr()[0]})
        if probe_recs and step != last_probe and (step % args.probe_every == 0 or micro == 1):
            last_probe = step
            try:
                from ego.step2_retrospection.train.probe_gen import run_probe
                pm = run_probe(model, processor, video_root, args.run_name, step, probe_recs)
                sw.update(metrics={"probe_acc": pm["probe_acc"]}, force=True)
            except Exception:
                pass

    model.save_pretrained(out_dir / "adapter")
    ckpt.clear_ckpt(out_dir)   # 완료 — 다음 런이 낡은 resume 상태를 물지 않게
    sw.finish(metrics={"steps": step, "n_sft": n_sft, "micro": micro,
                       "final_loss_ema": round(ema["loss"], 4) if ema["loss"] else None})
    write_marker(f"S6_{args.run_name.upper()}_DONE",
                 {"steps": step, "rho": args.ce_replay_rho, "n_sft": n_sft})
    print(f"[sft_v2:{args.run_name}] rho={args.ce_replay_rho} steps={step} n_sft={n_sft}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
