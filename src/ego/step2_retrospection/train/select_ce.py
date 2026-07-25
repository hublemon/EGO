"""Stage-1 Predictive-Boundary Selection — candidate-normalized selection CE with τ.

v2 방법론 §1 Stage-1 (`2026-07-24_ce_sft_methodology_v2_handoff.md`):
  s_θ(a;c) = (1/|a|) Σ log π_θ(a_j | c, a_<j)           # length-norm 후보 점수
  p_θ(a|c,D) = softmax(s_θ(a)/τ) over D                  # 후보 집합 정규화
  L_sel = − log p_θ(a_GT | c, D)

**핵심 (사용자 2026-07-24 지시)**: 후보 채점은 반드시 **video-grounded** —
관측창 8프레임(video prefix) + action history를 VLM이 본다. (기존
intervention.py / r2_candidate_loss 는 tok()만 써 vision-blind 였음 → 여기서 교정.)
teacher reasoning/belief prefix 는 미사용 (순수 action selection).

Arm 스위치 (v2 §3 baseline — WM 경계 vs 그냥 GT CE 격리):
  wm_cand     WM Top-K 후보 + selection CE        (기본, θ_CE)
  cand_free   후보 無 + 순수 GT-span CE           (GT CE 자체 효과)
  random_cand GT+무작위 K-1 후보 + selection CE   (multiple-choice 효과)
  no_video    WM Top-K, 프레임 제거                (영상 사용 검증)
  no_history  WM Top-K, history 제거               (task progression 검증)

CE-replay (조합 §2①)용으로 selection_ce_step() 을 sft_v2 가 재사용한다.

출력: outputs/step2_retrospection/cesft_v2/<run_name>/adapter (LoRA), train_log.jsonl
사용:
  RETRO3_RUNS=runs/cesft_v2 PYTHONPATH=src python -m \
    ego.step2_retrospection.train.select_ce --run_name theta_ce --arm wm_cand \
    --tau 1.0 --epochs 1 [--init_adapter PATH] [--max_steps N]
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

# arm 프리셋 → (candidate_source, use_video, use_history, loss_type)
ARMS = {
    "wm_cand":     ("wm",     True,  True,  "sel"),
    "cand_free":   ("free",   True,  True,  "gt"),
    "random_cand": ("random", True,  True,  "sel"),
    "no_video":    ("wm",     False, True,  "sel"),
    "no_history":  ("wm",     True,  False, "sel"),
}

SYS_NOCAND = (
    "You are an egocentric activity assistant. You see frames from the last 8 seconds of a "
    "first-person video and a list of actions the person already COMPLETED. Each action is "
    "'verb noun'. Name the single next action the person does next "
    f"({vlm.NEXT_GAP_TEXT}).\n<action>\nverb noun\n</action>"
)


def lora_wrap(model, r: int = 16):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=r * 2, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def build_action_pool(rows: list[dict]) -> list[str]:
    """random_cand distractor 풀 — 데이터 전체 후보/GT의 'verb noun' union (포맷 독립)."""
    pool = set()
    for r in rows:
        pool.update(r.get("candidates", []))
        pool.add(f"{r['gt_verb']} {r['gt_noun']}")
    return sorted(pool)


def build_messages_arm(rec: dict, imgs, cand_source: str, candidates: list[str], use_history: bool):
    """arm별 프롬프트 구성. cand_free 는 후보 블록·후보 시스템문구 제거."""
    content = [{"type": "image", "image": im} for im in imgs]
    hist = vlm.fmt_history(rec) if use_history else "(history withheld)"
    if cand_source == "free":
        text = (f"Completed actions so far (oldest to newest):\n{hist}\n\n"
                "What is the next action? Follow the required format.")
        sys = SYS_NOCAND
    else:
        text = (f"Completed actions so far (oldest to newest):\n{hist}\n\n"
                f"Candidate next actions (shuffled):\n{vlm.fmt_candidates(candidates)}\n\n"
                "Which candidate is the next action? Follow the required format.")
        sys = vlm.SYSTEM_PROMPT
    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": [{"type": "text", "text": sys}]},
            {"role": "user", "content": content}]


def _candidate_logps(model, processor, messages, cand_actions: list[str], device):
    """K개 후보 action-span 의 length-norm logp (grad 유지). ONE batched forward
    (K행 = 동일 frames+history prompt, 다른 후보). vision-grounded."""
    tok = processor.tokenizer
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [c["image"] for m in messages for c in m["content"]
              if isinstance(c, dict) and c.get("type") == "image"]
    K = len(cand_actions)
    proc = processor(text=[text] * K, images=[images] * K if images else None,
                     return_tensors="pt").to(device)
    P = proc["input_ids"].shape[1]
    act_prefix = tok("<action>\n", add_special_tokens=False)["input_ids"]
    cand_tok = [tok(c, add_special_tokens=False)["input_ids"] for c in cand_actions]
    comp = [act_prefix + ct for ct in cand_tok]
    maxc = max(len(x) for x in comp)
    comp_ids = torch.full((K, maxc), tok.eos_token_id, dtype=torch.long, device=device)
    comp_mask = torch.zeros((K, maxc), dtype=torch.long, device=device)
    for i, x in enumerate(comp):
        comp_ids[i, :len(x)] = torch.tensor(x, device=device)
        comp_mask[i, :len(x)] = 1
    new = {}
    for k, v in proc.items():
        if k == "input_ids":
            new[k] = torch.cat([v, comp_ids], dim=1)
        elif k == "attention_mask":
            new[k] = torch.cat([v, comp_mask], dim=1)
        elif isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == P:
            pad = torch.zeros((v.shape[0], maxc, *v.shape[2:]), dtype=v.dtype, device=device)
            new[k] = torch.cat([v, pad], dim=1)
        else:
            new[k] = v
    logits = model(**new).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = new["input_ids"][:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (K, P+maxc-1)
    s0 = P + len(act_prefix) - 1
    scores = torch.stack([tok_lp[i, s0:s0 + len(ct)].mean() for i, ct in enumerate(cand_tok)])
    return scores  # (K,) length-norm mean logp


def selection_ce_step(model, processor, rec, imgs, device, tau: float, arm: str,
                      registry: list[str], seed_rng: random.Random):
    """한 샘플의 selection-CE loss (또는 cand_free 의 GT-span CE).
    imgs 는 호출부가 prefetch 로 공급 (no_video arm 은 [])."""
    cand_source, use_video, use_history, loss_type = ARMS[arm]
    gt_action = f"{rec['gt_verb']} {rec['gt_noun']}"
    if cand_source == "wm":
        candidates = list(rec["candidates"])
        gt_idx = candidates.index(gt_action)
    elif cand_source == "random":
        pool = [a for a in registry if a != gt_action] or \
               [c for c in rec["candidates"] if c != gt_action]
        distract = seed_rng.sample(pool, min(len(rec["candidates"]) - 1, len(pool)))
        candidates = distract + [gt_action]
        seed_rng.shuffle(candidates)
        gt_idx = candidates.index(gt_action)
    else:  # free — 후보 미제시, GT-span CE
        candidates = [gt_action]
        gt_idx = 0
    messages = build_messages_arm(rec, imgs, cand_source, candidates, use_history)
    scores = _candidate_logps(model, processor, messages, candidates, device)
    if loss_type == "gt":
        return -scores[gt_idx], {"sel_ce": round(float((-scores[gt_idx]).detach()), 4)}
    logp = torch.log_softmax(scores / tau, dim=0)
    loss = -logp[gt_idx]
    return loss, {"sel_ce": round(float(loss.detach()), 4),
                  "p_gt": round(float(logp[gt_idx].exp().detach()), 4)}


_VIDEO_ROOT = None


def rec_video_root() -> Path:
    return _VIDEO_ROOT


def main() -> None:
    global _VIDEO_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml")
    ap.add_argument("--run_name", default="theta_ce")
    ap.add_argument("--arm", default="wm_cand", choices=list(ARMS))
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--init_adapter", default=None, help="warm-start (C-stack 등)")
    ap.add_argument("--max_steps", type=int, default=None, help="equal-budget 통제 (C-ctrl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_every", type=int, default=100)
    ap.add_argument("--subset_file", default="train_subset.json",
                    help="학습 풀 제한 파일 (data/ 기준). 'none' = full covered 풀 전량")
    ap.add_argument("--ckpt_every", type=int, default=100,
                    help="opt.step 주기 중간 체크포인트 (0=끄기)")
    ap.add_argument("--resume", default="auto", choices=["auto", "off"],
                    help="auto: ckpt/ 가 있으면 그 지점부터 이어 학습")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="gradient checkpointing 끄기 — VRAM ↔ 속도 트레이드(수치 동일)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    _VIDEO_ROOT = Path(cfg["shared_assets"]["video_root"])
    device = "cuda"
    data_dir = runs_root() / "data"

    # covered 학습 샘플 = (subset ∩) gt_rank<=10. --subset_file none 이면 full 풀 전량.
    sub_ids = None
    if args.subset_file and args.subset_file.lower() != "none":
        sp = data_dir / args.subset_file
        if sp.is_file():
            subset = json.loads(sp.read_text())
            if isinstance(subset, dict) and "sample_ids" in subset:
                sub_ids = set(subset["sample_ids"])
    rows = [r for r in read_jsonl(data_dir / "context_train.jsonl")
            if r.get("gt_rank", 99) <= 10 and (sub_ids is None or r["sample_id"] in sub_ids)]
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    registry = build_action_pool(rows) if args.arm == "random_cand" else []

    model, processor = vlm.load_model()
    model = lora_wrap(model)
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(str(Path(args.init_adapter) / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, sd)
        print(f"[select_ce] warm-start from {args.init_adapter}")
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = math.ceil(len(rows) * args.epochs / args.accum)
    if args.max_steps:
        total = min(total, args.max_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total))

    out_dir = Path(cfg["experiment"]["output_dir"]).parent / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    sw = StatusWriter(f"S_CE_{args.run_name}", total=(args.max_steps * args.accum) if args.max_steps else len(rows) * args.epochs)

    probe_recs = None
    if args.probe_every:
        try:
            from ego.step2_retrospection.train.probe_gen import build_probe_set
            probe_recs = build_probe_set()
        except Exception as e:  # probe 실패는 학습 비차단
            print(f"[select_ce] probe disabled: {e}")

    use_video = ARMS[args.arm][1]

    def grouped(seed: int):
        """비디오별로 묶어 정렬 — 같은 영상 연속 접근 → decord 리더 캐시 히트
        (500MB 영상을 샘플마다 재디코드하던 병목 제거). 영상 순서만 셔플."""
        by_v: dict[str, list] = {}
        for r in rows:
            by_v.setdefault(r["video_uid"], []).append(r)
        vids = list(by_v)
        random.Random(seed).shuffle(vids)
        return [r for v in vids for r in by_v[v]]

    def sample_stream(epoch: int, start_pos: int = 0):
        # start_pos 는 프레임 로딩 **전에** 잘라낸다 — resume 시 이미 본 구간을 다시 디코드하지 않게.
        ordered = grouped(args.seed + epoch)[start_pos:]
        if use_video:  # prefetch: 다음 chunk 디코드(CPU)가 현재 chunk 연산(GPU)과 겹침
            for chunk, frames in vlm.prefetch_chunks(_VIDEO_ROOT, ordered, 8):
                for rec, (imgs, err) in zip(chunk, frames):
                    yield rec, (imgs if err is None else None)
        else:
            for rec in ordered:
                yield rec, []

    step = n_seen = 0
    last_probe = -1
    rng = random.Random(args.seed)
    ema = None
    # ── resume: LoRA·optimizer·scheduler·rng·스트림 위치 복원 (§ckpt.py) ──
    start_epoch, start_pos = 0, 0
    if args.resume == "auto":
        st = ckpt.load_ckpt(out_dir, model, opt, sched)
        if st:
            start_epoch, start_pos = st.get("epoch", 0), st.get("stream_pos", 0)
            step, n_seen, ema = st.get("step", 0), st.get("n_seen", 0), st.get("ema")
            ckpt.restore_rng(st, rng)
            sw.update(done=n_seen, force=True)
    opt.zero_grad()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        pos = (start_pos if epoch == start_epoch else 0) - 1
        for rec, imgs in sample_stream(epoch, pos + 1):
            pos += 1
            if imgs is None:  # decode 실패
                append_jsonl(log_path, {"skip_decode": rec["sample_id"]})
                continue
            t0 = time.time()
            try:
                loss, m = selection_ce_step(model, processor, rec, imgs, device, args.tau,
                                            args.arm, registry, rng)
                (loss / args.accum).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt.zero_grad()
                append_jsonl(log_path, {"skip_oom": rec["sample_id"]})
                continue
            except Exception as e:
                # 2026-07-24: 개별 샘플의 vision reshape 등(예: Qwen3-VL patch 타일링 실패,
                #   size 69395200) 하나가 4h 학습을 죽이지 않도록 스킵. decode/OOM 스킵과 동일 정책.
                torch.cuda.empty_cache()
                opt.zero_grad()
                append_jsonl(log_path, {"skip_error": rec["sample_id"], "err": str(e)[:200]})
                continue
            n_seen += 1
            if n_seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if args.ckpt_every and step % args.ckpt_every == 0:
                    ckpt.save_ckpt(out_dir, model, opt, sched,
                                   {"epoch": epoch, "stream_pos": pos + 1, "n_seen": n_seen,
                                    "step": step, "ema": ema, "rng": rng.getstate(),
                                    "arm": args.arm, "run_name": args.run_name})
            lv = float(loss.detach())
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            if n_seen % 10 == 0:
                append_jsonl(log_path, {"step": step, "seen": n_seen, "epoch": epoch,
                                        "loss": round(lv, 4), **m,
                                        "sec": round(time.time() - t0, 2)})
            sw.update(done=n_seen, metrics={"step": step, "arm": args.arm,
                                            "loss_ema": round(ema, 4), "sel_ce": m.get("sel_ce"),
                                            "lr": sched.get_last_lr()[0]})
            if probe_recs and step != last_probe and (step % args.probe_every == 0 or n_seen == 1):
                last_probe = step
                try:
                    from ego.step2_retrospection.train.probe_gen import run_probe
                    pm = run_probe(model, processor, _VIDEO_ROOT, args.run_name, step, probe_recs)
                    sw.update(metrics={"probe_acc": pm["probe_acc"]}, force=True)
                except Exception:
                    pass
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

    model.save_pretrained(out_dir / "adapter")
    ckpt.clear_ckpt(out_dir)   # 완료 — 다음 런이 낡은 resume 상태를 물지 않게
    sw.finish(metrics={"steps": step, "final_loss_ema": round(ema, 4) if ema else None})
    write_marker(f"S_CE_{args.run_name.upper()}_DONE", {"steps": step, "arm": args.arm})
    print(f"[select_ce:{args.run_name}] arm={args.arm} steps={step} loss_ema={ema}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
