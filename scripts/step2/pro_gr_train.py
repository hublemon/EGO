#!/usr/bin/env python3
"""pro_gr_train.py — F0-GR 진단: action-only 생성 기반 REINFORCE + EMA 기준선.

설계 결정(2026-07-20): 후보 문자열 teacher-forcing 스코어링(exact-CE)은 "생성으로 행동을
선택하는 에이전트" 전제와 train-test 불일치라 기각. 행동 선택은 **생성**으로 유지하되,
GRPO 의 그룹-내 기준선(형제 롤아웃 평균 — 짧은 출력에선 전원 동일해 신호 소멸)을
**보상 이동평균(EMA) 기준선**으로 교체한다:

  프롬프트당 롤아웃 1개 생성(T 샘플링) → r = 1[action==GT] (무효 0)
  advantage = r − EMA(r)  →  loss = −advantage · mean_logp(생성 토큰)

그룹 다양성이 없어도 r ≠ EMA 면 gradient 가 흐른다. 결정론적 정책이어도 학습 가능.
진단 지위: GT 사용 → 방법 아님 (F0-G 와 동일). oracle-subset 학습.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import deque
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from ego.step2_vlm_alignment import train_grpo_action as T  # noqa: E402
from ego.common.run_provenance import write_run_config  # noqa: E402


def _action_token_start(tokenizer, comp_ids, tag: str = "<action>"):
    """완성 토큰열에서 `<action>` 태그가 **완성되는** 첫 토큰 index+1 을 이진탐색으로 찾는다.

    생성 id 를 그대로 쓰므로 재토큰화 불일치가 없다. 반환 k 는 'k 이후가 action 페이로드'.
    태그가 없으면 None.
    """
    ids = comp_ids.tolist()
    if tag not in tokenizer.decode(ids, skip_special_tokens=True):
        return None
    lo, hi = 1, len(ids)                      # decode(ids[:hi]) 에는 태그가 있다 (위에서 확인)
    while lo < hi:
        mid = (lo + hi) // 2
        if tag in tokenizer.decode(ids[:mid], skip_special_tokens=True):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _token_index_at_char(tokenizer, ids, char_pos: int) -> int:
    """decode 길이가 `char_pos` 를 처음 넘어서는 토큰(=그 문자를 포함한 토큰)의 index.

    prefix decode 길이는 토큰 수에 대해 단조 비감소이므로 `_action_token_start` 와 같은
    이진탐색이 성립한다. 반환값은 '그 문자가 시작되는 토큰' — 페이로드 끝(exclusive)으로 쓴다.
    """
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=True)) > char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo - 1


def _belief_token_span(tokenizer, comp_ids,
                       open_tag: str = "<task_belief>", close_tag: str = "</task_belief>"):
    """`<task_belief>…</task_belief>` 페이로드의 [start, end) 토큰 구간.

    start 는 `_action_token_start` 와 동일 규약(열림 태그가 완성되는 첫 index+1),
    end 는 닫힘 태그가 시작되는 토큰 index. 닫힘 태그가 없으면(절단) 완성부 끝까지.
    태그가 아예 없으면 None.
    """
    k = _action_token_start(tokenizer, comp_ids, tag=open_tag)
    if k is None:
        return None
    ids = comp_ids.tolist()
    p = tokenizer.decode(ids, skip_special_tokens=True).find(close_tag)
    if p < 0:
        return k, len(ids)
    return k, max(k, _token_index_at_char(tokenizer, ids, p))


# ── 개선 2 / P3: belief-swap consistency loss ────────────────────────────────
TAG_R = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
TAG_B = re.compile(r"<task_belief>(.*?)</task_belief>", re.DOTALL)


def cand_completion(v: str, n: str) -> str:
    """pro_gx_train.cand_completion 과 **문자 단위로 동일**해야 한다 (스코어 대조 보존)."""
    return f'<action>{{"verb": "{v}", "noun": "{n}"}}</action>'


def build_candidate_batch(tokenizer, base_ids, cands, pad_id):
    """base_ids 뒤에 후보별 완성 토큰을 이어붙인 **오른쪽 패딩** 배치. → (ids, attn, mask)

    ⚠ pro_gx_train:104-115 처럼 `base_text + completion` 문자열을 다시 토크나이즈하면 안 된다.
    두 가지가 동시에 깨진다:
      (1) 경계 병합 — base 가 공백으로 끝나면 그 공백과 '<' 가 한 토큰으로 합쳐져
          standalone base_len 이 실제 base 점유 토큰 수와 달라진다. 실측으로 completion
          첫 글자 '<' 가 마스크에서 누락됐다 (tests/step2/test_cons_mask.py 가 잡음).
      (2) 이 파일의 processor 는 배치 생성 때문에 left padding 이라 base 시작 위치가
          행마다 다르다 — 원본의 '앞 base_len 을 0' 마스킹이 성립하지 않는다.
    그래서 id 를 직접 이어붙여 경계를 확정한다 (RL 경로가 이미 쓰는 방식 — 아래 `full` 참조).
    스코어링 forward 는 생성이 아니므로 오른쪽 패딩이 안전하고 마스크가 자명해진다.
    """
    dev = base_ids.device
    base_len = base_ids.numel()
    comps = [torch.tensor(tokenizer(cand_completion(v, n), add_special_tokens=False)["input_ids"],
                          dtype=base_ids.dtype, device=dev) for v, n in cands]
    R, K = len(comps), max(c.numel() for c in comps)
    L = base_len + K
    ids = torch.full((R, L), pad_id, dtype=base_ids.dtype, device=dev)
    attn = torch.zeros((R, L), dtype=torch.long, device=dev)
    mask = torch.zeros((R, L - 1), dtype=torch.float32, device=dev)
    for r, c in enumerate(comps):
        k = c.numel()
        ids[r, :base_len] = base_ids
        ids[r, base_len:base_len + k] = c
        attn[r, :base_len + k] = 1
        # 위치 p 토큰의 logprob 은 shift 후 index p-1 에 온다
        mask[r, base_len - 1: base_len - 1 + k] = 1.0
    return ids, attn, mask, base_len, K


def score_candidates(model, processor, base_enc, cands):
    """base_enc(프롬프트+prefix, 이미지 포함) 뒤 각 후보의 sum-logp. → (len(cands),)"""
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids, attn, mask, base_len, K = build_candidate_batch(
        tok, base_enc["input_ids"][0], cands, pad_id)
    R = ids.shape[0]
    extra = {}
    if "mm_token_type_ids" in base_enc:      # Qwen3-VL M-RoPE: completion 은 텍스트(0)
        mm = base_enc["mm_token_type_ids"][0]
        mmb = torch.zeros(ids.shape, dtype=mm.dtype, device=mm.device)
        mmb[:, : mm.numel()] = mm
        extra["mm_token_type_ids"] = mmb
    pv, grid = base_enc.get("pixel_values"), base_enc.get("image_grid_thw")
    if pv is not None:                        # 같은 이미지를 후보 수만큼 복제
        pv = pv.repeat(R, *([1] * (pv.dim() - 1)))
        grid = grid.repeat(R, 1)
    logits = model(input_ids=ids, attention_mask=attn,
                   pixel_values=pv, image_grid_thw=grid, **extra).logits
    # ⚠ `logits[:, :-1].float()` 를 통째로 만들면 OOM 이 난다. 실측: reasoning 이 길어져
    # L≈1500 이 되면 (R=6, L, V≈151k) fp32 사본이 5GB 를 요구하고 seen≈1800 에서 죽었다.
    # 우리가 필요한 위치는 completion 구간 K(≈21) 뿐이므로 **softmax 전에 잘라낸다** (약 70배 절감).
    s = base_len - 1
    logp = torch.log_softmax(logits[:, s:s + K].float(), dim=-1)
    tgt = ids[:, 1:][:, s:s + K]
    tokl = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (tokl * mask[:, s:s + K]).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=7000)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--full_trace", action="store_true",
                    help="3태그 full-trace 출력 유지 (F0-WE 확정 run 용)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_gen", type=int, default=1,
                    help="생성 배치 크기 (autoregressive 생성이 지배 비용)")
    ap.add_argument("--ema", type=float, default=0.99, help="기준선 EMA 모멘텀")
    ap.add_argument("--credit", choices=["all", "action", "belief"], default="all",
                    help="advantage 를 걸 토큰 범위. all=완성부 전체(기존) · "
                         "action=<action> 이후만 (credit 국소화 — ③ 인과 겨냥) · "
                         "belief=<task_belief> 페이로드만 (고엔트로피 자유텍스트 → gradient 생존)")
    ap.add_argument("--credit-reduction", "--credit_reduction",
                    choices=["mean", "sum"], default="mean",
                    help="span logp 집계. mean=기존(span 길이로 나눔) · "
                         "sum=길이 정규화 없음. credit=action 의 mean 은 거의 결정적인 JSON "
                         "토큰 몇 개를 다시 나눠 gradient 를 소멸시킨다(실측 mean|loss| "
                         "0.004160→0.000092, 45.4× 축소) — 개선 3.")
    ap.add_argument("--reward", choices=["gt", "wm"], default="gt",
                    help="gt=바이너리(진단/WE) · wm=후보 정규화 likelihood (W-EMA, GT-free)")
    # ── 개선 2 / P3: belief-swap consistency loss ──
    ap.add_argument("--cons_weight", type=float, default=0.0,
                    help="belief-swap consistency loss 가중치. 0=off(기존 동작 보존). "
                         "belief 를 다른 샘플 것으로 바꿨을 때 action 이 따라 바뀌지 않으면 "
                         "페널티 — ③ 를 간접 신호가 아니라 **직접** 최적화하는 유일한 항")
    ap.add_argument("--cons_margin", type=float, default=0.5,
                    help="hinge max(0, log q(a_orig) − log q(a_swap) + m). 분리가 m 만큼 "
                         "이뤄지면 gradient 가 멈춘다. 0=핸드오프 원식(무한정 밀어냄) — "
                         "**실측으로 폭주했다**: cons_loss 가 +10.7 → −13.8 로 부호를 넘어 "
                         "계속 밀렸고 reward_ma 가 0.34 → 0.115 로 붕괴했다. 0 은 진단용으로만.")
    ap.add_argument("--reward_floor", type=float, default=0.0,
                    help="reward_ma 가 이 값 아래로 연속 --reward_floor_patience 회 떨어지면 "
                         "체크포인트를 남기고 중단한다. 0=끔. 300샘플 스모크는 1,200샘플에서 "
                         "나타나는 붕괴를 볼 수 없으므로 본실행에는 자체 가드가 필요하다.")
    ap.add_argument("--reward_floor_patience", type=int, default=2)
    ap.add_argument("--cons_buffer", type=int, default=64,
                    help="swap belief 링버퍼 크기. batch_gen=1 이라 in-batch derangement 가 "
                         "불가능해 최근 생성 belief 를 재사용한다")
    ap.add_argument("--cons_warmup", type=int, default=16,
                    help="링버퍼가 이만큼 차기 전에는 consistency 항을 걸지 않는다")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    # 개선 0: 실행 출처(argv·git SHA·입력 데이터 지문)를 산출물로 남긴다 — 모델 로드 전에 기록
    write_run_config(out, args, data_paths=[args.train_jsonl], extra={"runner": "pro_gr_train"})
    T.ACTION_ONLY = not args.full_trace   # F0-WE: full-trace 유지

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_name, padding_side="left", use_fast=True,
        min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": args.device})
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    rows = [json.loads(l) for l in open(args.train_jsonl, encoding="utf-8") if l.strip()]
    kept = []
    for ex in rows:
        if not Path(ex["image_path"]).exists():
            continue
        if args.reward == "wm":   # GT-free: GT 필터 금지 — 전 샘플 유지
            kept.append(ex); continue
        # gt 계열: oracle-subset (gt_only/gt_action_only 와 동일 정책)
        c5 = [(str(a.get("verb", "")), str(a.get("noun", ""))) for a in (ex.get("topk_actions") or [])[:5]]
        if (ex.get("gt_verb"), ex.get("gt_noun")) in c5:
            kept.append(ex)
    print(f"[data] oracle-subset {len(kept)}/{len(rows)} (coverage {len(kept)/len(rows):.3f})")
    (out / "oracle_manifest.json").write_text(json.dumps(
        {"num_total": len(rows), "num_kept": len(kept),
         "coverage": round(len(kept) / len(rows), 4), "policy": "drop"}))

    rng = random.Random(42)
    order = kept * ((args.max_samples // max(1, len(kept))) + 1)
    rng.shuffle(order)
    order = order[: args.max_samples]

    prompt_rng = random.Random(42)
    log = open(out / "gr_log.jsonl", "a", encoding="utf-8")
    baseline = 0.3   # 초기값 ≈ base 예상 acc — 수십 샘플 내 EMA 로 수렴
    run_loss = run_r = run_adv = 0.0
    seen = 0
    opt.zero_grad(set_to_none=True)
    def encode_one(ex):
        conv = T.build_joint_conversation(ex, top_k=5, rng=prompt_rng)
        img = Image.open(conv["image"]).convert("RGB")
        msgs = [{"role": "system", "content": [{"type": "text", "text": conv["prompt"][0]["content"]}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": conv["prompt"][1]["content"]}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text, img, conv

    bel_buf: deque[str] = deque(maxlen=args.cons_buffer)   # swap belief 링버퍼 (P3)
    run_cons, cons_n, low = 0.0, 0, 0
    B = max(1, args.batch_gen)
    for bstart in range(0, len(order), B):
        batch = order[bstart:bstart + B]
        pairs = [encode_one(ex) for ex in batch]
        benc = processor(text=[p[0] for p in pairs], images=[[p[1]] for p in pairs],
                         return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            bgen = model.generate(**benc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                  temperature=args.temperature, top_p=0.95,
                                  pad_token_id=processor.tokenizer.pad_token_id)
        plen_pad = benc["input_ids"].shape[1]
        items = []
        for bi, ex in enumerate(batch):
            cids = bgen[bi][plen_pad:]
            items.append((ex, pairs[bi], cids))
        del benc, bgen
        i = bstart - 1
        for ex, (text, img, conv), comp_ids in items:
         i += 1
         comp_text = processor.tokenizer.decode(comp_ids, skip_special_tokens=True)
         enc = processor(text=[text], images=[[img]], return_tensors="pt").to(model.device)
         v, n, _ = T.parse_action_from_think_format(comp_text)
         if args.reward == "gt":
             r = 1.0 if (v and n and v == ex["gt_verb"] and n == ex["gt_noun"]) else 0.0
         else:   # wm: 선택 후보의 재정규화 likelihood (wm_clean 과 동일 정의, GT 불사용)
             acts = (ex.get("topk_actions_with_score") or [])[:5]
             liks = [a.get("likelihood") for a in acts]
             mi = next((k for k, a in enumerate(acts)
                        if a.get("verb") == v and a.get("noun") == n), None)
             s = sum(float(x) for x in liks if x is not None)
             r = (float(liks[mi]) / s) if (mi is not None and liks[mi] is not None and s > 0) else 0.0
         adv = r - baseline
         baseline = args.ema * baseline + (1 - args.ema) * r

         keep = comp_ids != processor.tokenizer.pad_token_id
         comp_ids = comp_ids[keep]
         if comp_ids.numel() == 0:
             continue
         full = torch.cat([enc["input_ids"][0], comp_ids]).unsqueeze(0)
         attn = torch.ones_like(full)
         extra = {}
         if "mm_token_type_ids" in enc:   # Qwen3-VL M-RoPE: completion 토큰은 텍스트(0)
             mm = enc["mm_token_type_ids"][0]
             pad = torch.zeros(comp_ids.numel(), dtype=mm.dtype, device=mm.device)
             extra["mm_token_type_ids"] = torch.cat([mm, pad]).unsqueeze(0)
         out_logits = model(input_ids=full, attention_mask=attn,
                            pixel_values=enc.get("pixel_values"),
                            image_grid_thw=enc.get("image_grid_thw"), **extra).logits
         plen = enc["input_ids"].shape[1]
         lp = torch.log_softmax(out_logits[0, plen - 1:-1].float(), dim=-1)
         tok_lp = lp.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)
         # credit=action: advantage 를 <action> 태그 이후 토큰에만 건다. 기본(all)은 완성부 전체.
         #   R1 진단(belief span +0.917 vs action +0.007)과 F0 ③≈0.008 이 같은 원인 —
         #   행동으로 정해진 credit 이 reasoning/belief 토큰에 균등 분배된다 — 을 가리켜 도입.
         if args.credit == "action":
             k = _action_token_start(processor.tokenizer, comp_ids)
             span_lp = tok_lp[k:] if (k is not None and k < tok_lp.numel()) else None
             if span_lp is None or span_lp.numel() == 0:
                 continue                      # action 태그를 못 찾은 완성 → 스킵(무학습)
             tok_lp = span_lp
         elif args.credit == "belief":
             # belief 는 action 과 달리 고엔트로피 자유텍스트 — span margin 을 전부 가져간
             # 자리(R1 +0.917 / P12 +0.40)이므로 여기에 credit 을 건다 (개선 3).
             sp = _belief_token_span(processor.tokenizer, comp_ids)
             span_lp = (tok_lp[sp[0]:min(sp[1], tok_lp.numel())]
                        if (sp is not None and sp[0] < tok_lp.numel()) else None)
             if span_lp is None or span_lp.numel() == 0:
                 continue                      # belief 태그를 못 찾은 완성 → 스킵(무학습)
             tok_lp = span_lp
         # credit_reduction: mean=기존(길이 정규화) · sum=정규화 없음(짧은 span 의 신호 보존)
         span_lp_red = tok_lp.sum() if args.credit_reduction == "sum" else tok_lp.mean()
         loss = -(adv * span_lp_red) / args.accum
         loss.backward()
         del out_logits, lp, enc

         # ── 개선 2 / P3: belief-swap consistency ──────────────────────────
         # belief 를 다른 샘플 것으로 바꾼 prefix 에서 후보 5개를 스코어링하고,
         # 원 action 의 상대 확률을 최선 대안 대비 끌어내린다. a_swap 을 생성하지 않고
         # WM top-5 안에서 고르므로 generate 추가 없이 forward 1회로 끝난다.
         if args.cons_weight > 0:
             mr, mb = TAG_R.search(comp_text), TAG_B.search(comp_text)
             belief_txt = mb.group(1).strip() if mb else ""
             if belief_txt and mr and v and n and len(bel_buf) >= args.cons_warmup:
                 b_swap = next((b for b in reversed(bel_buf)
                                if b.strip().lower() != belief_txt.lower()), None)
                 if b_swap is not None:
                     cands = [(str(d["verb"]), str(d["noun"]))
                              for d in json.loads(conv["topk_actions_display"])]
                     if (v, n) not in cands:
                         cands = cands + [(v, n)]   # 모델이 후보 밖 action 을 낸 경우
                     a_orig = cands.index((v, n))
                     base_txt = (f"{text}<reasoning>{mr.group(1).strip()}</reasoning>\n"
                                 f"<task_belief>{b_swap}</task_belief>\n")
                     benc2 = processor(text=[base_txt], images=[[img]],
                                       return_tensors="pt").to(model.device)
                     q = torch.log_softmax(
                         score_candidates(model, processor, benc2, cands), dim=0)
                     # a_swap 은 a_orig 을 **제외한** 최선 후보다. 제외하지 않고 단순 argmax 로
                     # 두면 belief 를 바꿔도 argmax 가 그대로일 때 두 항이 상쇄돼 loss=0 이 된다
                     # — 정확히 벌해야 할 경우에 gradient 가 사라진다. 핸드오프 원식의 이 구멍을
                     # 막기 위해 제외 argmax 로 쓴다.
                     alt = torch.cat([q[:a_orig], q[a_orig + 1:]])
                     if alt.numel() > 0:
                         gap = q[a_orig] - alt.max()      # = log q(a_orig) − log q(a_swap)
                         # hinge 가 없으면 목표 달성 후에도 계속 밀어 정책을 파괴한다 (실측).
                         cl = (torch.clamp(gap + args.cons_margin, min=0.0)
                               if args.cons_margin > 0 else gap)
                         (args.cons_weight * cl / args.accum).backward()
                         run_cons += float(cl); cons_n += 1
             if belief_txt:
                 bel_buf.append(belief_txt)

         run_loss += float(loss) * args.accum; run_r += r; run_adv += abs(adv); seen += 1
         if (i + 1) % args.accum == 0:
             torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
             opt.step(); opt.zero_grad(set_to_none=True)
         if seen % args.log_every == 0:
             rec = {"seen": seen, "loss": round(run_loss / args.log_every, 4),
                    "reward_ma": round(run_r / args.log_every, 4),
                    "baseline": round(baseline, 4),
                    "mean_abs_adv": round(run_adv / args.log_every, 4)}
             if args.cons_weight > 0:
                 # cons_loss > 0 이면 swap belief 하에서도 원 action 이 여전히 최선 —
                 # belief 가 조향에 실패한 상태다. 이 값이 내려가는지가 P3 의 학습 신호.
                 rec["cons_loss"] = round(run_cons / cons_n, 4) if cons_n else None
                 rec["cons_applied"] = round(cons_n / args.log_every, 4)
             print(f"[gr] {rec}", flush=True)
             log.write(json.dumps(rec) + "\n"); log.flush()
             # 붕괴 가드: consistency 항이 belief 를 쓰게 만드는 대신 행동 선호 자체를
             # 파괴하는 경로로 빠지면 reward_ma 가 먼저 무너진다. 6시간을 태우지 않고 멈춘다.
             if args.reward_floor > 0:
                 low = low + 1 if rec["reward_ma"] < args.reward_floor else 0
                 if low >= args.reward_floor_patience:
                     print(f"[gr][ABORT] reward_ma {rec['reward_ma']} < {args.reward_floor} 가 "
                           f"{low}회 연속 — 정책 붕괴로 판단하고 중단한다.", flush=True)
                     log.write(json.dumps({"abort": "reward_floor", "seen": seen,
                                           "reward_ma": rec["reward_ma"]}) + "\n"); log.flush()
                     model.save_pretrained(out / f"checkpoint-abort-{seen}")
                     raise SystemExit(3)
             run_loss = run_r = run_adv = run_cons = 0.0
             cons_n = 0
         if seen % args.save_every == 0:
             model.save_pretrained(out / f"checkpoint-{seen}")
    model.save_pretrained(out / "checkpoint-final")
    processor.save_pretrained(out / "checkpoint-final")
    print(f"[DONE] F0-GR adapter → {out}/checkpoint-final")


if __name__ == "__main__":
    main()
