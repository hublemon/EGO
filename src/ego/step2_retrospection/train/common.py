"""트레이너 공통 — 3태그 span 토큰 매핑, VLM 수동 forward 입력 조립.

핵심 설계:
- 프롬프트는 processor(chat template + 이미지)로, completion(assistant 턴)은
  tokenizer 조각별(add_special_tokens=False)로 만들어 이어붙인다. 조각 경계가
  span 경계와 일치하므로 field별 토큰 마스크가 정확하다.
- Qwen3-VL M-RoPE: 프롬프트 인코딩의 per-token 텐서(mm_token_type_ids 등)는
  completion 길이만큼 0으로 연장한다 (F0 시절 확인된 이슈의 일반화 대응).
"""
from __future__ import annotations

import torch

FIELD_WEIGHTS = {"task_belief": 1.0, "reasoning": 0.5, "action": 0.25}
FIELDS = ("reasoning", "task_belief", "action")


def completion_pieces(reasoning: str, task_belief: str, action: str) -> list[tuple[str, str]]:
    """(field, text) 조각 — serialize_trace와 동일한 직렬화를 조각으로 표현."""
    return [
        ("reasoning", "<reasoning>\n"), ("reasoning", reasoning.strip()),
        ("reasoning", "\n</reasoning>\n\n"),
        ("task_belief", "<task_belief>\n"), ("task_belief", task_belief.strip()),
        ("task_belief", "\n</task_belief>\n\n"),
        ("action", "<action>\n"), ("action", action.strip()), ("action", "\n</action>"),
    ]


def encode_completion(tokenizer, fields: dict, assistant_prefix: str = "<|im_start|>assistant\n",
                      assistant_suffix: str = "<|im_end|>\n"):
    """completion 토큰 ids + field 라벨 리스트를 반환."""
    ids: list[int] = []
    field_of: list[str] = []
    pieces = [("action", assistant_prefix)] + completion_pieces(
        fields["reasoning"], fields["task_belief"], fields["action"]) + [("action", assistant_suffix)]
    for field, text in pieces:
        t = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(t)
        field_of.extend([field] * len(t))
    return ids, field_of


def build_prompt_inputs(processor, messages, device):
    """assistant 직전까지의 프롬프트 인코딩 (이미지 포함)."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [c["image"] for m in messages for c in m["content"]
              if isinstance(c, dict) and c.get("type") == "image"]
    return processor(text=[text], images=images or None, return_tensors="pt").to(device)


def cat_completion(prompt_inputs, comp_ids: list[int], device):
    """프롬프트 인코딩 뒤에 completion을 이어붙인 forward 입력 + completion 마스크."""
    p_len = prompt_inputs["input_ids"].shape[1]
    comp = torch.tensor([comp_ids], device=device)
    out = {}
    for k, v in prompt_inputs.items():
        if k == "input_ids":
            out[k] = torch.cat([v, comp], dim=1)
        elif k == "attention_mask":
            out[k] = torch.cat([v, torch.ones_like(comp)], dim=1)
        elif isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == p_len:
            pad = torch.zeros((v.shape[0], len(comp_ids), *v.shape[2:]), dtype=v.dtype, device=device)
            out[k] = torch.cat([v, pad], dim=1)  # mm_token_type_ids 등 per-token 텐서
        else:
            out[k] = v
    return out, p_len


def field_logps(model, inputs, p_len: int, field_of: list[str]):
    """completion 구간의 field별 (length-normalized logp, 토큰수) — FB-DPO/SFT 공용."""
    logits = model(**inputs).logits  # (1, L, V)
    ids = inputs["input_ids"]
    # 위치 t 예측 대상 = ids[t]; completion 시작은 p_len
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]  # (L-1,)
    out = {}
    for f in FIELDS:
        idxs = [p_len - 1 + i for i, ff in enumerate(field_of) if ff == f]
        sel = tok_lp[idxs]
        out[f] = (sel.mean(), len(idxs))
    return out


def span_ce_loss(model, inputs, p_len: int, field_of: list[str],
                 weights: dict = FIELD_WEIGHTS):
    """span-normalized SFT loss: Σ λ_f · mean CE(field). (loss, per-field dict)"""
    fps = field_logps(model, inputs, p_len, field_of)
    loss = sum(weights[f] * (-fps[f][0]) for f in FIELDS)
    return loss, {f: round(float(-fps[f][0].detach()), 4) for f in FIELDS}
