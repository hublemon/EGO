"""generate_faa_traces.py — frozen FAA online full-trace rollout (핸드오프 §4, GPU).

freeze 된 FAA(LoRA adapter)로 각 past-only prompt 에서 full-trace 를 num_generations 개 생성.
F0 와 **동일한 프롬프트 빌더**(train_grpo_action.build_joint_conversation)를 사용해야
train/inference 분포가 일치한다. 여기서는 F0 학습 jsonl(grpo_train.jsonl, 이미 4f·L2-c 반영)을
읽어 그대로 프롬프트를 만들고, FAA 를 얹어 생성한다.

출력: faa_traces_{split}.jsonl — {sample_id, prompt, image_path, memory_context,
      candidates, faa_traces:[completion...], faa_checkpoint_hash}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
sys.path.insert(0, str(EGO_ROOT / "src/ego/step2_vlm_alignment"))


def _ckpt_hash(adapter_path: str) -> str:
    p = Path(adapter_path) / "adapter_model.safetensors"
    if not p.exists():
        p = next(Path(adapter_path).glob("*.safetensors"), None)
    if p and p.exists():
        return hashlib.md5(p.read_bytes()).hexdigest()[:16]
    return "nockpt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faa_adapter", required=True, help="frozen FAA LoRA adapter 경로")
    ap.add_argument("--train_jsonl", required=True, help="F0 grpo_{train,heldout}.jsonl (4f·L2-c)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # F0 프롬프트 빌더 재사용 (분포 일치). reward_mode=wm_likelihood_joint 경로.
    import train_grpo_action as T  # noqa: E402
    T.MASK_FRAME_PROB = 0.0        # rollout 은 마스킹 없음

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.faa_adapter)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True, padding_side="left")
    ckpt_hash = _ckpt_hash(args.faa_adapter)

    rows = [json.loads(l) for l in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    import random
    rng = random.Random(42)
    out_f = Path(args.out).open("w", encoding="utf-8")
    n = 0
    for ex in rows:
        conv = T.build_joint_conversation(ex, top_k=5, rng=rng)
        # 프롬프트/이미지 준비 (evaluate.py 와 동일한 멀티모달 변환)
        img = Image.open(conv["image"]).convert("RGB")
        messages = T_to_multimodal(conv["prompt"], img)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        gens = []
        for _ in range(args.num_generations):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=args.temperature, top_p=0.95)
            gen = out[0][inputs["input_ids"].shape[1]:]
            gens.append(processor.decode(gen, skip_special_tokens=True))
        # candidates 는 화면에 보인 (verb,noun) 목록 (build_joint 이 topk_actions_display 로 기록)
        candidates = json.loads(conv.get("topk_actions_display", "[]"))
        out_f.write(json.dumps({
            "sample_id": conv["sample_id"],
            "prompt": conv["prompt"],
            "image_path": conv["image"],
            "memory_context": conv.get("memory_context", ""),
            "candidates": candidates,
            "faa_traces": gens,
            "faa_checkpoint_hash": ckpt_hash,
        }, ensure_ascii=False) + "\n")
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}")
    out_f.close()
    print(f"[done] {n} samples × {args.num_generations} traces → {args.out} (faa={ckpt_hash})")


def T_to_multimodal(prompt_msgs, image):
    """train 포맷(문자열 content) → 멀티모달 메시지 (user 턴 앞 image 주입).
    evaluate.py.to_multimodal_messages 와 동일 규칙."""
    out = []
    for m in prompt_msgs:
        if m["role"] == "user":
            out.append({"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": m["content"]},
            ]})
        else:
            out.append(m)
    return out


if __name__ == "__main__":
    main()
