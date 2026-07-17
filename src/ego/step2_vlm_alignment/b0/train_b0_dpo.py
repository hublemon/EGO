"""train_b0_dpo.py — full-trace sequence-level DPO (핸드오프 §11, GPU).

B0 = FAA adapter 에서 초기화 · frozen FAA 를 reference 로 사용하는 표준 DPO.
  L_B0 = -log σ[ β( log π_B0(y+)/π_FAA(y+) - log π_B0(y-)/π_FAA(y-) ) ]
전체 completion 을 하나의 preference 로 — 필드 분리·splicing 없음.

데이터: b0_dpo_{split}.jsonl (build_dpo_dataset 산출) — {prompt, image_path, chosen, rejected}.
TRL DPOTrainer 사용. reference = 학습 전 FAA adapter (동일 가중치 복제, frozen).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_dpo_dataset(path: str, limit: int | None = None):
    """b0_dpo jsonl → TRL DPOTrainer 데이터셋.

    ⚠ TRL VLM DPO 규약: 이미지는 "images" 컬럼(샘플당 리스트). GRPO 의 "image" 단수와 다르다.
      prompt 는 F0 와 동일한 conversational(문자열 content) — trl 이 chat template 적용 시
      images 를 주입한다. 서버의 trl 버전에서 컬럼명 규약을 스모크로 확인할 것
      (docs/experiments/2026-07-18_b0_implementation.md §7)."""
    from datasets import Dataset, Image as DSImage, Sequence
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append({
            "prompt": r["prompt"],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
            "images": [r["image_path"]] if r.get("image_path") else [],
        })
        if limit and len(rows) >= limit:
            break
    ds = Dataset.from_list(rows)
    if rows and rows[0]["images"]:
        ds = ds.cast_column("images", Sequence(DSImage()))
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_jsonl", required=True)
    ap.add_argument("--faa_adapter", required=True, help="초기값 + reference (frozen FAA)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO KL 온도")
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=640)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=2)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import DPOConfig, DPOTrainer

    # policy: base + FAA adapter 를 trainable 로 이어학습 (FAA 에서 초기화)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))}
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "auto")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    policy = PeftModel.from_pretrained(base, args.faa_adapter, is_trainable=True)

    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True, padding_side="left")
    train_ds = load_dpo_dataset(args.dpo_jsonl)
    print(f"[data] {len(train_ds)} DPO pairs from {args.dpo_jsonl}")

    cfg = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        # reference = adapter 비활성화한 동일 모델 (TRL 이 PEFT 시 ref_model=None 이면
        # adapter off 를 reference 로 사용 — frozen FAA 가 아니라 base 가 되므로 주의).
    )
    # ⚠ reference 는 frozen FAA 여야 한다(핸드오프 §11). PEFT + ref_model=None 은 base 를
    #   reference 로 쓰므로, 명시적으로 FAA adapter 를 얹은 별도 frozen 모델을 ref 로 전달.
    ref_base = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="auto")
    ref = PeftModel.from_pretrained(ref_base, args.faa_adapter)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=cfg,
        train_dataset=train_ds,
        processing_class=processor,
    )
    meta = {
        "experiment": Path(args.output_dir).name,
        "faa_adapter": args.faa_adapter, "beta": args.beta,
        "lr": args.learning_rate, "epochs": args.num_train_epochs,
        "n_pairs": len(train_ds), "model": args.model_name,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "b0_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[DONE] B0 adapter → {args.output_dir}")


if __name__ == "__main__":
    main()
