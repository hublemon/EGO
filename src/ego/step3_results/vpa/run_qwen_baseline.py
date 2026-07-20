"""Task 5 -- Baseline 2: local Qwen3-VL-7B-Instruct via transformers.

Same text-conditioned VPA prompt and output format as the frontier baseline, so
its preds json scores with eval_vpa.py unchanged. The history is text-only
(goal + observed step labels); a frame-input hook is left as a commented stub so
video frames can be added later without changing the interface.

Requirements (install when weights arrive):
    pip install "transformers>=4.51" accelerate torch qwen-vl-utils

Modes:
    --dry-run : no weights needed. Runs the loader-less path on a couple of dummy
                samples to smoke-test prompt construction + reply parsing on
                CPU/GPU. Use this to verify plumbing before the model exists.
    (default) : loads --model-path and runs the full split. If the model is
                missing it exits with a friendly download hint.

Usage:
    python src/ego/step3_results/vpa/run_qwen_baseline.py --dry-run \
        --gt src/ego/step3_results/vpa/data/goalstep_vpa_T3.json \
        --vocab src/ego/step3_results/vpa/data/candidate_vocab.json --out /tmp/preds_qwen_T3.json
    # once weights are present:
    python src/ego/step3_results/vpa/run_qwen_baseline.py --model-path Qwen/Qwen3-VL-7B-Instruct \
        --gt ... --vocab ... --out src/ego/step3_results/vpa/data/preds_qwen_T3.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpa_common import dump_json, load_json  # noqa: E402
# Reuse the exact same prompt + parser as the frontier baseline for parity.
from run_frontier_baseline import build_prompt, parse_prediction  # noqa: E402


def load_qwen(model_path):
    """Load Qwen3-VL. Kept isolated so --dry-run never imports heavy deps."""
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        sys.exit(
            "ERROR: transformers/torch not installed. Run:\n"
            "  pip install \"transformers>=4.51\" accelerate torch qwen-vl-utils\n"
            f"(import error: {e})"
        )
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto")
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"ERROR: could not load Qwen3-VL from '{model_path}'.\n"
            "If weights are not downloaded yet:\n"
            "  huggingface-cli download Qwen/Qwen3-VL-7B-Instruct --local-dir checkpoints/qwen3vl-7b\n"
            "then pass --model-path checkpoints/qwen3vl-7b\n"
            f"(load error: {str(e)[:200]})"
        )
    return model, processor


def qwen_generate(model, processor, system, user, max_new_tokens=128):
    import torch  # noqa: F401
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        # FRAME HOOK: to add video later, append {"type": "video", "video": <path>} here
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def dummy_samples(vocab, T):
    v = vocab[:6] if len(vocab) >= 6 else vocab
    return [
        {"sample_id": "DUMMY_1", "goal_text": "make bread",
         "observed_steps": v[:2], "future_steps": v[2:2 + T], "horizon": T, "eval_split": "dev"},
        {"sample_id": "DUMMY_2", "goal_text": "make salad",
         "observed_steps": v[:1], "future_steps": v[1:1 + T], "horizon": T, "eval_split": "dev"},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-7B-Instruct")
    p.add_argument("--dry-run", action="store_true",
                   help="smoke-test prompt+parse on dummy samples, no weights needed")
    args = p.parse_args()

    vmeta = load_json(args.vocab)
    vocab = vmeta["labels"]
    item_name = "action (verb + object)" if vmeta.get("label_mode") == "action" else "step"

    if args.dry_run:
        T = load_json(args.gt)[0]["horizon"] if load_json(args.gt) else 3
        samples = dummy_samples(vocab, T)
        print(f"[dry-run] verifying prompt build + parse on {len(samples)} dummy samples (T={T}), no model loaded")
        preds = {}
        for s in samples:
            system, user = build_prompt(s, vocab, T, item_name)
            fake_reply = "```json\n" + str(s["future_steps"]) + "\n```"  # simulate a well-formed reply
            pred = parse_prediction(fake_reply.replace("'", '"'), T)
            preds[s["sample_id"]] = pred
            print(f"  {s['sample_id']}: prompt_chars={len(system)+len(user)} parsed={pred}")
        dump_json(args.out, preds)
        print(f"[dry-run] OK -- wrote {args.out}. Plumbing verified; supply real weights to get scores.")
        return

    samples = load_json(args.gt)
    if args.split != "all":
        samples = [s for s in samples if s["eval_split"] == args.split]
    T = samples[0]["horizon"] if samples else 3
    if args.limit:
        samples = samples[:args.limit]

    model, processor = load_qwen(args.model_path)
    print(f"[info] loaded {args.model_path}; running {len(samples)} samples (T={T})")
    preds = {}
    for i, s in enumerate(samples):
        system, user = build_prompt(s, vocab, T, item_name)
        reply = qwen_generate(model, processor, system, user)
        preds[s["sample_id"]] = parse_prediction(reply, T)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)}")
    dump_json(args.out, preds)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
