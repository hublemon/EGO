"""Task 4 -- Baseline 1: frontier VLM via an OpenAI-compatible chat API.

Text-conditioned VPA: each sample's goal + observed step history + the candidate
vocabulary are sent as a prompt; the model must return exactly T next steps,
in order, chosen from the vocabulary, as a JSON array.

SECURITY: the API key is read ONLY from the environment variable
FRONTIER_API_KEY. It is never printed, logged, or written to any output. The
endpoint and model are configurable via FRONTIER_BASE_URL / FRONTIER_MODEL
(or --base-url / --model).

A future RL-trained VLM plugs in here unchanged: emit the same preds json
({sample_id: [labels]}) and score it with eval_vpa.py.

Usage:
    export FRONTIER_API_KEY=...            # never hardcode
    export FRONTIER_BASE_URL=https://api.openai.com/v1
    export FRONTIER_MODEL=gpt-4o-mini
    python scripts/vpa/run_frontier_baseline.py \
        --gt outputs/goalstep/vpa/goalstep_vpa_T3.json \
        --vocab outputs/goalstep/vpa/candidate_vocab.json \
        --out outputs/goalstep/vpa/preds_frontier_T3.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpa_common import dump_json, load_json  # noqa: E402

import requests  # noqa: E402


def build_prompt(sample, vocab, T, item_name="step"):
    goal = sample["goal_text"]
    observed = sample["observed_steps"]
    obs_txt = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(observed)) or "  (none yet)"
    vocab_txt = "\n".join(f"- {v}" for v in vocab)
    system = (
        "You are a procedural planning assistant for cooking videos. Given the "
        f"goal and the {item_name}s already performed, predict the next {item_name}s. You MUST "
        f"choose labels only from the provided candidate {item_name} list, output EXACTLY "
        f"{T} labels, in temporal order, as a JSON array of strings and nothing else."
    )
    user = (
        f"GOAL: {goal}\n\n"
        f"{item_name.upper()}S ALREADY DONE (in order):\n{obs_txt}\n\n"
        f"CANDIDATE {item_name.upper()} LABELS (choose only from these):\n{vocab_txt}\n\n"
        f"Predict the next {T} {item_name}s as a JSON array of exactly {T} labels copied "
        f"verbatim from the candidate list, in order. Output ONLY the JSON array."
    )
    return system, user


def parse_prediction(content, T):
    """Extract a JSON array of up to T strings from the model reply; tolerate
    code fences / surrounding prose."""
    if not content:
        return []
    txt = content.strip()
    if "```" in txt:
        parts = txt.split("```")
        for seg in parts:
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            if seg.startswith("["):
                txt = seg
                break
    start, end = txt.find("["), txt.rfind("]")
    if start != -1 and end != -1 and end > start:
        txt = txt[start:end + 1]
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            return [str(x) for x in arr][:T]
    except json.JSONDecodeError:
        pass
    return []


def call_api(base_url, api_key, model, system, user, max_retries=4, timeout=60):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], None
        except Exception as e:  # noqa: BLE001 -- network/parse errors are all retryable here
            last_err = str(e)[:200]
            time.sleep(min(2 ** attempt, 8))
    return None, last_err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--limit", type=int, default=None, help="only run the first N samples (cost control)")
    p.add_argument("--base-url", default=os.environ.get("FRONTIER_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--model", default=os.environ.get("FRONTIER_MODEL", "gpt-4o-mini"))
    args = p.parse_args()

    api_key = os.environ.get("FRONTIER_API_KEY")
    if not api_key:
        sys.exit("ERROR: set FRONTIER_API_KEY in the environment (never hardcode it).")

    samples = load_json(args.gt)
    if args.split != "all":
        samples = [s for s in samples if s["eval_split"] == args.split]
    vmeta = load_json(args.vocab)
    vocab = vmeta["labels"]
    item_name = "action (verb + object)" if vmeta.get("label_mode") == "action" else "step"
    T = samples[0]["horizon"] if samples else 3
    if args.limit:
        samples = samples[:args.limit]
    print(f"[info] endpoint={args.base_url} model={args.model} samples={len(samples)} T={T}")

    preds, n_ok, n_fail = {}, 0, 0
    for i, s in enumerate(samples):
        system, user = build_prompt(s, vocab, T, item_name)
        content, err = call_api(args.base_url, api_key, args.model, system, user)
        if content is None:
            n_fail += 1
            preds[s["sample_id"]] = []  # fallback: empty prediction
            print(f"  [{i+1}/{len(samples)}] {s['sample_id']}: API FAIL ({err})")
            continue
        pred = parse_prediction(content, T)
        preds[s["sample_id"]] = pred
        n_ok += 1 if pred else 0
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}: {pred}")

    dump_json(args.out, preds)
    print(f"\nwrote {args.out}  (ok={n_ok}, fail={n_fail}, total={len(samples)})")
    if n_fail == len(samples):
        print("NOTE: all API calls failed -- pipeline/format is verified but no live scores were produced.")


if __name__ == "__main__":
    main()
