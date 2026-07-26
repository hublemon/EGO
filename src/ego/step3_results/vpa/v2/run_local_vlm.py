"""VPA v2 — 로컬 VLM 러너. Qwen3-VL 백본(무학습) 및 blind control.

두 arm 을 한 스크립트로 돌린다:
  --mode frames : 8프레임 + 텍스트  (A1 백본)
  --mode blind  : 텍스트만          (A2 blind control)

blind 는 **프롬프트에서 프레임 관련 문장만 빼고 나머지는 완전히 동일**하다
(`common.build_prompt(with_frames=False)`). 두 arm 의 차이가 곧 프레임 기여분이며,
이것이 "프레임을 넣는다"는 본 재작성의 정당성을 재는 유일한 대조다.

백본 모델 기본값은 step2 의 베이스와 동일한 `Qwen/Qwen3-VL-8B-Instruct`
(`step2_retrospection/vlm.py`의 MODEL_NAME). step2 학습은 이 위의 LoRA 어댑터이므로
이 arm 은 임의의 오픈웨이트 비교군이 아니라 **어댑터를 뗀 대조군**이다.

사용:
  PYTHONPATH=src python -m ego.step3_results.vpa.v2.run_local_vlm \
      --gt runs/vpa_v2/vpa_v2_T3.json --mode frames --out runs/vpa_v2/preds/qwen_backbone_T3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ego.step3_results.vpa.v2 import common as C
from ego.step3_results.vpa.v2 import frames as F

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_model(model_path: str, adapter: str | None = None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    # ⚠ 순서 주의: 패치는 **로딩 후**에 적용한다. 로딩 경로가 prod를 반복 호출하기 때문에
    # 먼저 적용하면 CPU 왕복이 누적돼 로딩이 사실상 멈춘다 (실측: 133s → 10분+ 미완).
    from ego.step3_results.vpa.v2 import gb10_compat
    if gb10_compat.apply():
        print("[info] GB10(sm_121) nvrtc JIT 우회 패치 적용 — prod 연산만 CPU 경유")
    return model, processor


def generate(model, processor, system: str, user: str, images: list, max_new_tokens: int = 96) -> str:
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": user})
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--vocab", default="runs/vpa_v2/vocab.json")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["frames", "blind"], default="frames")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="PEFT 어댑터 경로 (ours arm 용, 선택)")
    p.add_argument("--candidates", choices=["vocab", "wm10_first"], default="vocab",
                   help="vocab=전체 어휘 293(전 arm 공통·비교 가능) · "
                        "wm10_first=1번째만 WM top-10 제약, 2번째 이후 자유(EGO 배포 형태)")
    p.add_argument("--frames-dir", default="runs/vpa_v2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--require-frames", action="store_true", default=True,
                   help="blind 도 frames arm 과 **동일 표본**에서만 돌린다(짝 비교 보장)")
    args = p.parse_args()

    samples = C.load_json(args.gt)
    T = samples[0]["horizon"]
    vocab = C.load_json(args.vocab)["labels"]
    cache_root = Path(args.frames_dir) / F.cache_dirname()

    # frames arm 과 blind arm 의 표본을 일치시킨다 — 프레임이 있는 샘플만 양쪽 모두 평가
    todo = [s for s in samples if all(p.is_file() for p in F.frame_paths(cache_root, s))]
    if args.limit:
        todo = todo[: args.limit]

    out_prefix = Path(args.out)
    rec_path = out_prefix.with_suffix(".records.jsonl")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["sample_id"] for r in C.read_jsonl(rec_path) if r.get("ok")}
    todo = [s for s in todo if s["sample_id"] not in done]

    print(f"[info] mode={args.mode} model={args.model_path} adapter={args.adapter}")
    print(f"[info] eligible={len(todo) + len(done)} · done={len(done)} · todo={len(todo)}")
    if not todo:
        print("[info] 남은 샘플 없음.")
    else:
        from PIL import Image
        model, processor = load_model(args.model_path, args.adapter)
        with open(rec_path, "a") as fh:
            for i, s in enumerate(todo, 1):
                system, user = C.build_prompt(s, vocab, T, with_frames=(args.mode == "frames"),
                                              candidate_mode=args.candidates)
                images = []
                if args.mode == "frames":
                    images = [Image.open(p).convert("RGB") for p in F.frame_paths(cache_root, s)]
                try:
                    raw = generate(model, processor, system, user, images)
                    pred = C.parse_prediction(raw, T)
                    err = None
                except Exception as e:  # noqa: BLE001
                    raw, pred, err = "", [], str(e)[:200]
                fh.write(json.dumps({"sample_id": s["sample_id"], "video_uid": s["video_uid"],
                                     "ok": bool(pred), "pred": pred, "raw": raw[:400],
                                     "error": err}, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] last={pred}")

    preds = {r["sample_id"]: r["pred"] for r in C.read_jsonl(rec_path) if r.get("ok")}
    C.dump_json(out_prefix.with_suffix(".json"), preds)
    print(f"\nwrote {out_prefix.with_suffix('.json')} ({len(preds)} preds)")


if __name__ == "__main__":
    main()
