#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_f0_v2.py — F0 final plan v2 데이터/프롬프트 로직 스모크 테스트 (GPU/torch 불필요).

검증 대상 (F0 final plan §0):
  §0-4 strict cutoff: stop < trigger 엄격 부등호 + 3제외 (가로지름·동일·NaN)
  L2-c  frame-aligned 직렬화가 프레임 시각과 정합하고 미래 누설이 없다
  future_gt_actions 가 학습 프롬프트로 새지 않고 b0meta 로만 분리된다
  L2-a  프레임 마스킹이 결정론적이고 시스템 프롬프트를 교체한다
  L2-d  belief 재진술 금지 지시문이 프롬프트에 있다
  프록시 휴리스틱(history_reference_rate, belief_restatement_flag)이 동작한다

torch/decord/transformers 없이 순수 로직만 검사 — 학습 서버 이전 로컬에서 실행 가능.
합성 데이터로 각 규칙의 positive/negative 케이스를 만들어 단언한다.
exit 0 = 전부 통과.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "src/ego/step2_vlm_alignment/data"
TRAIN = REPO / "src/ego/step2_vlm_alignment/train_grpo_action.py"

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "OK  " if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not cond else ""))


def load_module(path: Path, name: str, stub_pandas: bool = False):
    """단일 파일을 모듈로 로드. pandas 의존 파일은 최소 stub 로 우회."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# 1. strict cutoff / frame-aligned / future — extract_memory_train 의 순수 함수 직접 테스트
#    (pandas DataFrame 대신 가벼운 shim 으로 필요한 연산만 지원)
# ─────────────────────────────────────────────────────────────────────────────
def test_memory_logic() -> None:
    print("[1] extract_memory_train — strict cutoff / frame-aligned / future")
    import pandas as pd  # 학습 서버엔 있음. 로컬에 없으면 이 블록만 스킵.

    mem = load_module(DATA / "extract_memory_train.py", "emt")

    # 합성: 한 비디오, fps=60, trigger_frame=600 (=10초 지점)
    # take knife: 100-400 완료 (target 360=frame1 을 span → L2-c 양성 매치)
    # wash tomato: stop==600 (제외 ⑵, 현재 진행 중)  cross: 가로지름(550-650, 제외 ⑴)
    # cut tomato: 미래(start 660)  nanx: NaN stop (제외 ⑶)
    rows = [
        dict(video_id="V", verb="take", noun="knife", verb_class=0, noun_class=0, start_frame=100, stop_frame=400),
        dict(video_id="V", verb="wash", noun="tomato", verb_class=1, noun_class=1, start_frame=400, stop_frame=600),
        dict(video_id="V", verb="cross", noun="over",  verb_class=2, noun_class=2, start_frame=550, stop_frame=650),
        dict(video_id="V", verb="cut",  noun="tomato", verb_class=3, noun_class=1, start_frame=660, stop_frame=800),
        dict(video_id="V", verb="nanx", noun="bad",    verb_class=4, noun_class=3, start_frame=200, stop_frame=float("nan")),
    ]
    df = pd.DataFrame(rows)
    trigger = 600

    hist = mem.get_task_history_strict(df, "V", trigger)
    check("strict history excludes stop==trigger (wash tomato)", "wash tomato" not in hist,
          f"hist={hist}")
    check("strict history excludes crossing segment (cross over)", "cross over" not in hist)
    check("strict history excludes future (cut tomato)", "cut tomato" not in hist)
    check("strict history keeps completed (take knife)", "take knife" in hist)

    legacy = mem.get_task_history_legacy(df, "V", 601)   # legacy 는 stop<601 이라 wash 포함
    check("legacy DIFFERS from strict (leak reproduced)", ("wash tomato" in legacy) and ("wash tomato" not in hist))

    fa = mem.get_frame_aligned_context(df, "V", trigger, fps=60.0)
    check("frame-aligned captures past completed (frame1=take knife)",
          fa.get("frame1_t-4.0s") == "take knife", f"fa={fa}")
    check("frame-aligned excludes current(in-progress) action", all(
        v != "wash tomato" for v in fa.values()), f"fa={fa}")
    check("frame-aligned 'now' frame is empty (no leak of GT)", fa.get("frame4_t-0.0s") is None)
    check("frame-aligned keys match 4 frames", len(fa) == 4)

    fut = mem.get_future_gt_actions(df, "V", trigger, fps=60.0)
    verbs = [f["verb"] for f in fut]
    check("future = start>=trigger only (cut tomato in, wash out)",
          "cut" in verbs and "wash" not in verbs, f"fut={verbs}")
    check("future rows carry offset+timestamps", all("offset" in f and "start_sec" in f for f in fut))


# ─────────────────────────────────────────────────────────────────────────────
# 2. convert — L2-c 직렬화 / future 물리 분리
# ─────────────────────────────────────────────────────────────────────────────
def test_convert_logic() -> None:
    print("[2] convert_to_train_format — L2-c serialize / future separation")
    # pandas.read_csv 를 모듈 임포트 시점에 호출하므로 stub 후 로드
    import pandas as pd
    orig_read_csv = pd.read_csv
    # convert 는 로드 시점에 pd.read_csv(...).set_index("id")["key"].to_dict() 를 호출한다.
    # → "id","key" 컬럼을 가진 (인덱스 안 걸린) DataFrame 을 돌려줘야 한다.
    pd.read_csv = lambda *a, **k: pd.DataFrame({"id": [0, 1, 3], "key": ["take", "wash", "cut"]})
    try:
        conv = load_module(DATA / "convert_to_train_format.py", "c2tf")
    finally:
        pd.read_csv = orig_read_csv

    # L2-c: frame_aligned 있으면 시간정렬 직렬화
    s = conv.serialize_memory_v2(
        ["take knife", "wash tomato"],
        {"frame1_t-4.0s": "take knife", "frame2_t-2.67s": None,
         "frame3_t-1.33s": None, "frame4_t-0.0s": None},
        [4.0, 2.67, 1.33, 0.0])
    check("L2-c serialize mentions Frame labels", "Frame 1 (4.0s ago): take knife" in s, s)
    check("L2-c serialize marks empty moments", "no completed action" in s)
    check("L2-c serialize keeps earlier history", "Earlier completed actions" in s)

    # frame_aligned 없으면 legacy 폴백
    s2 = conv.serialize_memory_v2(["take knife"], {}, [0.0])
    check("empty frame_aligned falls back to legacy", "Previously completed actions" in s2)

    # b0_meta_of 가 future 를 담고, convert 결과는 담지 않음
    rec = {
        "sample_id": "s1", "video_id": "V", "trigger_frame": 600, "trigger_timestamp": 10.0,
        "frame_path": __file__,  # 존재하는 경로 아무거나 (convert 는 image 존재검사 안 함)
        "gt_label": {"verb": "cut", "noun": "tomato", "verb_class": 3, "noun_class": 1},
        "wm_output": {"top5_action": [{"verb_class": 3, "noun_class": 1, "likelihood": 0.2}],
                      "top5_noun": [{"noun": "tomato", "likelihood": 0.2}],
                      "top5_verb": [{"verb": "cut", "likelihood": 0.2}]},
        "memory_context": {"task_history": ["take knife"], "frame_aligned_context": {},
                           "cutoff_rule": "strict"},
        "frame_meta": {"n_frames": 4, "offsets_sec": [4.0, 2.67, 1.33, 0.0]},
        "future_gt_actions": [{"offset": 1, "verb": "serve", "noun": "dish"}],
    }
    # VERB_ID2KEY/NOUN_ID2KEY 는 stub 라 3/1 키가 없음 → convert 는 실제 CSV 필요.
    # 여기선 b0_meta_of 분리만 검증 (convert 자체는 서버 CSV 의존).
    b0 = conv.b0_meta_of(rec)
    check("b0_meta_of carries future_gt_actions", b0["future_gt_actions"] == rec["future_gt_actions"])
    check("b0_meta_of carries gt_action_t", b0["gt_action_t"]["verb"] == "cut")


# ─────────────────────────────────────────────────────────────────────────────
# 3. train_grpo — L2-a 마스킹 / L2-d 지시문 / 프록시 (torch 임포트 회피)
# ─────────────────────────────────────────────────────────────────────────────
def test_train_logic() -> None:
    print("[3] train_grpo_action — L2-a mask / L2-d instruction / proxy heuristics")
    # train_grpo_action.py 는 상단에서 torch/transformers/trl/datasets/peft 를 임포트한다.
    # 스모크는 순수 함수만 필요하므로 무거운 의존을 stub 로 주입.
    for name in ["torch", "transformers", "trl", "datasets", "peft", "PIL", "PIL.Image",
                 "qwen_vl_utils", "yaml", "numpy"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    # 필요한 속성 최소 stub
    sys.modules["torch"].bfloat16 = "bf16"
    sys.modules["torch"].no_grad = lambda: (lambda f: f)   # @torch.no_grad() 모듈 데코레이터용
    sys.modules["datasets"].Dataset = object
    sys.modules["datasets"].Image = object
    for attr in ("AutoModelForImageTextToText", "AutoProcessor"):
        setattr(sys.modules["transformers"], attr, object)
    sys.modules["transformers"].TrainerCallback = object
    for attr in ("GRPOConfig", "GRPOTrainer"):
        setattr(sys.modules["trl"], attr, object)
    for attr in ("LoraConfig", "PeftModel", "get_peft_model"):
        setattr(sys.modules["peft"], attr, object)
    sys.modules["PIL"].Image = types.SimpleNamespace(new=lambda *a, **k: None, open=lambda *a, **k: None)
    sys.modules["PIL.Image"].new = lambda *a, **k: None

    try:
        tg = load_module(TRAIN, "tg")
    except Exception as e:
        check("train_grpo_action import", False, f"{type(e).__name__}: {e}")
        return

    # L2-d: 시스템 프롬프트에 재진술 금지 지시
    check("L2-d belief anti-restatement in system prompt",
          "restatement" in tg.JOINT_SYSTEM_PROMPT.lower() or "NOT a restatement" in tg.JOINT_SYSTEM_PROMPT)
    check("masked system prompt drops frame availability",
          "unavailable" in tg.JOINT_SYSTEM_PROMPT_MASKED.lower())

    # L2-a: 결정론적 마스킹
    tg.MASK_FRAME_PROB = 0.0
    check("mask off → never masks", not tg._mask_this_sample("any_id"))
    tg.MASK_FRAME_PROB = 1.0
    check("mask 1.0 → always masks", tg._mask_this_sample("any_id"))
    tg.MASK_FRAME_PROB = 0.2
    ids = [f"s{i}" for i in range(2000)]
    frac = sum(tg._mask_this_sample(i) for i in ids) / len(ids)
    check("mask 0.2 → ~20% deterministic", 0.16 < frac < 0.24, f"frac={frac:.3f}")
    check("mask is deterministic per id",
          tg._mask_this_sample("s1") == tg._mask_this_sample("s1"))
    tg.MASK_FRAME_PROB = 0.0

    # 프록시: history_reference_rate
    r_hi = tg.history_reference_rate("I already took the knife and washed the tomato",
                                     "Previously completed actions: take knife -> wash tomato.")
    r_lo = tg.history_reference_rate("The fridge looks closed and I should open it",
                                     "Previously completed actions: take knife -> wash tomato.")
    check("history_reference_rate high when reasoning cites history", r_hi > 0.3, f"hi={r_hi:.2f}")
    check("history_reference_rate low when it ignores history", r_lo < r_hi, f"lo={r_lo:.2f}")

    # 프록시: belief 재진술 flag
    check("belief restatement detected", tg.belief_restatement_flag("cut the tomato", "cut", "tomato"))
    check("global belief not flagged", not tg.belief_restatement_flag("prepare a salad", "cut", "tomato"))

    # build_joint_conversation: L2-a 마스킹 샘플이 blank 이미지/masked 프롬프트 사용
    import random as _r
    tg.MASK_FRAME_PROB = 1.0
    tg.BLANK_IMAGE_PATH = "/tmp/_blank.jpg"
    ex = {
        "topk_actions_with_score": [{"verb": "cut", "noun": "tomato", "likelihood": 0.3, "rank": 1},
                                    {"verb": "take", "noun": "plate", "likelihood": 0.2, "rank": 2}],
        "image_path": "/real/frame.jpg", "frame_id": "s1", "episode_id": "V",
        "memory_context": "Previously completed actions: take knife.",
        "frame_meta": {"n_frames": 4, "offsets_sec": [4.0, 2.67, 1.33, 0.0]},
    }
    conv = tg.build_joint_conversation(ex, top_k=5, rng=_r.Random(0))
    check("masked sample uses blank image", conv["image"] == "/tmp/_blank.jpg")
    check("masked sample flagged", conv["frame_masked"] is True)
    check("masked sample uses masked system prompt",
          "unavailable" in conv["prompt"][0]["content"].lower())
    tg.MASK_FRAME_PROB = 0.0
    conv2 = tg.build_joint_conversation(ex, top_k=5, rng=_r.Random(0))
    check("unmasked 4f sample keeps real image", conv2["image"] == "/real/frame.jpg")
    check("unmasked 4f announces grid", "grid" in conv2["prompt"][1]["content"].lower()
          or "four" in conv2["prompt"][1]["content"].lower())
    check("no future leak in joint prompt",
          "future" not in json.dumps(conv2["prompt"]).lower())


def main() -> None:
    print("=" * 68)
    print("F0 final plan v2 — smoke test (pure logic, no GPU)")
    print("=" * 68)
    blocks = [("memory", test_memory_logic), ("convert", test_convert_logic),
              ("train", test_train_logic)]
    for label, fn in blocks:
        try:
            fn()
        except ModuleNotFoundError as e:
            print(f"  [SKIP] {label}: missing dependency ({e.name}) — 학습 서버에서 재실행")
        except Exception as e:
            global FAIL
            FAIL += 1
            print(f"  [FAIL] {label} raised {type(e).__name__}: {e}")
    print("-" * 68)
    print(f"passed={PASS}  failed={FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
