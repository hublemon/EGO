# NOTE: F0 (WM-only GRPO) 트랙의 **실제로 검증된 코드**를 그대로 옮긴 것이다.
# 2026-07-17 F0 final 결과(docs/experiments/2026-07-17_f0_final.md)를 만든 코드가 이 파일이다.
# 패키지 구조에 맞춘 추측성 리팩터는 하지 않았다 — 재검증 없이 쪼개면 결과 재현이 깨진다.
# 실행은 configs/step2/f0_final_wm_only.yaml 과 scripts/step2/train_f0_final.sh 참조.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage GRPO LoRA training for Qwen2.5-VL on EK100 frames + V-JEPA2 AC top-K outputs.

V-JEPA2 top-1 정렬 방식 (GT 아님). 참고 스크립트를 trl 1.5.1 / transformers 5.9.0 에 맞게 적응:
  - 멀티모달 입력: prompt 는 conversational(문자열 content) + 별도 "image"(PIL) 컬럼.
    trl 이 내부에서 chat template 을 적용하고 prepare_multimodal_messages 로 이미지를 주입한다.
    (사전 템플릿 문자열을 넣으면 trl 1.5.1 이 ValueError 를 던짐 — grpo_trainer.py:1853)
  - GRPOConfig 에서 max_prompt_length 제거 (trl 1.5.1 에 없음).
  - flash_attn 미설치 → attn_implementation 기본값 "sdpa".

출력 포맷 (양 stage 공통):
    {"action_index": 1, "verb": "...", "noun": "...", "reason": "..."}

Stage 1 (noun): 선택한 action 의 noun 이 V-JEPA2 top-1 noun 과 같으면 보상.
Stage 2 (action): 선택한 (verb,noun) 이 V-JEPA2 AC top-1 action 과 같으면 보상.

입력 JSONL: data/grpo_dataset/grpo_train.jsonl
    image_path, topk_nouns[{noun,score}], topk_actions[{verb,noun,score}], memory_context(str), ...
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, Image as DSImage
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


BASE_SYSTEM_PROMPT = """You are an embodied long-horizon action selection model.

You receive:
1. An egocentric frame image.
2. V-JEPA2 top-K noun predictions.
3. V-JEPA2 top-K action predictions.
4. Optional memory context from previous VLM decisions.

You must choose exactly ONE action from the provided V-JEPA2 top-K action candidates.
Do not invent new nouns, verbs, or actions.
Keep the reason short.
"""

STAGE1_INSTRUCTION = """Training stage:
STAGE 1 - NOUN-ONLY ALIGNMENT THROUGH ACTION SELECTION.

Your job:
Choose exactly ONE action from the V-JEPA2 top-K action candidates.

Although you output a full action, the reward in this stage depends only on the noun
of your selected action. The preferred noun is the V-JEPA2 top-1 noun.

Output JSON only, with no markdown:
{"action_index": 1, "verb": "...", "noun": "...", "reason": "..."}

Rules:
- action_index must be an integer from 1 to K.
- verb and noun must exactly match the candidate at action_index.
- The selected action must be one of the V-JEPA2 top-K action candidates.
- Do not invent a new noun or verb.
- Prefer an action whose noun matches the V-JEPA2 top-1 noun.
"""

STAGE2_INSTRUCTION = """Training stage:
STAGE 2 - ACTION-PAIR ALIGNMENT.

Your job:
Choose exactly ONE action from the V-JEPA2 top-K action candidates.

The reward in this stage depends on whether your selected (verb, noun) action equals
the V-JEPA2 AC top-1 action.

Output JSON only, with no markdown:
{"action_index": 1, "verb": "...", "noun": "...", "reason": "..."}

Rules:
- action_index must be an integer from 1 to K.
- verb and noun must exactly match the candidate at action_index.
- The selected action must be one of the V-JEPA2 top-K action candidates.
- Do not invent a new noun or verb.
- Prefer the V-JEPA2 AC top-1 action.
"""


STAGE_GT_INSTRUCTION = """Training stage:
NEXT-ACTION SELECTION (align to the action that actually happens next).

Your job:
Look carefully at the egocentric frame and choose exactly ONE action from the
V-JEPA2 top-K action candidates that is MOST LIKELY the real next action.

Output JSON only, no markdown:
{"action_index": N, "verb": "...", "noun": "...", "reason": "..."}

Rules:
- action_index is the integer index (1..K) of your chosen candidate in the list above.
- verb and noun must EXACTLY match the candidate at that index.
- The candidates are NOT sorted by correctness — do not just pick index 1.
  Use the image, the per-candidate scores, the task goal, and memory context to decide.
- The selected action must be one of the listed candidates. Do not invent new verbs/nouns.
- Keep the reason short (one clause).
"""


# ===================== 실험 4/5b: think-format (reasoning 먼저) =====================
# 입력: verb top-5 + noun top-5 분리 (score 제거, 셔플). VLM 이 둘을 조합.
# 출력: <think>...</think><action>{"verb","noun"}</action>  — think 가 답의 원인이 되도록.

THINK_SYSTEM_PROMPT = """You are an embodied action planner with access to an egocentric video frame.

You receive:
1. A first-person video frame.
2. Verb candidates from a world model (V-JEPA2).
3. Noun candidates from a world model (V-JEPA2).

Your job: reason step by step, then select the best next action.

Output format (strictly follow):
<think>
Step 1. What do you observe in the frame?
Step 2. What still needs to be done toward the task goal?
Step 3. Which verb+noun combination best fits? Evaluate candidates explicitly.
</think>
<action>{"verb": "...", "noun": "..."}</action>

Rules:
- verb MUST be from the verb candidates list.
- noun MUST be from the noun candidates list.
- the think block MUST come before the action block.
- do not skip any step in think.
"""

THINK_INSTRUCTION = """Reason about the egocentric frame and choose the single most likely NEXT action
by composing one verb candidate with one noun candidate."""


# ── joint action top-5 포맷 (F0 v2) ────────────────────────────────────────────
# 25조합(verb-5 × noun-5) 대신 WM 의 action head 가 실제로 낸 joint top-5 를 그대로 후보로.
# 근거: held-out 실측 — 25조합 논리상한 0.666 vs joint top-5 포함률 0.620 (−4.6pp 뿐)인데,
# Run1 step1000 출력의 52.6% 가 joint top-5 **밖** 조합이었고 그 구간 정확도는 0.027.
# → 선택지를 5개로 좁히면 그 질량을 회수. 상한(=WM top-1 정확도 0.374)은 불변.
JOINT_SYSTEM_PROMPT = """You are an embodied agent reasoning about your own ongoing activity
from a first-person view.

You receive:
1. A first-person video frame.
2. Your recent action history.
3. Five candidate next-actions from a world model (V-JEPA2), in no particular order.

Think freely in <reasoning> — there is no required structure. But your reasoning should arrive at
a judgement about what you are currently trying to accomplish, and that judgement should be what
decides your choice among the five candidates.

State that judgement in <task_belief>: the overall goal your recent actions are building toward,
not the next single action. Then choose.

You MUST emit exactly three tags, in this order, and nothing outside them.

<reasoning>
... free reasoning ...
</reasoning>
<task_belief>the overall goal you are working toward</task_belief>
<action>{"verb": "...", "noun": "..."}</action>

Rules:
- start your reply with the literal string "<reasoning>".
- the action MUST be one of the five candidates, copied exactly (both verb and noun).
- do not invent new verb+noun combinations.
"""

JOINT_INSTRUCTION = """Reason about the egocentric frame and your action history, infer what you are
trying to accomplish, and choose the single most likely NEXT action from the five candidates below."""


# parse 포맷 디스패치: main() 에서 reward_mode 에 따라 "json" 또는 "think" 로 설정.
PARSE_FORMAT = "json"
HIDE_SCORES = False   # main() 에서 --hide_scores 로 설정 (후보 점수 노출 제거 → rank 자명해 차단)
THINK_REWARD_MODES = {"think_format", "think_ranking", "think_gt",
                      "think_wm_rank_fix", "think_gt_combo",
                      "think_noun_gt", "think_noun_combo",
                      "think_gt_final", "wm_likelihood", "wm_likelihood_p3",
                      "wm_likelihood_joint"}
# joint action top-5 포맷을 쓰는 모드 (프롬프트 빌더 분기용)
JOINT_REWARD_MODES = {"wm_likelihood_joint"}
JSON_REWARD_MODES = {"wm_ranking", "noun_ranking", "action_ranking_from_noun"}
RANK_REWARD_TABLE = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.2, 5: 0.1}
# GT-free 모드: 학습 신호는 물론 데이터셋 필터에도 GT 를 쓰지 않는다 (Run 1~).
GT_FREE_REWARD_MODES = {"wm_likelihood", "wm_likelihood_p3", "wm_likelihood_joint"}
# wm_likelihood reward 정규화 방식 — main() 에서 --wm_likelihood_norm 으로 설정.
#   "candidate": likelihood / sum(top-k likelihood). 후보 5개가 실제 선택지이므로 조건부 분포가
#                맞는 target 이고, raw softmax(~3800 클래스)는 median std 0.015 로 스케일이 작아
#                format(0.15)/gate(0.5) 항에 묻힌다 (4,998행 실측: renorm median std 0.147).
#   "raw":       probe softmax 값 그대로.
WM_LIK_NORM = "candidate"

# P1 sharpening 온도 — main()/eval 에서 --wm_likelihood_temp 로 설정.
#   각 후보 likelihood 를 p^(1/T) 로 변환 후 재정규화. T<1 이면 top 후보로 뾰족해져
#   reward 가 argmax(=WM-copy) 쪽으로 정렬 → 정확도 회복. T=1 이면 Run 1 과 동일.
#   "분포매칭 reward 가 WM-copy 보다 낮다"(G2 구조적 한계) 를 직접 완화/정량화하는 레버.
WM_LIK_TEMP = 1.0

# memory-off (misalignment ③) — main()/eval 에서 --no_memory 로 설정.
#   True 면 프롬프트의 memory_context 를 GT-유래 과거 action 대신 공란 문구로 대체.
#   학습·평가 동일 경로(make_conversation)를 타므로 플래그 하나로 양쪽 일관 적용.
NO_MEMORY = False


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def normalize_score(item: Dict[str, Any]) -> float:
    v = item.get("score", item.get("prob", item.get("probability", item.get("likelihood", 0.0))))
    return float(v) if v is not None else 0.0


def normalize_topk_nouns(topk_nouns: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(topk_nouns[:top_k], 1):
        noun = str(item.get("noun", "")).strip()
        if not noun:
            continue
        out.append({"index": i, "noun": noun, "score": normalize_score(item)})
    return out


def normalize_topk_actions(topk_actions: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(topk_actions[:top_k], 1):
        verb = str(item.get("verb", "")).strip()
        noun = str(item.get("noun", "")).strip()
        if not verb or not noun:
            continue
        out.append({
            "index": i, "verb": verb, "noun": noun,
            "action": f"{verb} {noun}", "score": normalize_score(item),
        })
    return out


def format_topk_nouns(topk_nouns: List[Dict[str, Any]]) -> str:
    if HIDE_SCORES:
        return "\n".join(f'{it["index"]}. {it["noun"]}' for it in topk_nouns)
    return "\n".join(f'{it["index"]}. {it["noun"]} ({it["score"]:.3f})' for it in topk_nouns)


def format_topk_actions(topk_actions: List[Dict[str, Any]]) -> str:
    if HIDE_SCORES:
        return "\n".join(
            f'{it["index"]}. ({it["verb"]}, {it["noun"]}) -> {it["verb"]} {it["noun"]}'
            for it in topk_actions
        )
    return "\n".join(
        f'{it["index"]}. ({it["verb"]}, {it["noun"]}) -> {it["verb"]} {it["noun"]} ({it["score"]:.3f})'
        for it in topk_actions
    )


def build_problem_text(example: Dict[str, Any], disp_nouns: List[Dict[str, Any]],
                       disp_actions: List[Dict[str, Any]], stage: str,
                       hide_top1_hint: bool,
                       vjepa_top1_noun: str, vjepa_top1_action: str) -> str:
    """disp_nouns/disp_actions 는 (옵션 셔플 후) 화면에 보일 순서·index 가 매겨진 후보.
    hide_top1_hint=True 면 정답 노출 라인을 제거 (reward 포화/복사 collapse 방지)."""
    task_goal = example.get("task_goal", "unknown task")
    episode_id = example.get("episode_id", "")
    frame_id = example.get("frame_id", "")
    memory_context = ("No previous memory context is available." if NO_MEMORY
                      else (example.get("memory_context") or "No previous memory context is available."))
    k = len(disp_actions)

    stage_instruction = {
        "noun": STAGE1_INSTRUCTION,
        "action": STAGE2_INSTRUCTION,
        "gt": STAGE_GT_INSTRUCTION,
    }[stage]

    hint = ""
    if not hide_top1_hint and stage in ("noun", "action"):
        hint = (f"\nV-JEPA2 top-1 noun for stage-1 alignment:\n{vjepa_top1_noun}\n"
                f"\nV-JEPA2 AC top-1 action for stage-2 alignment:\n{vjepa_top1_action}\n")

    return f"""{stage_instruction}

Task goal:
{task_goal}

Episode:
{episode_id}

Frame:
{frame_id}

V-JEPA2 noun predictions (score = V-JEPA2 confidence):
{format_topk_nouns(disp_nouns)}

Action candidates (score = V-JEPA2 confidence):
{format_topk_actions(disp_actions)}
{hint}
Memory context:
{memory_context}

Now choose exactly one action (by index) from the {k} candidates above.
"""


def _common_reward_columns(example: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    """reward fn 들이 공통으로 참조하는 rank-보존 컬럼 (json 직렬화)."""
    actions_ws = (example.get("topk_actions_with_score") or [])[:top_k]
    nouns_ws = (example.get("topk_nouns_with_score") or [])[:top_k]
    return {
        "topk_actions_with_score": json.dumps(actions_ws, ensure_ascii=False),
        "topk_nouns_with_score": json.dumps(nouns_ws, ensure_ascii=False),
        "gt_verb": str(example.get("gt_verb", "")),
        "gt_noun": str(example.get("gt_noun", "")),
    }


def build_think_conversation(example: Dict[str, Any], top_k: int,
                             rng: random.Random) -> Dict[str, Any]:
    """실험 4/5b: verb top-5 + noun top-5 분리 입력 (score 제거 + 셔플), think-format 출력."""
    verbs = [str(v).strip() for v in (example.get("topk_verbs") or []) if str(v).strip()][:top_k]
    nouns = [str(n.get("noun", "")).strip() for n in (example.get("topk_nouns") or [])]
    nouns = [n for n in nouns if n][:top_k]
    # 중복 제거 (verb 후보는 action 에서 파생 시 중복 가능) + 셔플 (score/순서 단서 제거)
    verbs = list(dict.fromkeys(verbs)); nouns = list(dict.fromkeys(nouns))
    disp_verbs = rng.sample(verbs, len(verbs))
    disp_nouns = rng.sample(nouns, len(nouns))

    task_goal = example.get("task_goal", "unknown task")
    memory_context = ("No previous memory context is available." if NO_MEMORY
                      else (example.get("memory_context") or "No previous memory context is available."))
    problem_text = f"""{THINK_INSTRUCTION}

Task goal:
{task_goal}

Verb candidates (unordered):
{", ".join(disp_verbs)}

Noun candidates (unordered):
{", ".join(disp_nouns)}

Memory context:
{memory_context}

Reason step by step in <think>, then output <action>{{"verb": "...", "noun": "..."}}</action>.
"""
    prompt = [
        {"role": "system", "content": THINK_SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    out = {
        "prompt": prompt,
        "image": example["image_path"],
        "stage": "think",
        "sample_id": str(example.get("frame_id", "")),
        "episode_id": str(example.get("episode_id", "")),
        # think candidate reward 용: 화면에 보인 후보 이름 목록
        "topk_verbs": json.dumps(disp_verbs, ensure_ascii=False),
        "topk_nouns": json.dumps(disp_nouns, ensure_ascii=False),
        # 아래는 think 에선 안 쓰지만 schema 일관성 위해 빈 값 채움
        "valid_nouns_json": "[]",
        "valid_actions_json": "[]",
        "vjepa_top1_noun": "", "vjepa_top1_verb": "", "vjepa_top1_action_noun": "",
    }
    out.update(_common_reward_columns(example, top_k))
    return out


def build_joint_conversation(example: Dict[str, Any], top_k: int,
                             rng: random.Random) -> Dict[str, Any]:
    """F0 v2: WM joint action top-5 를 그대로 후보로 제시 (5지선다). score 제거 + 셔플.

    think_convergence(P4) 는 topk_nouns 를 참조하므로 5개 후보의 noun 집합을 넣어 준다.
    wm_likelihood(P1) 는 topk_actions_with_score 를 rank 보존으로 받으므로 변경 없음.
    """
    acts = (example.get("topk_actions_with_score") or [])[:top_k]
    pairs = []
    for a in acts:
        v, n = str(a.get("verb", "")).strip(), str(a.get("noun", "")).strip()
        if v and n and (v, n) not in pairs:
            pairs.append((v, n))
    disp = rng.sample(pairs, len(pairs))   # 순서 단서 제거 (rank 자명해 차단)

    # task_goal 은 **의도적으로 프롬프트에 넣지 않는다.**
    # 실측: select_train.py 가 만드는 task_goal = "그 비디오의 첫 narration" = 사실상 video ID.
    #   비디오당 종류 1개(80/80) · GT 예측력 1.1% · 샘플은 중앙값 9.1분 뒤 시점.
    # 목표 정보는 없고 video 식별 지름길만 열어준다 → 제거하고 모델이 history 로 추론하게 한다.
    memory_context = ("No previous action history is available." if NO_MEMORY
                      else (example.get("memory_context") or "No previous action history is available."))
    cand_lines = "\n".join(f'- {{"verb": "{v}", "noun": "{n}"}}' for v, n in disp)
    problem_text = f"""{JOINT_INSTRUCTION}

Action history:
{memory_context}

Action candidates (unordered):
{cand_lines}

Begin your reply with "<reasoning>". Reason inside <reasoning></reasoning>, then state the overall goal in
<task_belief></task_belief>, then output <action>{{"verb": "...", "noun": "..."}}</action>
copying exactly one candidate above.
"""
    prompt = [
        {"role": "system", "content": JOINT_SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    out = {
        "prompt": prompt,
        "image": example["image_path"],
        "stage": "joint",
        "sample_id": str(example.get("frame_id", "")),
        "episode_id": str(example.get("episode_id", "")),
        # gate 용: 화면에 보인 (verb,noun) 쌍 목록
        "topk_actions_display": json.dumps([{"verb": v, "noun": n} for v, n in disp],
                                           ensure_ascii=False),
        # judge_reasoning_curve.py 용 — **실제로 프롬프트에 보인** history 를 그대로 기록
        # (--no_memory 면 공란 문구가 들어감). 리워드 함수는 이 컬럼을 읽지 않는다.
        "memory_context": memory_context,
        # P4 용: 후보 5개에 등장하는 noun 집합 (중복 제거, 표시 순서)
        "topk_nouns": json.dumps(list(dict.fromkeys(n for _, n in disp)), ensure_ascii=False),
        "topk_verbs": json.dumps(list(dict.fromkeys(v for v, _ in disp)), ensure_ascii=False),
        "valid_nouns_json": "[]",
        "valid_actions_json": "[]",
        "vjepa_top1_noun": "", "vjepa_top1_verb": "", "vjepa_top1_action_noun": "",
    }
    out.update(_common_reward_columns(example, top_k))
    return out


def make_conversation(example: Dict[str, Any], stage: str, top_k: int,
                      shuffle_candidates: bool = False, hide_top1_hint: bool = False,
                      rng: Optional[random.Random] = None,
                      reward_mode: Optional[str] = None) -> Dict[str, Any]:
    """trl 1.5.1 멀티모달 포맷: prompt(conversational, 문자열 content) + "image" 컬럼.
    이미지는 경로만 저장 → cast_column("image", Image()) 로 lazy decode.

    reward_mode in THINK_REWARD_MODES → think-format (verb/noun 분리 입력) 로 분기.
    그 외 → 기존 JSON action 선택 포맷.

    shuffle_candidates: 후보 표시 순서를 섞고 index 를 1..K 로 재부여 (index 복사 collapse 방지).
    hide_top1_hint:     프롬프트에서 V-JEPA2 top-1 정답 노출 라인 제거.
    """
    rng = rng or random
    if reward_mode in JOINT_REWARD_MODES:
        return build_joint_conversation(example, top_k, rng)
    if reward_mode in THINK_REWARD_MODES:
        return build_think_conversation(example, top_k, rng)

    nouns = normalize_topk_nouns(example["topk_nouns"], top_k)
    actions = normalize_topk_actions(example["topk_actions"], top_k)
    # 원래 rank-1 (셔플 전) — vjepa_top1 정렬 reward 용
    vjepa_top1_noun = nouns[0]["noun"] if nouns else ""
    vjepa_top1_action = actions[0] if actions else {}
    vjepa_top1_action_str = vjepa_top1_action.get("action", "UNKNOWN") if vjepa_top1_action else "UNKNOWN"

    # 표시용: 옵션 셔플 후 index 1..K 재부여
    on = list(nouns); oa = list(actions)
    if shuffle_candidates:
        on = rng.sample(on, len(on)); oa = rng.sample(oa, len(oa))
    disp_nouns = [{**x, "index": i} for i, x in enumerate(on, 1)]
    disp_actions = [{**x, "index": i} for i, x in enumerate(oa, 1)]

    problem_text = build_problem_text(example, disp_nouns, disp_actions, stage,
                                      hide_top1_hint, vjepa_top1_noun, vjepa_top1_action_str)
    prompt = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]

    out = {
        "prompt": prompt,
        "image": example["image_path"],
        "stage": stage,
        "sample_id": str(example.get("frame_id", "")),
        "episode_id": str(example.get("episode_id", "")),
        "topk_verbs": "[]", "topk_nouns": "[]",   # think 전용 컬럼 (schema 일관성)
        "valid_nouns_json": json.dumps(disp_nouns, ensure_ascii=False),
        "valid_actions_json": json.dumps(disp_actions, ensure_ascii=False),
        "vjepa_top1_noun": vjepa_top1_noun,
        "vjepa_top1_verb": vjepa_top1_action.get("verb", ""),
        "vjepa_top1_action_noun": vjepa_top1_action.get("noun", ""),
    }
    out.update(_common_reward_columns(example, top_k))
    return out


# ------------------------- completion parsing -------------------------

def extract_text_from_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        chunks: List[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            chunks.append(str(c["text"]))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(completion)


def extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    text = extract_text_from_completion(text).strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_int_index(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_action_json(completion: Any) -> Tuple[Optional[int], Optional[str], Optional[str], str]:
    obj = extract_json_object(completion)
    if obj is None:
        return None, None, None, ""
    action_index = parse_int_index(obj.get("action_index", obj.get("index")))
    verb = obj.get("verb")
    noun = obj.get("noun")
    reason = obj.get("reason", "")
    if not isinstance(verb, str):
        verb = None
    if not isinstance(noun, str):
        noun = None
    return action_index, verb.strip() if verb else None, noun.strip() if noun else None, str(reason)


def extract_think_block(completion: Any) -> Optional[str]:
    """<think> 또는 <reasoning> 블록. 둘 다 받는 이유:
    Qwen3-VL 토크나이저에서 <think>/</think> 는 **예약된 단일 토큰**(151667/151668)이고
    Instruct 변형은 이 토큰을 내지 않도록 튜닝돼 있어 아무리 지시해도 생성하지 않는다
    (실측: 프롬프트로 강제해도 0회, 대신 태그 없는 산문을 냄 → think='' → P4 영구 0).
    그래서 F0 v2 는 <reasoning> 을 쓴다. <think> 은 Qwen2.5 기반 구 run 하위호환용."""
    text = extract_text_from_completion(completion)
    m = re.search(r"<(?:think|reasoning)>(.*?)</(?:think|reasoning)>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_action_from_think_format(completion: Any) -> Tuple[Optional[str], Optional[str], str]:
    """<action>{"verb":..,"noun":..}</action> 에서 verb/noun 추출. think 텍스트도 반환."""
    text = extract_text_from_completion(completion)
    think = extract_think_block(completion) or ""
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if not m:
        return None, None, think
    raw = m.group(1).strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    jm = re.search(r"\{.*\}", raw, re.DOTALL)
    if jm:
        raw = jm.group(0)
    try:
        obj = json.loads(raw)
        verb = obj.get("verb"); noun = obj.get("noun")
        verb = verb.strip() if isinstance(verb, str) else None
        noun = noun.strip() if isinstance(noun, str) else None
        return verb, noun, think
    except Exception:
        return None, None, think


def parse_vn(completion: Any) -> Tuple[Optional[str], Optional[str]]:
    """PARSE_FORMAT 에 따라 (verb, noun) 추출 — reward fn 공용."""
    if PARSE_FORMAT == "think":
        v, n, _ = parse_action_from_think_format(completion)
        return v, n
    _, v, n, _ = parse_action_json(completion)
    return v, n


def get_valid_action_by_index(valid_actions: List[Dict[str, Any]], idx: Optional[int]) -> Optional[Tuple[str, str]]:
    if idx is None:
        return None
    for item in valid_actions:
        if int(item["index"]) == idx:
            return item["verb"], item["noun"]
    return None


# ------------------------- rewards -------------------------

def format_reward(completions, **kwargs) -> List[float]:
    rewards = []
    for comp in completions:
        ai, verb, noun, _ = parse_action_json(comp)
        rewards.append(0.15 if ai is not None and verb and noun else 0.0)
    return rewards


def action_candidate_consistency_reward(completions, valid_actions_json: List[str], **kwargs) -> List[float]:
    rewards = []
    for comp, valid_json in zip(completions, valid_actions_json):
        ai, pv, pn, _ = parse_action_json(comp)
        valid_actions = json.loads(valid_json)
        valid_pairs = {(x["verb"], x["noun"]) for x in valid_actions}
        indexed_pair = get_valid_action_by_index(valid_actions, ai)
        r = 0.0
        if pv is not None and pn is not None and (pv, pn) in valid_pairs:
            r += 0.20
        else:
            r -= 0.20
        if indexed_pair is not None and indexed_pair == (pv, pn):
            r += 0.30
        elif ai is not None:
            r -= 0.10
        rewards.append(r)
    return rewards


def stage1_top1_noun_reward(completions, vjepa_top1_noun: List[str], **kwargs) -> List[float]:
    rewards = []
    for comp, t1 in zip(completions, vjepa_top1_noun):
        _, _, pn, _ = parse_action_json(comp)
        rewards.append(1.0 if pn == t1 and t1 else 0.0)
    return rewards


def stage2_top1_action_reward(completions, vjepa_top1_verb: List[str], vjepa_top1_action_noun: List[str], **kwargs) -> List[float]:
    rewards = []
    for comp, tv, tn in zip(completions, vjepa_top1_verb, vjepa_top1_action_noun):
        _, pv, pn, _ = parse_action_json(comp)
        rewards.append(1.0 if pv == tv and pn == tn and tv and tn else 0.0)
    return rewards


def gt_accuracy_reward(completions, gt_verb: List[str], gt_noun: List[str], **kwargs) -> List[float]:
    """EK100 GT 정렬: verb +0.25, noun +0.35, 둘 다 +0.40 (최대 1.0).
    V-JEPA2 top-1 과 GT 가 34% 불일치 → 후보 중 GT 를 고르려면 영상/문맥 추론 필요 (비자명)."""
    rewards = []
    for comp, gv, gn in zip(completions, gt_verb, gt_noun):
        _, pv, pn, _ = parse_action_json(comp)
        r = 0.0
        vok = bool(pv) and bool(gv) and pv == gv
        nok = bool(pn) and bool(gn) and pn == gn
        if vok:
            r += 0.25
        if nok:
            r += 0.35
        if vok and nok:
            r += 0.40
        rewards.append(r)
    return rewards


# ------------------------- rewards: think-format (실험 4/5b) -------------------------

def format_reward_think(completions, **kwargs) -> List[float]:
    """<think>...</think> 와 <action>...</action> 가 모두 존재하면 0.15."""
    rewards = []
    for comp in completions:
        text = extract_text_from_completion(comp)
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        has_action = bool(re.search(r"<action>.*?</action>", text, re.DOTALL))
        rewards.append(0.15 if (has_think and has_action) else 0.0)
    return rewards


def think_quality_reward(completions, topk_verbs, topk_nouns, **kwargs) -> List[float]:
    """think 블록이 실질 추론을 담는지: (1) 20단어 이상 + 후보 단어 1개 이상 언급 → 0.20,
    (2) 10단어 이상 → 0.08, 그 외 0.0. reasoning 이 답에 기여하도록 유도."""
    rewards = []
    for comp, tv, tn in zip(completions, topk_verbs, topk_nouns):
        think = extract_think_block(comp)
        if not think:
            rewards.append(0.0)
            continue
        tl = think.lower()
        wc = len(tl.split())
        try:
            cands = [c.lower() for c in json.loads(tv)] + [c.lower() for c in json.loads(tn)]
        except Exception:
            cands = []
        mentions = any(c and c in tl for c in cands)
        if wc >= 20 and mentions:
            rewards.append(0.20)
        elif wc >= 10:
            rewards.append(0.08)
        else:
            rewards.append(0.0)
    return rewards


def candidate_reward_think(completions, topk_verbs, topk_nouns, **kwargs) -> List[float]:
    """verb ∈ topk_verbs AND noun ∈ topk_nouns → 0.50 / 하나만 → 0.10 / 둘다 아님 → -0.20."""
    rewards = []
    for comp, tv, tn in zip(completions, topk_verbs, topk_nouns):
        verb, noun, _ = parse_action_from_think_format(comp)
        if not verb or not noun:
            rewards.append(0.0)
            continue
        try:
            valid_v = set(json.loads(tv)); valid_n = set(json.loads(tn))
        except Exception:
            valid_v, valid_n = set(), set()
        in_v = verb in valid_v; in_n = noun in valid_n
        if in_v and in_n:
            rewards.append(0.50)
        elif in_v or in_n:
            rewards.append(0.10)
        else:
            rewards.append(-0.20)
    return rewards


def gt_accuracy_reward_think(completions, gt_verb, gt_noun, **kwargs) -> List[float]:
    """think-format 출력에 대한 GT 정렬 (verb +0.25, noun +0.35, 둘다 +0.40)."""
    rewards = []
    for comp, gv, gn in zip(completions, gt_verb, gt_noun):
        verb, noun, _ = parse_action_from_think_format(comp)
        r = 0.0
        vok = bool(verb) and bool(gv) and verb == gv
        nok = bool(noun) and bool(gn) and noun == gn
        if vok:
            r += 0.25
        if nok:
            r += 0.35
        if vok and nok:
            r += 0.40
        rewards.append(r)
    return rewards


# ------------------------- rewards: 개선 (실험 7, collapse 방지) -------------------------

def _noun_fuzzy_match(pred: str, gt: str) -> bool:
    """EK100 계층 noun 레이블 퍼지 매칭.
    'towel:kitchen' ↔ 'towel', 'milk:soy' ↔ 'milk' 등 ':' 기준 base가 같으면 true.
    'cup' ≠ 'cupboard' 등 substring false-positive 방지를 위해 base 분리 비교만 사용.
    실제 데이터에서 영향 받는 샘플은 ~1/4998로 매우 드물지만 올바른 신호를 줌.
    """
    if not pred or not gt:
        return False
    return pred.split(":")[0] == gt.split(":")[0]


def candidate_gate_reward_think(completions, topk_verbs, topk_nouns, **kwargs) -> List[float]:
    """후보 보상을 '게이트'로: 유효(verb∈ & noun∈) → 0.0, 무효 → -0.5.
    '유효하면 +0.5' 를 제거해 정답(gt_accuracy)이 그룹 내 주신호가 되게 함 (collapse 완화)."""
    rewards = []
    for comp, tv, tn in zip(completions, topk_verbs, topk_nouns):
        verb, noun, _ = parse_action_from_think_format(comp)
        if not verb or not noun:
            rewards.append(-0.5)
            continue
        try:
            valid_v = set(json.loads(tv)); valid_n = set(json.loads(tn))
        except Exception:
            valid_v, valid_n = set(), set()
        rewards.append(0.0 if (verb in valid_v and noun in valid_n) else -0.5)
    return rewards


def format_reward_joint(completions, **kwargs) -> List[float]:
    """F0 v2 구조 리워드: <reasoning> · <task_belief> · <action> 3태그의 **존재와 순서**만 검사.

    ⚠️ 계약: **belief 의 '내용'은 읽지 않는다.** 태그가 있는지만 본다.
    (계약은 "어떤 reward 도 belief 텍스트를 채점하지 않는다"이고, 태그 존재는 content 가 아니라
     format 이다. 이 구분이 없으면 무보상 필드의 역설이 생긴다 — belief 에 gradient 압력이 0이면
     출현율이 base 수준(2태그에서도 parse_rate 0.744)에 고정되어 **학습으로 개선되지 않고**,
     B0 가 파싱할 belief 가 없는 샘플이 대량 발생한다.)

      +0.05  <reasoning>...</reasoning> 존재
      +0.05  <task_belief>...</task_belief> 존재 (내용 무관, 비어있지 않기만)
      +0.05  <action>...</action> 존재
      +0.05  순서가 reasoning → task_belief → action
    """
    rewards = []
    for comp in completions:
        text = extract_text_from_completion(comp)
        r = 0.0
        mt = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        mb = re.search(r"<task_belief>(.*?)</task_belief>", text, re.DOTALL)
        ma = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
        if mt and mt.group(1).strip():
            r += 0.05
        if mb and mb.group(1).strip():
            r += 0.05
        if ma and ma.group(1).strip():
            r += 0.05
        if mt and mb and ma and mt.start() < mb.start() < ma.start():
            r += 0.05
        rewards.append(r)
    return rewards


def candidate_gate_reward_joint(completions, topk_actions_display, **kwargs) -> List[float]:
    """F0 v2 게이트: 선택한 (verb,noun) 쌍이 제시된 joint top-5 안에 있으면 0.0, 아니면 -0.5.

    think 포맷의 gate 는 verb∈verb후보 AND noun∈noun후보 만 봤다 — 25조합 공간이라
    'WM 이 낸 적 없는 조합'을 통과시켰다 (Run1 step1000 에서 52.6%, 그 구간 정확도 0.027).
    joint 포맷에선 쌍 자체의 소속을 검사하므로 그 누수가 구조적으로 막힌다."""
    rewards = []
    for comp, disp in zip(completions, topk_actions_display):
        verb, noun, _ = parse_action_from_think_format(comp)
        if not verb or not noun:
            rewards.append(-0.5)
            continue
        try:
            valid = {(str(a.get("verb", "")), str(a.get("noun", ""))) for a in json.loads(disp)}
        except Exception:
            valid = set()
        rewards.append(0.0 if (verb, noun) in valid else -0.5)
    return rewards


def gt_accuracy_reward_think_v2(completions, gt_verb, gt_noun, **kwargs) -> List[float]:
    """정답 비중 강화 (verb +0.4, noun +0.5, 둘다 +0.6, 최대 1.5). 정답이 보상을 지배하도록."""
    rewards = []
    for comp, gv, gn in zip(completions, gt_verb, gt_noun):
        verb, noun, _ = parse_action_from_think_format(comp)
        r = 0.0
        vok = bool(verb) and bool(gv) and verb == gv
        nok = bool(noun) and bool(gn) and noun == gn
        if vok:
            r += 0.4
        if nok:
            r += 0.5
        if vok and nok:
            r += 0.6
        rewards.append(r)
    return rewards


def gt_accuracy_reward_think_v3(completions, gt_verb, gt_noun, **kwargs) -> List[float]:
    """v2 + 퍼지 noun 매칭 (EK100 계층 레이블 대응).
    exact: verb +0.4, noun +0.5, 둘다 +0.6 (max 1.5)
    fuzzy: verb exact + noun fuzzy(base일치) → noun +0.25, bonus +0.3 (max 0.95)
    데이터셋 내 영향 샘플 ~0.1%지만 'milk:soy' vs 'milk' 같은 오채점을 구제함.
    """
    rewards = []
    for comp, gv, gn in zip(completions, gt_verb, gt_noun):
        verb, noun, _ = parse_action_from_think_format(comp)
        r = 0.0
        vok = bool(verb) and bool(gv) and verb == gv
        nok_exact = bool(noun) and bool(gn) and noun == gn
        nok_fuzzy = (not nok_exact) and bool(noun) and bool(gn) and _noun_fuzzy_match(noun, gn)
        if vok:
            r += 0.4
        if nok_exact:
            r += 0.5
        elif nok_fuzzy:
            r += 0.25
        if vok and nok_exact:
            r += 0.6
        elif vok and nok_fuzzy:
            r += 0.3
        rewards.append(r)
    return rewards


# ------------------------- rewards: 실험 9~13 (새 mode용 추가 함수) -------------------------

def noun_gt_reward_think(completions, gt_noun, **kwargs) -> List[float]:
    """2-stage Stage1: 선택 noun == GT noun → +0.5, 불일치 → 0.0. (think format)"""
    rewards = []
    for comp, gn in zip(completions, gt_noun):
        _, noun, _ = parse_action_from_think_format(comp)
        rewards.append(0.5 if (noun and gn and noun == gn) else 0.0)
    return rewards


# ------------------------- rewards: WM ranking (실험 5/6) -------------------------

def wm_ranking_reward(completions, topk_actions_with_score, **kwargs) -> List[float]:
    """선택 (verb,noun) 의 WM action rank 에 따라 차등 (rank1=1.0..rank5=0.1, 후보 밖=-0.2).
    parse_vn 으로 json/think 양 포맷 모두 지원."""
    rewards = []
    for comp, actions_json in zip(completions, topk_actions_with_score):
        verb, noun = parse_vn(comp)
        if not verb or not noun:
            rewards.append(0.0)
            continue
        try:
            actions = json.loads(actions_json)
        except Exception:
            actions = []
        matched = next((a.get("rank") for a in actions
                        if a.get("verb") == verb and a.get("noun") == noun), None)
        rewards.append(RANK_REWARD_TABLE.get(matched, -0.2))
    return rewards


def noun_ranking_reward(completions, topk_nouns_with_score, **kwargs) -> List[float]:
    """선택 noun 의 WM noun rank 에 따라 차등 (rank1=1.0..rank5=0.1, 후보 밖=-0.2). Stage1 용."""
    rewards = []
    for comp, nouns_json in zip(completions, topk_nouns_with_score):
        _, noun = parse_vn(comp)
        if not noun:
            rewards.append(0.0)
            continue
        try:
            nouns = json.loads(nouns_json)
        except Exception:
            nouns = []
        matched = next((n.get("rank") for n in nouns if n.get("noun") == noun), None)
        rewards.append(RANK_REWARD_TABLE.get(matched, -0.2))
    return rewards


# ------------------------- rewards: WM likelihood (Run 1, GT-free) -------------------------

def _likelihood_reward_value(liks: List[Optional[float]], matched_idx: Optional[int]) -> float:
    """matched_idx 후보의 likelihood 를 WM_LIK_NORM 방식으로 reward 로 변환.
    matched 없음(top-k 액션 리스트 밖 조합) → 0.0 (gate 가 후보-외를 별도 감점).
    likelihood null → 0.0."""
    if matched_idx is None:
        return 0.0
    v = liks[matched_idx]
    if v is None:
        return 0.0
    if WM_LIK_NORM == "raw":
        return float(v)
    # candidate 정규화 (+ 선택적 sharpening). T=1 → Run 1 과 동일한 조건부 분포.
    # T<1 → p^(1/T) 로 top 후보에 질량 집중, matched 가 강후보면 reward↑ (WM-copy 정렬).
    if WM_LIK_TEMP != 1.0:
        inv_t = 1.0 / WM_LIK_TEMP
        vals = [float(x) ** inv_t for x in liks if x is not None]
        denom = sum(vals)
        num = float(v) ** inv_t
        return num / denom if denom > 0 else 0.0
    denom = sum(x for x in liks if x is not None)
    return float(v) / denom if denom > 0 else 0.0


def wm_likelihood_reward(completions, topk_actions_with_score, **kwargs) -> List[float]:
    """P1 (주 신호): 선택 (verb,noun) 의 WM probe likelihood 를 연속값 reward 로.
    rank bucket(RANK_REWARD_TABLE) 폐기 — reward 는 WM 의 예측 분포 그 자체에서 산출.
    parse 실패 → 0.0, top-k 액션 밖 조합 → 0.0 (candidate_gate 가 후보-외 조합을 감점)."""
    rewards = []
    for comp, actions_json in zip(completions, topk_actions_with_score):
        verb, noun = parse_vn(comp)
        if not verb or not noun:
            rewards.append(0.0)
            continue
        try:
            actions = json.loads(actions_json)
        except Exception:
            actions = []
        liks = [a.get("likelihood") for a in actions]
        matched = next((i for i, a in enumerate(actions)
                        if a.get("verb") == verb and a.get("noun") == noun), None)
        rewards.append(_likelihood_reward_value(liks, matched))
    return rewards


def _noun_mention_spans(think_lower: str, noun: str) -> List[int]:
    """think 내 후보 noun 의 등장 위치들. EK100 계층 키('board:chopping')는 base 로도 매칭."""
    positions = []
    variants = {noun.lower()}
    base = noun.split(":")[0].lower()
    variants.add(base)
    # 'board:chopping' 을 모델이 'chopping board' 로 풀어 쓰는 경우
    if ":" in noun:
        parts = noun.lower().split(":")
        variants.add(" ".join(reversed(parts)))
    for v in variants:
        if not v:
            continue
        for m in re.finditer(rf"(?<![a-z]){re.escape(v)}(?![a-z])", think_lower):
            positions.append(m.start())
    return positions


def _verb_mention_spans(think_lower: str, verb: str) -> List[int]:
    """think 내 후보 verb 의 등장 위치들. EK100 verb 는 'put-down'/'turn-off' 처럼 하이픈이
    있는데 모델은 'put down' 으로 풀어 쓰기도 하므로 두 표기를 모두 매칭."""
    positions = []
    v = verb.lower()
    variants = {v, v.replace("-", " "), v.split("-")[0]}
    for x in variants:
        if not x:
            continue
        for m in re.finditer(rf"(?<![a-z]){re.escape(x)}(?![a-z])", think_lower):
            positions.append(m.start())
    return positions


def think_convergence_reward_joint(completions, topk_actions_display, **kwargs) -> List[float]:
    """P4 의 joint 포맷 이식판 — 수렴을 **noun 이 아니라 (verb,noun) 쌍** 단위로 채점.

    이식이 필요한 이유(실측): joint top-5 안의 distinct noun 이 5개인 샘플은 19% 뿐이고,
    ≤2개(= noun 만으론 후보 구분 불가)가 16.3%. 'close lid / put lid / take lid' 를
    noun 으로는 구분할 수 없으므로 원본 P4 를 그대로 쓰면 신호가 퇴화한다.

      +0.10  최종 선택 쌍의 verb·noun 이 모두 think 에 언급됨
      +0.15  think 에서 '마지막으로' 언급된 후보 쌍 == 최종 선택 (표류 없이 수렴)
      -0.10  think 는 있으나 선택 쌍을 언급 안 함 (장식 think)
       0.0   parse 불가 (format/gate 항이 처리)

    쌍의 언급 위치 = min(verb 마지막 위치, noun 마지막 위치) — 두 토큰이 모두 나온 시점.
    GT 는 일절 참조하지 않는다 (GT-free 유지)."""
    rewards = []
    for comp, disp in zip(completions, topk_actions_display):
        verb, noun, think = parse_action_from_think_format(comp)
        if not think or not verb or not noun:
            rewards.append(0.0)
            continue
        try:
            cands = [(str(a.get("verb", "")), str(a.get("noun", ""))) for a in json.loads(disp)]
        except Exception:
            cands = []
        if not cands:
            rewards.append(0.0)
            continue
        tl = think.lower()
        last_pos = {}
        for cv, cn in cands:
            vs = _verb_mention_spans(tl, cv)
            ns = _noun_mention_spans(tl, cn)
            if vs and ns:
                last_pos[(cv, cn)] = min(max(vs), max(ns))
        if (verb, noun) not in last_pos:
            rewards.append(-0.10)
            continue
        r = 0.10
        if max(last_pos, key=last_pos.get) == (verb, noun):
            r += 0.15
        rewards.append(r)
    return rewards


def think_convergence_reward(completions, topk_nouns, **kwargs) -> List[float]:
    """P4 (CoVo식 후보 언급 일관성, GT 불사용): think 의 후보 언급 궤적이 최종 선택으로
    수렴하는지 문자열 규칙으로 채점. 후보 집합(=WM 출력)과 생성 텍스트만의 결정론적 함수.
      +0.10  최종 선택 noun 이 think 에서 언급됨
      +0.15  think 에서 '마지막으로' 언급된 후보 noun == 최종 선택 noun (표류 없이 수렴)
      -0.10  think 는 있으나 최종 선택 noun 을 한 번도 언급 안 함 (장식 think)
       0.0   think/action parse 불가 (format/gate 항이 처리)
    '배제 논리의 옳고 그름'은 판정하지 않음 — GT 뒷문 차단."""
    rewards = []
    for comp, tn in zip(completions, topk_nouns):
        verb, noun, think = parse_action_from_think_format(comp)
        if not think or not noun:
            rewards.append(0.0)
            continue
        try:
            cand_nouns = [str(c) for c in json.loads(tn) if str(c).strip()]
        except Exception:
            cand_nouns = []
        if not cand_nouns:
            rewards.append(0.0)
            continue
        tl = think.lower()
        last_pos = {}  # cand noun → 마지막 언급 위치
        for c in cand_nouns:
            spans = _noun_mention_spans(tl, c)
            if spans:
                last_pos[c] = max(spans)
        chosen_mentioned = noun in last_pos
        if not chosen_mentioned:
            rewards.append(-0.10)
            continue
        r = 0.10
        final_mention = max(last_pos, key=last_pos.get)
        if final_mention == noun:
            r += 0.15
        rewards.append(r)
    return rewards


# ------------------------- rewards: P3 (Run 2 분기 A 전용) -------------------------
# 신경망 판정이 개입하는 유일한 항 — 논문에서는 WM-유래 reward 와 구분되는
# "coherence regularizer" 로 분리 서술. 안전장치:
#   (a) think 말미의 명시적 결론 선언 문장은 마스킹 후 측정 (답안 예고편 hacking 차단)
#   (b) 가중치(P3_WEIGHT)는 P1(최대 ~0.8)·P4(0.25)보다 작게 — 기본 0.25 × p ∈ [0, 0.25]
#   (c) 판정자는 '학습 전 base 모델'의 동결 사본 (정책이 아님 — 정책이 조작 불가)
# 측정은 text-only: p(<action> JSON | 후보 목록, 마스킹된 think). 이미지 forward 를 빼서
# step 오버헤드를 최소화 — grounding 은 P1(WM) 담당, P3 은 '추론→결론 지지'만 담당.
# (eval_reasoning_trace.py 계층 2의 결론-마스킹 테스트가 이 항의 hacking 검출기)

P3_CONCLUSION_PAT = re.compile(
    r"(?:therefore|so,|thus|hence|i (?:will|should|choose|select|pick)|"
    r"the (?:best|most likely|next) action is|my (?:choice|answer|selection))",
    re.IGNORECASE)
P3_WEIGHT = 0.0          # main() 에서 --p3_weight 로 설정 (0 이면 이 항은 항상 0 반환)
P3_REF_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
_P3_REF: Dict[str, Any] = {"model": None, "tok": None}


def p3_mask_conclusion(think: str) -> str:
    """think 말미의 결론 선언 문장들을 뒤에서부터 제거 (eval_reasoning_trace 와 동일 규칙)."""
    sents = re.split(r"(?<=[.!?])\s+", think.strip())
    while sents and P3_CONCLUSION_PAT.search(sents[-1]):
        sents.pop()
    return " ".join(sents).strip()


def _p3_ref():
    """동결 base 모델 lazy 로드 (프로세스당 1회, 정책과 같은 GPU)."""
    if _P3_REF["model"] is None:
        from transformers import AutoTokenizer
        dev = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu"
        print(f"[P3] loading frozen reference model on {dev} (text-only scoring)")
        m = AutoModelForImageTextToText.from_pretrained(
            P3_REF_MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": dev})
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        _P3_REF["model"] = m
        _P3_REF["tok"] = AutoTokenizer.from_pretrained(P3_REF_MODEL_NAME)
    return _P3_REF["model"], _P3_REF["tok"]


@torch.no_grad()
def _p3_batch_mean_logp(pairs: List[Tuple[str, str]]) -> List[float]:
    """pairs=(context, target) 각각에 대해 target 토큰의 평균 log p 를 배치로 계산."""
    model, tok = _p3_ref()
    dev = next(model.parameters()).device
    ctx_ids = [tok(c, add_special_tokens=False).input_ids for c, _ in pairs]
    tgt_ids = [tok(t, add_special_tokens=False).input_ids for _, t in pairs]
    seqs = [c + t for c, t in zip(ctx_ids, tgt_ids)]
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id or 0
    input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
    out = model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev))
    logp = torch.log_softmax(out.logits.float(), dim=-1)
    means = []
    for i, (c, t) in enumerate(zip(ctx_ids, tgt_ids)):
        if not t:
            means.append(float("-inf"))
            continue
        pos = torch.arange(len(c) - 1, len(c) + len(t) - 1)
        labels = torch.tensor(t)
        tok_lp = logp[i, pos, labels]
        means.append(float(tok_lp.mean().item()))
    return means


def think_support_reward(completions, topk_verbs, topk_nouns, **kwargs) -> List[float]:
    """P3: reasoning 이 결론을 지지하는 정도 = p(선택 행동 | 후보, 결론-마스킹 think).
    reward = P3_WEIGHT × exp(target 평균 log p) ∈ [0, P3_WEIGHT].
    think/action parse 불가, 마스킹 후 think 가 비면 0.0."""
    n = len(completions)
    if P3_WEIGHT <= 0:
        return [0.0] * n
    rewards = [0.0] * n
    pairs, idxs = [], []
    for i, (comp, tv, tn) in enumerate(zip(completions, topk_verbs, topk_nouns)):
        verb, noun, think = parse_action_from_think_format(comp)
        if not verb or not noun or not think:
            continue
        masked = p3_mask_conclusion(think)
        if not masked:
            continue                      # 결론 선언만으로 구성된 think → 보너스 없음
        try:
            verbs = ", ".join(json.loads(tv)); nouns = ", ".join(json.loads(tn))
        except Exception:
            verbs = nouns = ""
        ctx = (f"Verb candidates: {verbs}\nNoun candidates: {nouns}\n"
               f"Reasoning:\n{masked}\n"
               f"Based only on this reasoning, the selected action is: ")
        tgt = json.dumps({"verb": verb, "noun": noun})
        pairs.append((ctx, tgt)); idxs.append(i)
    if not pairs:
        return rewards
    try:
        means = _p3_batch_mean_logp(pairs)
    except Exception as e:
        print(f"[P3][WARN] scoring failed: {type(e).__name__}: {e} — 이 배치 P3=0")
        return rewards
    import math
    for i, m in zip(idxs, means):
        rewards[i] = P3_WEIGHT * math.exp(m) if m > float("-inf") else 0.0
    return rewards


# reward_mode → reward fn 리스트 (이름은 로깅 키로도 사용)
def build_reward_funcs(reward_mode: str):
    table = {
        "think_format":  [format_reward_think, think_quality_reward, candidate_reward_think, gt_accuracy_reward_think],
        # 실험7: candidate 게이트 + gt 강화 (collapse 방지). KL(β)·온도는 GRPOConfig 에서.
        "think_gt":      [format_reward_think, think_quality_reward, candidate_gate_reward_think, gt_accuracy_reward_think_v2],
        "think_ranking": [format_reward_think, think_quality_reward, candidate_reward_think, wm_ranking_reward],
        "wm_ranking":    [format_reward, action_candidate_consistency_reward, wm_ranking_reward],
        "noun_ranking":  [format_reward, noun_ranking_reward],
        "action_ranking_from_noun": [format_reward, action_candidate_consistency_reward, wm_ranking_reward],
        # 실험 9~13: 처방 1+2+3 (think + gate + β + max_steps=750)
        # Exp10: WM rank 단독 (GT 없음, graded rank으로 영상→WM 정렬 학습)
        "think_wm_rank_fix":  [format_reward_think, think_quality_reward, candidate_gate_reward_think, wm_ranking_reward],
        # Exp11: GT + WM rank 복합 (GT 주신호 1.5, WM 보조 1.0)
        "think_gt_combo":     [format_reward_think, think_quality_reward, candidate_gate_reward_think, gt_accuracy_reward_think_v2, wm_ranking_reward],
        # Exp12 Stage1: noun GT만 (2-stage 커리큘럼 시작)
        "think_noun_gt":      [format_reward_think, think_quality_reward, candidate_gate_reward_think, noun_gt_reward_think],
        # Exp13 Stage1: noun GT + noun WM rank 복합 (2-stage + combo)
        "think_noun_combo":   [format_reward_think, think_quality_reward, candidate_gate_reward_think, noun_gt_reward_think, noun_ranking_reward],
        # Exp14 (grpo_final): GT v3(퍼지 noun) + WM ranking, 4998샘플, num_gen=8, 1250 steps
        "think_gt_final":     [format_reward_think, think_quality_reward, candidate_gate_reward_think, gt_accuracy_reward_think_v3, wm_ranking_reward],
        # Run 1 (WM-only, GT-free): P1 likelihood 주신호 + P4 수렴 + 최소 구조 gate.
        # think_quality_reward(20단어 이상 길이 보너스)는 '길이 보너스 금지' 원칙에 따라 제외
        # — 후보 언급 유인은 P4 가 기능(수렴) 기준으로 담당.
        "wm_likelihood":      [format_reward_think, candidate_gate_reward_think, wm_likelihood_reward, think_convergence_reward],
        # Run 2 분기 A: Run 1 구성 + P3(coherence regularizer, 결론 마스킹 필수).
        # --p3_weight > 0 필요 (기본 0 이면 P3 항이 항상 0 — 실수 방지용 가드는 main() 에).
        "wm_likelihood_p3":   [format_reward_think, candidate_gate_reward_think, wm_likelihood_reward, think_convergence_reward, think_support_reward],
        # F0 v2: joint action top-5 (5지선다) 포맷. Run1 과 리워드 구성은 동일하고
        # gate 만 joint 쌍 소속 검사로 교체 — **리워드 신호는 WM likelihood 단독 유지**
        # (history-consistency 같은 비-WM 항은 넣지 않는다. G1 오염 방지).
        "wm_likelihood_joint": [format_reward_joint, candidate_gate_reward_joint, wm_likelihood_reward, think_convergence_reward_joint],
    }
    return table[reward_mode]


# ------------------------- dataset -------------------------

def _wm_spread(ex: Dict[str, Any], top_k: int) -> float:
    """top-k action likelihood 를 후보셋 내 재정규화한 뒤의 표준편차.
    이 값이 작으면 WM 분포가 flat → 어떤 선택을 해도 reward 차이가 없어
    GRPO advantage 가 생성될 수 없는 샘플 (Exp.10 실패의 원인 구간)."""
    liks = [a.get("likelihood") for a in (ex.get("topk_actions_with_score") or [])[:top_k]]
    liks = [float(x) for x in liks if x is not None]
    if len(liks) < 2:
        return 0.0
    s = sum(liks)
    if s <= 0:
        return 0.0
    ren = [x / s for x in liks]
    mean = sum(ren) / len(ren)
    return (sum((x - mean) ** 2 for x in ren) / len(ren)) ** 0.5


def filter_rows_for_stage(rows, stage, top_k, drop_unrewardable_samples, reward_mode=None,
                          min_wm_spread: float = 0.0):
    # GT-free 모드: GT 기반 필터를 쓰지 않는다 (GT 가 학습 분포에 새는 뒷문 차단).
    # 대신 dynamic sampling 의 정적 절반 — WM 분포가 flat 해 advantage 가 태생적으로
    # 생성 불가능한 샘플을 사전에 제거 (reward 는 WM likelihood 만의 함수이므로
    # 프롬프트별 achievable reward spread 가 학습 전에 결정된다).
    if reward_mode in GT_FREE_REWARD_MODES:
        if min_wm_spread <= 0:
            return rows
        kept = [ex for ex in rows if _wm_spread(ex, top_k) >= min_wm_spread]
        print(f"[filter] wm_spread >= {min_wm_spread}: {len(kept)}/{len(rows)} rows kept "
              f"(dynamic sampling, static half)")
        return kept
    if not drop_unrewardable_samples:
        return rows
    filtered = []
    for ex in rows:
        nouns = normalize_topk_nouns(ex.get("topk_nouns", []), top_k)
        actions = normalize_topk_actions(ex.get("topk_actions", []), top_k)
        if not nouns or not actions:
            continue
        if reward_mode in THINK_REWARD_MODES:
            # think 모드: GT verb∈topk_verbs AND GT noun∈topk_nouns (퍼지 포함)이어야
            # GT 보상 > 0 가능. 둘 다 후보 밖인 51개(~1%)는 학습 신호 없으므로 제거.
            gt_v = ex.get("gt_verb", "").strip()
            gt_n = ex.get("gt_noun", "").strip()
            verbs = [str(v).strip() for v in (ex.get("topk_verbs") or []) if str(v).strip()][:top_k]
            cand_nouns = [n["noun"] for n in nouns]
            v_ok = gt_v in verbs
            n_ok = gt_n in cand_nouns or any(_noun_fuzzy_match(gt_n, cn) for cn in cand_nouns)
            if v_ok or n_ok:  # verb OR noun 중 하나라도 가능하면 부분 신호 유지
                filtered.append(ex)
        elif stage == "noun":
            if nouns[0]["noun"] in {a["noun"] for a in actions}:
                filtered.append(ex)
        else:
            filtered.append(ex)
    return filtered


def _prompt_text_of(converted: Dict[str, Any]) -> str:
    return "\n".join(str(m.get("content", "")) for m in converted.get("prompt", []))


def assert_no_score_leak(converted_rows: List[Dict[str, Any]], raw_rows: List[Dict[str, Any]],
                         top_k: int, n_check: int = 500) -> None:
    """P1 leakage 방지 assertion: probe likelihood(원본/재정규화, 다양한 자릿수 표기)가
    프롬프트 텍스트에 노출되지 않았는지 검사. 5a 에서 후보 점수 노출이 즉시 붕괴를
    유발했으므로, wm_likelihood reward 도입 시 구조적으로 재발을 차단한다."""
    checked = 0
    for conv, raw in zip(converted_rows, raw_rows):
        if checked >= n_check:
            break
        text = _prompt_text_of(conv)
        liks = [a.get("likelihood") for a in (raw.get("topk_actions_with_score") or [])[:top_k]]
        liks += [n.get("likelihood") for n in (raw.get("topk_nouns_with_score") or [])[:top_k]]
        liks = [float(x) for x in liks if x is not None]
        s = sum(liks)
        variants = list(liks) + ([x / s for x in liks] if s > 0 else [])
        for v in variants:
            for fmt in (f"{v:.3f}", f"{v:.4f}", f"{v:.6f}", str(v)):
                # '0.000' 같은 자명 문자열은 오탐이므로 제외
                if float(fmt) == 0.0:
                    continue
                assert fmt not in text, (
                    f"[LEAK] likelihood {fmt} found in prompt of sample "
                    f"{conv.get('sample_id')} — reward 전용 값이 프롬프트에 노출됨")
        checked += 1
    print(f"[check] no likelihood leak in {checked} prompts")


def build_dataset(jsonl_path, limit, stage, top_k, drop_unrewardable_samples,
                  shuffle_candidates=False, hide_top1_hint=False, seed=42,
                  reward_mode=None, min_wm_spread: float = 0.0):
    rows = load_jsonl(jsonl_path)
    rows = filter_rows_for_stage(rows, stage, top_k, drop_unrewardable_samples,
                                 reward_mode=reward_mode, min_wm_spread=min_wm_spread)
    if limit is not None:
        rows = rows[:limit]
    rng = random.Random(seed)
    converted, kept_raw, skipped = [], [], 0
    for ex in rows:
        try:
            if not Path(ex["image_path"]).exists():
                skipped += 1
                continue
            if not ex.get("topk_nouns") or not ex.get("topk_actions"):
                skipped += 1
                continue
            converted.append(make_conversation(ex, stage=stage, top_k=top_k,
                                                shuffle_candidates=shuffle_candidates,
                                                hide_top1_hint=hide_top1_hint, rng=rng,
                                                reward_mode=reward_mode))
            kept_raw.append(ex)
        except Exception as e:
            skipped += 1
            print(f"[WARN] skipped one sample: {e}")
    if skipped:
        print(f"[INFO] skipped samples: {skipped}")
    if not converted:
        raise RuntimeError("No valid samples after filtering.")
    if reward_mode in GT_FREE_REWARD_MODES:
        assert_no_score_leak(converted, kept_raw, top_k)
    ds = Dataset.from_list(converted)
    # "image" 경로 문자열 → datasets Image feature (접근 시 PIL 로 lazy decode)
    ds = ds.cast_column("image", DSImage())
    return ds


def _device_map():
    """DDP(accelerate, WORLD_SIZE>1)면 프로세스별 단일 GPU에 통째로 로드.
    단일 프로세스면 'auto' (가시 GPU에 배치)."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return {"": int(os.environ.get("LOCAL_RANK", "0"))}
    return "auto"


def create_or_load_lora_model(model_name, adapter_path, lora_r, lora_alpha, lora_dropout,
                              attn_impl, resume_lora=None):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=_device_map(),
    )
    model.config.use_cache = False

    def _new_lora(m):
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        return get_peft_model(m, lora_config)

    if resume_lora:
        # 실험6 Stage2: Stage1 어댑터를 가중치에 병합(merge_and_unload) 후 새 LoRA 학습.
        print(f"[INFO] resume_lora: merge prior adapter then train FRESH LoRA: {resume_lora}")
        model = PeftModel.from_pretrained(model, resume_lora)
        model = model.merge_and_unload()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = _new_lora(model)
    elif adapter_path:
        print(f"[INFO] Loading existing LoRA adapter (continued training): {adapter_path}")
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print("[INFO] Creating new LoRA adapter")
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  # gradient checkpointing + PEFT 필수
        model = _new_lora(model)
    model.print_trainable_parameters()
    return model


# ------------------------- 상세 로깅 (rank0 only) -------------------------

def _is_main_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _file_md5(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return "nofile"


class GRPOLogger(TrainerCallback):
    """rank0 전용 상세 로거.
      [A] reward_log.jsonl       — 매 logging_steps: 총/구성 reward, loss, grad_norm
      [B] completion_samples.jsonl — 매 capture_every step: 그룹 2개의 생성 4개 + breakdown
      [C] think_analysis.jsonl   — think 모드 한정: 단어수/후보언급률/생성다양성
    completion/think 캡처는 reward 'sink' 함수가 shared 버퍼에 채워두면 on_step_end 에서 flush.
    """

    def __init__(self, out_dir: str, reward_names: List[str], num_generations: int,
                 is_think: bool, shared: Dict[str, Any], capture_every: int = 100):
        self.dir = Path(out_dir)
        self.reward_names = reward_names
        self.ng = num_generations
        self.is_think = is_think
        self.shared = shared
        self.capture_every = capture_every
        self.dir.mkdir(parents=True, exist_ok=True)

    # capture 여부를 step 시작 시 결정 → reward sink 가 참조
    def on_step_begin(self, args, state, control, **kw):
        self.shared["step"] = state.global_step
        self.shared["capture"] = (state.global_step % self.capture_every == 0)
        self.shared["buffer"] = None

    def on_log(self, args, state, control, logs=None, **kw):
        if not _is_main_process() or not logs:
            return
        if "reward" not in logs:   # 학습-종료 요약 로그(train_runtime 등)는 건너뜀
            return
        rec = {
            "step": state.global_step,
            "reward_total": logs.get("reward"),
            "loss": logs.get("loss"),
            "grad_norm": logs.get("grad_norm"),
            "epoch": logs.get("epoch"),
        }
        for name in self.reward_names:
            rec[f"reward_{name}"] = logs.get(f"rewards/{name}/mean")
        # dynamic sampling 필터율 (P2) — 분기 B(advantage 소실 잔존) 판정의 1차 지표
        if "dynamic_sampling/frac_groups_filtered" in logs:
            rec["ds_frac_groups_filtered"] = logs.get("dynamic_sampling/frac_groups_filtered")
        if "frac_reward_zero_std" in logs:
            rec["frac_reward_zero_std"] = logs.get("frac_reward_zero_std")
        _append_jsonl(self.dir / "reward_log.jsonl", rec)

    def on_step_end(self, args, state, control, **kw):
        if not _is_main_process():
            return
        buf = self.shared.get("buffer")
        if not buf:
            return
        step = state.global_step
        comps = buf["completions"]          # flat list
        breakdown = buf["breakdown"]        # {name: [floats]}
        cols = buf["cols"]                  # {colname: [...]}
        n = len(comps)
        names = list(breakdown.keys())

        def total_at(i):
            return float(sum(breakdown[nm][i] for nm in names))

        # 그룹(=프롬프트) 단위로 num_generations 씩 묶음. 앞 2그룹 기록.
        ngroups = max(1, n // self.ng)
        for g in range(min(2, ngroups)):
            sl = slice(g * self.ng, (g + 1) * self.ng)
            idxs = list(range(g * self.ng, min((g + 1) * self.ng, n)))
            comp_recs = []
            for i in idxs:
                comp_recs.append({
                    "text": extract_text_from_completion(comps[i])[:1200],
                    "reward_total": round(total_at(i), 4),
                    "reward_breakdown": {nm: round(float(breakdown[nm][i]), 4) for nm in names},
                })
            sid = (cols.get("sample_id") or [""] * n)[idxs[0]] if idxs else ""
            prompt_tail = buf.get("prompt_tail", [""] * n)
            rec = {
                "step": step,
                "sample_id": sid,
                "prompt_tail": prompt_tail[idxs[0]] if idxs else "",
                "completions": comp_recs,
                "gt_verb": (cols.get("gt_verb") or [""] * n)[idxs[0]] if idxs else "",
                "gt_noun": (cols.get("gt_noun") or [""] * n)[idxs[0]] if idxs else "",
            }
            # judge_reasoning_curve.py 용 — judge 가 history·후보를 봐야 채점 가능.
            # (GT 는 rec 에 있지만 judge 스크립트가 **의도적으로 전달하지 않는다** — 품질과 정답률 분리)
            for col in ("topk_actions_display", "memory_context"):
                v = cols.get(col)
                if v and idxs:
                    rec[col] = v[idxs[0]]
            # WM rank1 (가능하면 topk_*_with_score 의 rank1)
            aws = cols.get("topk_actions_with_score")
            if aws and idxs:
                try:
                    a0 = json.loads(aws[idxs[0]])
                    r1 = next((a for a in a0 if a.get("rank") == 1), a0[0] if a0 else {})
                    rec["wm_rank1_verb"] = r1.get("verb", "")
                    rec["wm_rank1_noun"] = r1.get("noun", "")
                except Exception:
                    pass
            _append_jsonl(self.dir / "completion_samples.jsonl", rec)

        # [C] think 분석 (think 모드만)
        if self.is_think:
            think_texts, wcs, mention_flags, pairs = [], [], [], []
            tv = cols.get("topk_verbs") or ["[]"] * n
            tn = cols.get("topk_nouns") or ["[]"] * n
            cap = min(n, self.ng * 2)
            for i in range(cap):
                v, no, think = parse_action_from_think_format(comps[i])
                think = think or ""
                think_texts.append(think[:600])
                wc = len(think.split())
                wcs.append(wc)
                try:
                    cands = [c.lower() for c in json.loads(tv[i])] + [c.lower() for c in json.loads(tn[i])]
                except Exception:
                    cands = []
                tl = think.lower()
                mention_flags.append(any(c and c in tl for c in cands))
                pairs.append((v, no))
            uniq = len({p for p in pairs if p[0] and p[1]})
            denom = len([p for p in pairs if p[0] and p[1]]) or 1
            rec = {
                "step": step,
                "think_word_count_mean": round(sum(wcs) / max(1, len(wcs)), 2),
                "think_word_count_min": min(wcs) if wcs else 0,
                "think_word_count_max": max(wcs) if wcs else 0,
                "candidate_mention_rate": round(sum(mention_flags) / max(1, len(mention_flags)), 3),
                "generation_diversity": round(uniq / denom, 3),
                "think_texts": think_texts[: self.ng],
            }
            _append_jsonl(self.dir / "think_analysis.jsonl", rec)
        self.shared["buffer"] = None


def make_logging_sink(reward_components: List[Tuple[str, Any]], shared: Dict[str, Any]):
    """capture step 일 때 완성/구성 reward 를 shared['buffer'] 에 적재하는 reward fn.
    항상 0.0 반환 (GRPO advantage 에 영향 없음: 상수항)."""
    def log_sink(completions, **kwargs):
        if shared.get("capture") and _is_main_process() and shared.get("buffer") is None:
            breakdown = {nm: fn(completions, **kwargs) for nm, fn in reward_components}
            prompts = kwargs.get("prompts") or kwargs.get("prompt")
            tails = []
            if isinstance(prompts, list):
                for p in prompts:
                    tails.append(extract_text_from_completion(p)[-200:])
            shared["buffer"] = {
                "completions": completions,
                "breakdown": breakdown,
                "cols": {k: v for k, v in kwargs.items() if isinstance(v, list)},
                "prompt_tail": tails or [""] * len(completions),
            }
        return [0.0] * len(completions)
    log_sink.__name__ = "log_sink"
    return log_sink


# ------------------------- dynamic sampling (DAPO 계열, 런타임 절반) -------------------------

class DynamicSamplingGRPOTrainer(GRPOTrainer):
    """그룹(=프롬프트) 내 reward 표준편차가 임계치 이하인 그룹의 advantage 를 0 으로 마스킹.

    trl 1.5.1 에는 DAPO 의 dynamic sampling(무신호 그룹 필터/재샘플)이 없어 직접 구현.
    재샘플 대신 마스킹인 이유: trl 의 생성-스코어링 루프를 침습하지 않으면서,
    (1) 데이터셋 정적 필터(min_wm_spread)가 '태생적 flat 프롬프트'를 이미 제거했고
    (2) 남는 무신호 그룹은 '정책이 같은 답만 생성한 경우'라 마스킹으로 노이즈 gradient 차단이 목적.

    전제: scale_rewards="none" (Dr. GRPO). 이때 advantage = reward − group mean 이므로
    그룹 내 advantage std == 그룹 내 reward std. scale_rewards="group"(기본값)이면
    std 정규화로 모든 그룹의 std≈1 이 되어 이 필터가 무의미해짐 → __init__ 에서 검증.
    """

    def __init__(self, *args, group_std_threshold: float = 0.0, ds_shared: Optional[Dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._group_std_threshold = float(group_std_threshold)
        self._ds_shared = ds_shared if ds_shared is not None else {}
        if self._group_std_threshold > 0 and getattr(self.args, "scale_rewards", "group") != "none":
            raise ValueError(
                "dynamic_sampling_std_threshold > 0 requires scale_rewards='none' "
                "(Dr. GRPO). scale_rewards='group' 에선 그룹 std 가 정규화로 소거되어 "
                "advantage 에서 reward 분산을 복원할 수 없다.")

    def _generate_and_score_completions(self, generation_batch):
        output = super()._generate_and_score_completions(generation_batch)
        thr = self._group_std_threshold
        adv = output.get("advantages")
        G = self.num_generations
        if thr > 0 and adv is not None and adv.numel() >= G and adv.numel() % G == 0:
            groups = adv.view(-1, G)
            group_std = groups.std(dim=1, keepdim=True)
            keep = (group_std > thr).to(groups.dtype)
            frac_filtered = float(1.0 - keep.mean().item())
            output["advantages"] = (groups * keep).reshape(-1)
            self._metrics["train"]["dynamic_sampling/frac_groups_filtered"].append(frac_filtered)
            self._ds_shared["ds_frac_filtered"] = frac_filtered
        return output


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--stage", type=str, choices=["noun", "action", "gt"], default="gt",
                   help="legacy 모드용 프롬프트 stage. reward_mode 지정 시 JSON 후보 프롬프트의 지시문만 결정.")
    p.add_argument("--reward_mode", type=str, default=None,
                   choices=["think_format", "think_gt", "think_ranking", "wm_ranking",
                            "noun_ranking", "action_ranking_from_noun",
                            "think_wm_rank_fix", "think_gt_combo",
                            "think_noun_gt", "think_noun_combo",
                            "think_gt_final", "wm_likelihood", "wm_likelihood_p3",
                            "wm_likelihood_joint"],
                   help="실험 4/5/6/7 reward 구성. 미지정 시 legacy(stage+reward_target) 경로.")
    p.add_argument("--reward_target", type=str, default="auto", choices=["auto", "vjepa_top1", "gt"],
                   help="(legacy) auto: stage=gt→gt, noun/action→vjepa_top1")
    p.add_argument("--hide_top1_hint", action="store_true", help="프롬프트에서 V-JEPA2 top-1 정답 노출 제거")
    p.add_argument("--shuffle_candidates", action="store_true", help="후보 순서 셔플 (index 복사 방지)")
    p.add_argument("--hide_scores", action="store_true", help="후보 점수 노출 제거 (rank 자명해 차단)")
    p.add_argument("--beta", type=float, default=0.0, help="GRPO KL 정규화 계수 (>0 이면 collapse 억제)")
    p.add_argument("--temperature", type=float, default=0.8, help="생성 온도 (↑ 다양성)")
    p.add_argument("--adapter_path", type=str, default=None,
                   help="(legacy) 동일 어댑터 이어학습")
    p.add_argument("--resume_lora", type=str, default=None,
                   help="실험6 Stage2: 이전 LoRA 를 merge 후 새 LoRA 학습")
    p.add_argument("--train_samples", type=int, default=3000)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--drop_unrewardable_samples", action="store_true")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=None,
                   help="미지정 시 think 계열 256, 그 외 64 로 자동 결정")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=2)
    p.add_argument("--save_steps", type=int, default=100)
    # ----- Run 1 (WM-only) 추가 인자: P2/P5 최적화 장치 + P1 옵션 -----
    p.add_argument("--loss_type", type=str, default=None,
                   choices=["grpo", "dapo", "bnpo", "dr_grpo"],
                   help="trl GRPOConfig.loss_type. 미지정 시 trl 기본값 유지(기존 실험 재현성). "
                        "Run 1: dr_grpo (길이 편향 제거 — P5)")
    p.add_argument("--epsilon_high", type=float, default=None,
                   help="clip-higher 상한 ε_high (DAPO — P2). 미지정 시 epsilon 과 동일(대칭 클리핑)")
    p.add_argument("--scale_rewards", type=str, default=None,
                   choices=["group", "batch", "none"],
                   help="advantage std 정규화. 미지정 시 trl 기본(group). "
                        "dr_grpo 는 'none' 이 정합 (Dr. GRPO 는 std 정규화도 제거)")
    p.add_argument("--dynamic_sampling_std_threshold", type=float, default=0.0,
                   help="그룹 reward std 가 이 값 이하인 그룹의 advantage 를 0 으로 마스킹 "
                        "(P2 dynamic sampling 런타임 절반). 0 = off. scale_rewards='none' 필요")
    p.add_argument("--min_wm_spread", type=float, default=0.0,
                   help="재정규화 top-k likelihood std 가 이 값 미만인 샘플을 데이터셋에서 제거 "
                        "(P2 dynamic sampling 정적 절반, wm_likelihood 모드 전용). 0 = off. "
                        "실측 분포: 0.05→하위 7%%, 0.08→18.8%% 제거")
    p.add_argument("--wm_likelihood_norm", type=str, default="candidate",
                   choices=["candidate", "raw"],
                   help="P1 reward 정규화: candidate=likelihood/sum(top-k) (기본), raw=probe softmax 그대로")
    p.add_argument("--wm_likelihood_temp", type=float, default=1.0,
                   help="P1 sharpening 온도 (candidate norm 전용). <1 이면 top 후보로 뾰족(WM-copy 정렬), "
                        "1.0=Run 1 동일. G2 구조적 천장(분포매칭<argmax) 완화/정량화용")
    p.add_argument("--no_memory", action="store_true",
                   help="memory-off (misalignment ③): 프롬프트 memory_context 공란화. 학습·평가 동일 적용")
    # ----- Run 2 대비 인자 -----
    p.add_argument("--p3_weight", type=float, default=0.0,
                   help="P3(think_support) 가중치. wm_likelihood_p3 모드에서 >0 필수 (권장 0.25 — "
                        "P1 최대 ~0.8, P4 최대 0.25 보다 작거나 같게). ref 모델 forward 가 붙어 step 시간 증가")
    p.add_argument("--completion_log_every", type=int, default=25,
                   help="completion_samples.jsonl 캡처 주기(step). judge 리즈닝 곡선용 — "
                        "judge_reasoning_curve.py 가 이 파일을 읽는다. 학습 비용 없음(롤아웃 재사용).")
    p.add_argument("--reward_weights", type=str, default=None,
                   help="reward fn 별 가중치 (콤마 구분, 로깅 sink 제외한 fn 수와 일치). "
                        "예: 분기 C 에서 P4 하향 '1,1,1,0.5'. 미지정 시 전부 1.0")
    return p.parse_args()


def main():
    global PARSE_FORMAT, HIDE_SCORES, WM_LIK_NORM, WM_LIK_TEMP, NO_MEMORY, P3_WEIGHT
    args = parse_args()

    is_think = args.reward_mode in THINK_REWARD_MODES
    PARSE_FORMAT = "think" if is_think else "json"
    HIDE_SCORES = args.hide_scores
    WM_LIK_NORM = args.wm_likelihood_norm
    WM_LIK_TEMP = args.wm_likelihood_temp
    NO_MEMORY = args.no_memory
    P3_WEIGHT = args.p3_weight
    if WM_LIK_TEMP <= 0:
        raise ValueError("--wm_likelihood_temp 는 >0 이어야 함")
    if args.reward_mode == "wm_likelihood_p3" and args.p3_weight <= 0:
        raise ValueError("wm_likelihood_p3 모드는 --p3_weight > 0 필요 (권장 0.25)")
    if args.reward_mode != "wm_likelihood_p3" and args.p3_weight > 0:
        raise ValueError("--p3_weight 는 wm_likelihood_p3 모드에서만 유효")

    # max_completion_length 자동 결정 (미지정 시 think 256, 그 외 64)
    if args.max_completion_length is None:
        args.max_completion_length = 256 if is_think else 64

    print(f"[INFO] reward_mode={args.reward_mode} stage={args.stage} parse={PARSE_FORMAT} "
          f"max_compl={args.max_completion_length} top_k={args.top_k}")
    print(f"[INFO] train_jsonl={args.train_jsonl} output_dir={args.output_dir}")

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        padding_side="left",
        use_fast=True,
    )

    train_dataset = build_dataset(
        jsonl_path=args.train_jsonl,
        limit=args.train_samples,
        stage=args.stage,
        top_k=args.top_k,
        drop_unrewardable_samples=args.drop_unrewardable_samples,
        shuffle_candidates=args.shuffle_candidates,
        hide_top1_hint=args.hide_top1_hint,
        reward_mode=args.reward_mode,
        min_wm_spread=args.min_wm_spread,
    )
    print(train_dataset)

    model = create_or_load_lora_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        attn_impl=args.attn_impl,
        resume_lora=args.resume_lora,
    )

    # ----- reward 구성 -----
    if args.reward_mode is not None:
        reward_funcs = build_reward_funcs(args.reward_mode)
    else:
        # legacy: stage + reward_target
        rtgt = args.reward_target
        if rtgt == "auto":
            rtgt = "gt" if args.stage == "gt" else "vjepa_top1"
        if rtgt == "gt":
            reward_funcs = [format_reward, action_candidate_consistency_reward, gt_accuracy_reward]
        elif args.stage == "noun":
            reward_funcs = [format_reward, action_candidate_consistency_reward, stage1_top1_noun_reward]
        else:
            reward_funcs = [format_reward, action_candidate_consistency_reward, stage2_top1_action_reward]
    reward_names = [getattr(f, "__name__", f"reward{i}") for i, f in enumerate(reward_funcs)]
    print(f"[INFO] reward_funcs={reward_names} | hide_top1_hint={args.hide_top1_hint} | shuffle={args.shuffle_candidates}")

    # ----- 상세 로깅: reward sink + 콜백 -----
    shared: Dict[str, Any] = {"capture": False, "buffer": None, "step": 0}
    reward_components = list(zip(reward_names, reward_funcs))
    sink = make_logging_sink(reward_components, shared)
    reward_funcs_with_sink = reward_funcs + [sink]
    logger = GRPOLogger(args.output_dir, reward_names, args.num_generations, is_think, shared,
                        capture_every=args.completion_log_every)

    # 신규 최적화 인자는 지정된 경우에만 전달 → 기존 실험 커맨드의 재현성 유지
    extra_cfg: Dict[str, Any] = {}
    if args.loss_type is not None:
        extra_cfg["loss_type"] = args.loss_type
    if args.epsilon_high is not None:
        extra_cfg["epsilon_high"] = args.epsilon_high
    if args.scale_rewards is not None:
        extra_cfg["scale_rewards"] = args.scale_rewards
    if args.reward_weights is not None:
        w = [float(x) for x in args.reward_weights.split(",")]
        if len(w) != len(reward_funcs):
            raise ValueError(f"--reward_weights 개수({len(w)}) != reward fn 수({len(reward_funcs)}): {reward_names}")
        extra_cfg["reward_weights"] = w + [0.0]   # 마지막은 로깅 sink (항상 0 반환, 가중치 무관)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=True,
        remove_unused_columns=False,            # reward fn 에서 valid_actions_json 등 사용 → 필수
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        temperature=args.temperature,
        beta=args.beta,
        top_p=0.95,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        log_completions=True,
        # DDP(2 GPU)에서 LoRA(동결 파라미터 다수) → unused param 에러 방지
        ddp_find_unused_parameters=(int(os.environ.get("WORLD_SIZE", "1")) > 1),
        **extra_cfg,
    )

    trainer = DynamicSamplingGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs_with_sink,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[logger],
        group_std_threshold=args.dynamic_sampling_std_threshold,
        ds_shared=shared,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----- [D] meta.json (시작 시 1회, rank0) -----
    if _is_main_process():
        spec_md = Path(__file__).resolve().parent / "docs" / "GRPO_TRAIN_SPEC.md"
        meta = {
            "experiment": out.name,
            "reward_mode": args.reward_mode,
            "stage": args.stage,
            "reward_funcs": reward_names,
            "train_samples": args.train_samples,
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "model": args.model_name,
            "lora_r": args.lora_r,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "max_pixels": args.max_pixels,
            "resume_lora": args.resume_lora,
            "adapter_path": args.adapter_path,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "git_hash": _git_hash(),
            "grpo_train_spec_md5": _file_md5(spec_md),
            # Run 1 (WM-only) 신규 설정
            "loss_type": args.loss_type,
            "epsilon_high": args.epsilon_high,
            "scale_rewards": args.scale_rewards,
            "dynamic_sampling_std_threshold": args.dynamic_sampling_std_threshold,
            "min_wm_spread": args.min_wm_spread,
            "wm_likelihood_norm": args.wm_likelihood_norm,
            "wm_likelihood_temp": args.wm_likelihood_temp,
            "no_memory": args.no_memory,
            "beta": args.beta,
            "temperature": args.temperature,
            "p3_weight": args.p3_weight,
            "reward_weights": args.reward_weights,
        }
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    t0 = time.time()
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # ----- [E] summary.json (종료 시 1회, rank0) -----
    if _is_main_process():
        elapsed_h = round((time.time() - t0) / 3600, 3)
        rl = out / "reward_log.jsonl"
        rows = []
        if rl.exists():
            rows = [json.loads(l) for l in rl.read_text().splitlines() if l.strip()]
        first = rows[0] if rows else {}
        last = rows[-1] if rows else {}

        # reward_log 컬럼 중 'gt_accuracy' 포함 키 자동 탐색 (think/think_gt/legacy 호환)
        gt_key = next((k for k in last if "gt_accuracy" in k), None)
        gt_first = first.get(gt_key) if gt_key else None
        gt_last = last.get(gt_key) if gt_key else None
        summary = {
            "experiment": out.name,
            "reward_mode": args.reward_mode,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "total_steps": last.get("step"),
            "elapsed_hours": elapsed_h,
            "final_reward_total": last.get("reward_total"),
            "final_gt_accuracy": gt_last,
            "reward_trend": (f"초반 {first.get('reward_total')} → 최근 {last.get('reward_total')} "
                             f"(step {first.get('step')}~{last.get('step')})"),
            "gt_accuracy_trend": f"{gt_first} → {gt_last}" if gt_first is not None else None,
            "checkpoint_path": str(out),
        }
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 기존 호환: training_metadata.json 유지
        with open(out / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"reward_mode": args.reward_mode, "stage": args.stage,
                       "reward_funcs": reward_names, "train_samples": args.train_samples,
                       "elapsed_hours": elapsed_h}, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
