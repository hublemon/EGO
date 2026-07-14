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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainerCallback
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


# parse 포맷 디스패치: main() 에서 reward_mode 에 따라 "json" 또는 "think" 로 설정.
PARSE_FORMAT = "json"
HIDE_SCORES = False   # main() 에서 --hide_scores 로 설정 (후보 점수 노출 제거 → rank 자명해 차단)
THINK_REWARD_MODES = {"think_format", "think_ranking", "think_gt",
                      "think_wm_rank_fix", "think_gt_combo",
                      "think_noun_gt", "think_noun_combo",
                      "think_gt_final"}
JSON_REWARD_MODES = {"wm_ranking", "noun_ranking", "action_ranking_from_noun"}
RANK_REWARD_TABLE = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.2, 5: 0.1}


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
    memory_context = example.get("memory_context") or "No previous memory context is available."
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
    memory_context = example.get("memory_context") or "No previous memory context is available."
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
    text = extract_text_from_completion(completion)
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
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
    }
    return table[reward_mode]


# ------------------------- dataset -------------------------

def filter_rows_for_stage(rows, stage, top_k, drop_unrewardable_samples, reward_mode=None):
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


def build_dataset(jsonl_path, limit, stage, top_k, drop_unrewardable_samples,
                  shuffle_candidates=False, hide_top1_hint=False, seed=42,
                  reward_mode=None):
    rows = load_jsonl(jsonl_path)
    rows = filter_rows_for_stage(rows, stage, top_k, drop_unrewardable_samples, reward_mode=reward_mode)
    if limit is not None:
        rows = rows[:limit]
    rng = random.Random(seed)
    converted, skipped = [], 0
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
        except Exception as e:
            skipped += 1
            print(f"[WARN] skipped one sample: {e}")
    if skipped:
        print(f"[INFO] skipped samples: {skipped}")
    if not converted:
        raise RuntimeError("No valid samples after filtering.")
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
                            "think_gt_final"],
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
    return p.parse_args()


def main():
    global PARSE_FORMAT, HIDE_SCORES
    args = parse_args()

    is_think = args.reward_mode in THINK_REWARD_MODES
    PARSE_FORMAT = "think" if is_think else "json"
    HIDE_SCORES = args.hide_scores

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
                        capture_every=100)

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
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs_with_sink,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[logger],
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
