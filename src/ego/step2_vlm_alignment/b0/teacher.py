"""teacher.py — frozen judge/teacher (raw hindsight · projection · equivalence).

핸드오프 §5·§6·§7·§9. teacher 는 **frozen base VLM (FAA LoRA disabled)** 이다 —
외부 대형 모델에 의존하지 않는다는 기존 방침(F/B 회의) 준수. 세 역할:
  1. infer_raw_trace(Y_GT)          : GT action trajectory → raw hindsight (task+reasoning)
  2. project_full_trace(raw, c_t, a_GT): 시점 t 정보로 reasoning+belief 동시 projection
  3. equivalence(faa_belief, proj_belief) : SAME/DIFFERENT/UNCERTAIN (stop-gradient)

프롬프트 문자열은 핸드오프의 규칙/예시를 그대로 담는다. 실제 생성은 GPU 에서 수행하므로
이 파일은 (a) 프롬프트 빌더 (b) 모델 래퍼로 나눈다 — (a)는 dependency-free 라 smoke 로 검사,
(b)는 서버에서 HF 모델을 로드한다. build_dpo_dataset 은 teacher 를 duck-typing 으로 받으므로
smoke 에서 mock teacher 를 주입할 수 있다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .trace_utils import Trace, build_full_trace, parse_full_trace


# ─────────────────────────── 프롬프트 빌더 (순수) ───────────────────────────

def raw_hindsight_prompt(gt_trajectory: list[dict]) -> str:
    """§5·§6. 시간순 GT action sequence → overall task + trajectory 설명 추론."""
    seq = " -> ".join(f"{a.get('verb','')} {a.get('noun','')}".strip() for a in gt_trajectory)
    return (
        "You are given the temporally ordered ground-truth action sequence a person performed.\n"
        f"Action sequence: {seq}\n\n"
        "Infer the single overall procedural goal that best explains the whole sequence, and a\n"
        "brief explanation connecting the actions to that goal. Do not merely repeat one action.\n"
        "Return:\n"
        "<task_belief>the overall procedural goal</task_belief>\n"
        "<reasoning>how the sequence supports that goal</reasoning>"
    )


PROJECTION_RULES = (
    "Rules:\n"
    "- Project BOTH the reasoning and the task belief to time t.\n"
    "- Do NOT use any object, ingredient, or dish that first appears only in the future.\n"
    "- If the evidence at t is weak, LOWER the specificity of the belief.\n"
    "- Do NOT mention the GT next action (or any future action) as an already-observed fact.\n"
    "- The GT next action is the OUTPUT TARGET, never cited as prompt evidence.\n"
    "- The reasoning must connect the available past evidence, the projected belief, and the\n"
    "  plausibility of the GT action naturally.\n"
    "Allowed example:\n"
    "  A bowl is visible and prior actions suggest food preparation, so taking the bowl is a\n"
    "  plausible next step toward preparing a meal.\n"
    "Forbidden example:\n"
    "  The task is making spaghetti because future actions show boiling pasta. The next action\n"
    "  is take bowl because that is what actually happens next."
)


def projection_prompt(raw_trace: str, memory_context: str, candidates: list[dict],
                      gt_verb: str, gt_noun: str) -> str:
    """§6. raw hindsight + past-only context 로 reasoning+belief 를 시점 t 로 projection."""
    cand = "\n".join(f'- {{"verb": "{c.get("verb","")}", "noun": "{c.get("noun","")}"}}'
                     for c in candidates)
    return (
        "You will rewrite a hindsight trace so that it is justified ONLY by the information\n"
        "available at the current time t (a past-grounded projection).\n\n"
        f"Hindsight (may contain future info — DO NOT copy future specifics):\n{raw_trace}\n\n"
        f"Action history available at t:\n{memory_context}\n\n"
        f"World-model candidates at t:\n{cand}\n\n"
        f"Target next action (OUTPUT ONLY — never cite as observed): {gt_verb} {gt_noun}\n\n"
        f"{PROJECTION_RULES}\n\n"
        "Return exactly:\n"
        "<reasoning>...</reasoning>\n<task_belief>...</task_belief>\n"
        f'<action>{{"verb": "{gt_verb}", "noun": "{gt_noun}"}}</action>'
    )


def equivalence_prompt(faa_belief: str, projected_belief: str) -> str:
    """§9. procedural intent 동치 판단 (문체 무시)."""
    return (
        f"Task A: {faa_belief}\n"
        f"Task B: {projected_belief}\n\n"
        "Do these descriptions refer to the same overall procedural goal at the same practical\n"
        "level of intent? Ignore wording and harmless paraphrases.\n"
        "Return exactly SAME, DIFFERENT, or UNCERTAIN."
    )


def parse_equivalence(text: str) -> str:
    """모델 응답 → SAME/DIFFERENT/UNCERTAIN (기본 UNCERTAIN)."""
    t = (text or "").strip().upper()
    for label in ("DIFFERENT", "UNCERTAIN", "SAME"):   # DIFFERENT 우선(SAME 이 substring)
        if re.search(rf"\b{label}\b", t):
            return label
    return "UNCERTAIN"


# ─────────────────────────── teacher 인터페이스 ───────────────────────────

class TeacherProtocol(Protocol):
    def infer_raw_trace(self, gt_trajectory: list[dict]) -> str: ...
    def project_full_trace(self, raw_trace: str, memory_context: str,
                           candidates: list[dict], gt_verb: str, gt_noun: str) -> Optional[Trace]: ...
    def equivalence(self, faa_belief: str, projected_belief: str) -> str: ...


@dataclass
class FrozenVLMTeacher:
    """frozen base VLM 래퍼 (GPU). model/processor 는 서버에서 주입.

    build_teacher() 로 생성. base Qwen3-VL-Instruct 를 FAA LoRA 없이 로드해
    세 프롬프트를 greedy 로 생성한다. 결정론(temperature 0)으로 preference target 안정화.
    """
    model: object = None
    processor: object = None
    max_new_tokens: int = 512

    def _generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        # 서버 전용 경로. 여기서 무거운 import 를 지역화해 로직 모듈이 GPU 없이 import 되게 한다.
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True)

    def infer_raw_trace(self, gt_trajectory: list[dict]) -> str:
        return self._generate(raw_hindsight_prompt(gt_trajectory))

    def project_full_trace(self, raw_trace, memory_context, candidates, gt_verb, gt_noun):
        txt = self._generate(projection_prompt(raw_trace, memory_context, candidates,
                                               gt_verb, gt_noun))
        tr = parse_full_trace(txt)
        if not (tr.reasoning and tr.belief):
            return None
        # action 은 항상 GT 로 강제 (target). 파싱 실패해도 GT 로 채워 canonical 보장.
        tr.verb, tr.noun = gt_verb, gt_noun
        tr.raw = build_full_trace(tr.reasoning, tr.belief, gt_verb, gt_noun)
        return tr

    def equivalence(self, faa_belief: str, projected_belief: str) -> str:
        return parse_equivalence(self._generate(equivalence_prompt(faa_belief, projected_belief)))


def build_teacher(model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                  max_new_tokens: int = 512) -> FrozenVLMTeacher:
    """서버에서 frozen base VLM 로드 (FAA LoRA 미적용 — base weights 그대로)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    return FrozenVLMTeacher(model=model, processor=processor, max_new_tokens=max_new_tokens)
