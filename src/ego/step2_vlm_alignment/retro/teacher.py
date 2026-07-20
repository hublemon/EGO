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

from .trace_utils import Trace, build_full_trace, canonical_action, parse_full_trace


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
                           candidates: list[dict], gt_verb: str, gt_noun: str,
                           image_path: Optional[str] = None) -> Optional[Trace]: ...
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

    def _generate(self, prompt: str, image_path: Optional[str] = None,
                  temperature: Optional[float] = None) -> str:
        # 서버 전용 경로. 여기서 무거운 import 를 지역화해 로직 모듈이 GPU 없이 import 되게 한다.
        import torch
        # 정보 경계표(§2): offline teacher 는 시점 t 관측 x≤t 접근 가능 —
        # projection 이 시각 근거 없는 환각 주장을 만들지 않도록 이미지를 실제로 전달한다.
        if image_path:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image],
                                    return_tensors="pt").to(self.model.device)
        else:
            messages = [{"role": "user", "content": prompt}]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        gen_kw = (dict(do_sample=True, temperature=temperature, top_p=0.95)
                  if temperature else dict(do_sample=False))
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **gen_kw)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True)

    def infer_raw_trace(self, gt_trajectory: list[dict]) -> str:
        return self._generate(raw_hindsight_prompt(gt_trajectory))

    def project_full_trace(self, raw_trace, memory_context, candidates, gt_verb, gt_noun,
                           image_path=None):
        # 시점 t 프레임을 보고 projection — "A bowl is visible" 류 주장이 실제 관측에 근거하게.
        txt = self._generate(projection_prompt(raw_trace, memory_context, candidates,
                                               gt_verb, gt_noun),
                             image_path=image_path)
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


# ═══════════════ B0-R1: GT-hidden gated generation (hard action gate) ═══════════════
# v2 정정 설계: teacher 가 GT 를 못 본 채 goal(미래 suffix 에서 추출, target 제외)만 받고
# reasoning/belief/action 을 공동 생성 → canonical(predicted)==canonical(GT) 일 때만 PASS.
# gemini 는 인라인 게이트가 아니라 사후 process-audit 용 (본 모듈은 전량 로컬).

import re as _re

RE_GOAL = _re.compile(r"<goal>(.*?)</goal>", _re.DOTALL)


def goal_prompt(future_suffix: list[dict], forbid: Optional[list[str]] = None) -> str:
    """미래 suffix(타깃 action 제외) → 상위 목표 1구. 누출 방지 규칙 명문화."""
    seq = " -> ".join(f"{a.get('verb','')} {a.get('noun','')}".strip() for a in future_suffix)
    extra = ""
    if forbid:
        words = ", ".join(f"'{w}'" for w in forbid)
        extra = f"- Do NOT use the words {words} anywhere in the goal.\n"
    return (
        "Here is a list of actions a person performs later in an ongoing activity "
        "(temporal order preserved):\n"
        f"{seq}\n\n"
        "State the single overall goal this activity is building toward, as ONE short noun "
        "phrase at the level of the whole task (e.g. 'meal preparation', 'washing up').\n"
        "Rules:\n"
        "- Do NOT copy any verb+noun pair from the list verbatim.\n"
        "- Do NOT name one single action as the goal.\n"
        f"{extra}"
        "Return exactly: <goal>...</goal>"
    )


def gated_trace_prompt(goal: str, memory_context: str, candidates: list[dict]) -> str:
    """GT-hidden 공동 생성: goal + past-only 근거로 reasoning/belief/action 을 스스로 도출."""
    cand = "\n".join(f'- {{"verb": "{c.get("verb","")}", "noun": "{c.get("noun","")}"}}'
                     for c in candidates)
    return (
        "You are an embodied agent at time t choosing your next action from a first-person view.\n\n"
        f"Action history up to t:\n{memory_context}\n\n"
        f"World-model candidates at t:\n{cand}\n\n"
        f"Your working goal (what your ongoing activity is building toward): {goal}\n\n"
        "Reason from the history, the current frame, and the goal. State the belief that your\n"
        "reasoning arrives at, and let that belief decide the choice. Choose FROM THE CANDIDATES ONLY.\n"
        "Return exactly:\n"
        "<reasoning>...</reasoning>\n<task_belief>...</task_belief>\n"
        '<action>{"verb": "...", "noun": "..."}</action>'
    )


def _word_leak(text: str, word: str) -> bool:
    """goal 문자열에 GT 토큰이 단어 경계로 등장하는가 (계층 noun 은 base 로도 검사)."""
    tl = (text or "").lower()
    for v in {word.lower(), word.split(":")[0].lower(), word.replace("-", " ").lower()}:
        if v and _re.search(rf"(?<![a-z]){_re.escape(v)}(?![a-z])", tl):
            return True
    return False


def _verb_forms(verb: str) -> list[str]:
    """동사 활용형 (wash → washing/washed/washes 도 누출로 간주)."""
    v = verb.lower().strip()
    forms = {v, v.replace("-", " ")}
    stem = v[:-1] if v.endswith("e") else v
    forms |= {v + "s", v + "es", v + "ed", stem + "ing", stem + "ed"}
    return [f for f in forms if f]


def goal_leaks(goal: str, gt_verb: str, gt_noun: str) -> list[str]:
    leaked = []
    if _word_leak(goal, gt_noun):
        leaked.append(gt_noun)
    if any(_word_leak(goal, f) for f in _verb_forms(gt_verb)):
        leaked.append(gt_verb)
    return leaked


class GatedTeacherMixin:
    """FrozenVLMTeacher 에 R1 경로 추가 (상속 대신 mixin — 기존 MVP 경로 불변)."""

    def extract_goal(self, future_suffix, gt_verb, gt_noun, max_retries: int = 2):
        """(goal, meta). 누출 시 금지어 명시 재시도, 끝내 누출이면 (None, meta)."""
        meta = {"retries": 0, "leaked": []}
        forbid = None
        for i in range(max_retries + 1):
            txt = self._generate(goal_prompt(future_suffix, forbid))
            m = RE_GOAL.search(txt or "")
            goal = m.group(1).strip() if m else ""
            leaks = goal_leaks(goal, gt_verb, gt_noun) if goal else ["<parse_fail>"]
            meta["retries"] = i
            if goal and not leaks:
                return goal, meta
            meta["leaked"] = leaks
            forbid = [w for w in (gt_noun.split(":")[0], gt_verb) if w]
        return None, meta

    def generate_gated_trace(self, goal, memory_context, candidates, gt_verb, gt_noun,
                             image_path=None, attempts: int = 4):
        """hard action gate: 시도 1 greedy, 이후 T0.8 샘플링. PASS 시 정규화 Trace 반환."""
        prompt = gated_trace_prompt(goal, memory_context, candidates)
        cand_keys = {canonical_action(c.get("verb"), c.get("noun")) for c in candidates}
        gt_key = canonical_action(gt_verb, gt_noun)
        preds = []
        meta = {"attempts_used": 0, "predictions": preds}
        for i in range(attempts):
            txt = self._generate(prompt, image_path=image_path,
                                 temperature=None if i == 0 else 0.8)
            tr = parse_full_trace(txt)
            meta["attempts_used"] = i + 1
            key = canonical_action(tr.verb, tr.noun) if tr.verb else None
            preds.append(key)
            if not tr.is_complete() or key not in cand_keys:
                continue
            if key == gt_key:
                tr.raw = build_full_trace(tr.reasoning, tr.belief, tr.verb, tr.noun)
                return tr, meta
        return None, meta


class GatedFrozenVLMTeacher(GatedTeacherMixin, FrozenVLMTeacher):
    pass


def build_gated_teacher(model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                        max_new_tokens: int = 512) -> GatedFrozenVLMTeacher:
    base = build_teacher(model_name, max_new_tokens)
    return GatedFrozenVLMTeacher(model=base.model, processor=base.processor,
                                 max_new_tokens=max_new_tokens)
