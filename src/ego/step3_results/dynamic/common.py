"""Closed-Loop Dynamic Planning — 공통 계약 · 프롬프트 · 파싱.

VPA(open-loop, 한 시점에서 T개 예측)와 달리 **한 영상의 연속된 결정지점을 따라가며**
매 스텝 (WM top-10 제시 → VLM 선택)을 반복한다. 결정적 차이는 입력 히스토리다:

    VPA / step2 eval : history = **ground-truth** 완료 action
    closed loop      : history = **VLM 자신이 앞서 고른** action + 자신의 이전 task_belief

GT는 채점에만 쓰고 프롬프트에 **절대** 넣지 않는다 (`oracle_gt_hist` arm은 의도적 대조군).

시간 계약은 VPA v2 와 **동일 모듈을 임포트해** 강제한다 — 같은 프레임 캐시를 공유하고
표 간 비교가 성립해야 하기 때문이다:
    obs_end   = target_start - 1s,  obs_start = obs_end - 4s,  8프레임@2fps, 짧은 변 336px
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ego.step3_results.vpa.v2.common import (  # noqa: F401 — 계약 단일 출처
    FRAME_SHORT_SIDE, N_FRAMES, OBS_WINDOW_SEC, SAFETY_GAP_SEC,
    dump_json, load_json, normalize_label, observation_window, read_jsonl,
)

# ── 루프 계약 ────────────────────────────────────────────────────────────────
WM_ALIGN_GAP_SEC = 1.5   # WM 후보가 산출된 원본 창과 우리 창의 정렬 허용 오차.
                         # context_val 의 (target_start - obs_end) 가 이보다 크면 후보가
                         # 낡은 관측에서 나온 것이라 closed loop 의 "새 관찰" 전제가 깨진다.
MIN_TARGET_START_SEC = 5.5   # 4초 창이 영상 시작에 잘리지 않도록
BELIEF_CARRY = 3         # 프롬프트에 싣는 직전 belief 개수
HISTORY_MAX = 15         # step2 fmt_history 와 동일
MAX_NEW_TOKENS = 320     # step2 기본값 (3태그 출력)

TRACE_TAGS = ("reasoning", "task_belief", "action")
_TAG_RE = {t: re.compile(rf"<{t}>\s*(.*?)\s*</{t}>", re.S) for t in TRACE_TAGS}

# arm 정의 — history 출처와 belief 캐리 여부만 다르고 나머지는 완전히 동일하다.
ARMS: dict[str, dict] = {
    # 본 실험: 자기 예측을 히스토리로, 자기 belief 를 캐리
    "ego_closed": {"history": "pred", "belief": True,
                   "why": "닫힌 루프 — 동적 planning 주장 대상"},
    # belief 기여분 분리: 히스토리는 자기 예측이되 belief 는 넣지 않는다
    "ego_nobelief": {"history": "pred", "belief": False,
                     "why": "ego_closed − 이 arm = belief 캐리의 기여분"},
    # 오차 누적 비용: 히스토리를 GT 로 대체 (= step2/VPA 조건). 프레임·후보는 동일.
    "oracle_gt_hist": {"history": "gt", "belief": False,
                       "why": "이 arm − ego_nobelief = 자기 히스토리 오염의 비용"},
}


# ── 프롬프트 ────────────────────────────────────────────────────────────────
def system_prompt(arm: str) -> str:
    """step2 학습 프롬프트(`step2_retrospection/vlm.SYSTEM_PROMPT`)의 3태그 계약을 유지한다.

    LoRA 가 그 형식으로 학습됐으므로 태그·문장 골격을 바꾸지 않고, 창 길이(8s→4s)와
    closed-loop 에서 새로 들어가는 입력(goal · 자기 선택 이력 · 자기 belief)만 반영한다.
    """
    cfg = ARMS[arm]
    hist_txt = ("a list of actions YOU yourself selected at the earlier decision points of this "
                "same video" if cfg["history"] == "pred" else
                "a list of actions the person already COMPLETED")
    belief_txt = (", your own task beliefs from those earlier decision points" if cfg["belief"] else "")
    return (
        "You are an egocentric activity assistant following one video from start to end, one "
        f"decision point at a time. You see frames from the last {OBS_WINDOW_SEC:.0f} seconds of a "
        "first-person video, the overall GOAL of the video, "
        f"{hist_txt}{belief_txt}, and a shuffled list of candidate next actions. "
        "Each action is 'verb noun'. Exactly ONE candidate is what the person does next "
        f"(starting {SAFETY_GAP_SEC:.0f} second after the last frame).\n"
        "Respond in EXACTLY this format:\n"
        "<reasoning>\nCompare the candidates against the visual scene, the goal and the action "
        "history. 3-6 sentences.\n</reasoning>\n"
        "<task_belief>\nOne sentence: the local procedure or subgoal the person is currently in. "
        "Do NOT name the chosen next action verbatim.\n</task_belief>\n"
        "<action>\nverb noun\n</action>\n"
        "The <action> line must copy one candidate EXACTLY as written."
    )


def fmt_history(actions: list[str], arm: str) -> str:
    if not actions:
        return "(no actions yet — this is the first decision point of the video)"
    tail = actions[-HISTORY_MAX:]
    lines = [f"- {a}" for a in tail]
    if len(actions) > HISTORY_MAX:
        lines.insert(0, f"(... {len(actions) - HISTORY_MAX} earlier actions omitted)")
    return "\n".join(lines)


def fmt_beliefs(beliefs: list[tuple[int, str]]) -> str:
    if not beliefs:
        return "(none yet)"
    return "\n".join(f"- (decision point {i}) {b}" for i, b in beliefs[-BELIEF_CARRY:])


def user_prompt(step: dict, goal: str, history: list[str], beliefs: list[tuple[int, str]],
                arm: str) -> str:
    cfg = ARMS[arm]
    head = ("Actions YOU selected so far in this video (oldest to newest)"
            if cfg["history"] == "pred" else "Completed actions so far (oldest to newest)")
    cands = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(step["candidates"]))
    parts = [f"OVERALL GOAL OF THIS VIDEO: {goal or '(not specified)'}",
             f"{head}:\n{fmt_history(history, arm)}"]
    if cfg["belief"]:
        parts.append(f"Your task beliefs at the previous decision points:\n{fmt_beliefs(beliefs)}")
    parts.append(f"Candidate next actions (shuffled):\n{cands}")
    parts.append("Which candidate is the next action? Follow the required format.")
    return "\n\n".join(parts)


# ── 파싱 ────────────────────────────────────────────────────────────────────
def parse_trace(text: str) -> dict | None:
    """3태그 추출. 하나라도 없으면 None (malformed)."""
    out = {}
    for tag, rx in _TAG_RE.items():
        m = rx.search(text or "")
        if not m:
            return None
        out[tag] = m.group(1).strip()
    return out


def match_candidate(action_text: str, candidates: list[str]) -> str | None:
    """exact → 정규화 → 유일 부분일치 → 토큰겹침. step2 `vlm.match_candidate` 와 동일 규칙."""
    a = (action_text or "").strip()
    if a in candidates:
        return a
    norm = lambda x: re.sub(r"\s+", " ", x.lower().strip())  # noqa: E731
    na = norm(a)
    exact = [c for c in candidates if norm(c) == na]
    if len(exact) == 1:
        return exact[0]
    part = [c for c in candidates if norm(c) in na or na in norm(c)]
    if len(part) == 1:
        return part[0]
    strip = lambda x: re.sub(r"_\([^)]*\)", "", norm(x))  # noqa: E731
    sa = set(strip(a).split())
    scores = [(len(sa & set(strip(c).split())), c) for c in candidates]
    best = max(s for s, _ in scores)
    top = [c for s, c in scores if s == best and best > 0]
    return top[0] if len(top) == 1 else None


def force_into_candidates(action_text: str, candidates: list[str]) -> tuple[str, bool]:
    """계약상 선택은 반드시 top-10 안에 있어야 한다. 매칭 실패 시 difflib 최근접으로
    강제 투영하고 `forced=True` 로 기록한다 — WM top-1 로 폴백하지 **않는다**
    (그러면 baseline 행동이 ours 에 섞여 지표가 오염된다)."""
    m = match_candidate(action_text, candidates)
    if m is not None:
        return m, False
    import difflib
    near = difflib.get_close_matches(normalize_label(action_text),
                                     [normalize_label(c) for c in candidates], n=1, cutoff=0.0)
    if near:
        for c in candidates:
            if normalize_label(c) == near[0]:
                return c, True
    return candidates[0], True


def wm_top1(step: dict) -> str:
    """후보는 셔플 저장이므로 점수 최대값으로 top-1 을 복원한다."""
    i = max(range(len(step["wm_scores"])), key=lambda k: step["wm_scores"][k])
    return step["candidates"][i]


def append_jsonl(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
