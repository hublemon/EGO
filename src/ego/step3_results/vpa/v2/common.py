"""VPA v2 공통 계약 — 창 정의 · 프롬프트 · 어휘 · 라벨 정규화.

모든 arm(frontier / 로컬 VLM / blind / WM-only / baseline)이 **이 모듈 하나**를 공유한다.
arm마다 프롬프트나 창이 다르면 표 내부 비교가 무너지므로, 분기는 여기에만 둔다.

시간 계약 (2026-07-25_vpa_v2_frame_conditioned_plan.md §3 F1):
    obs_end   = target_start - SAFETY_GAP_SEC   (1s 안전 간격 — 온셋 누출 차단)
    obs_start = obs_end - OBS_WINDOW_SEC        (4s 관측창)
    관측 프레임은 예측 대상 action이 **시작하기 전**에서 끝난다. VLaMP 공식 구현의
    `[start-2s, start+2s)` pre-window(미래 2초 침범)를 의도적으로 따르지 않는다.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# ── 시간·프레임 계약 (전 arm 공통, 변경 시 전 arm 재실행 필요) ────────────────
OBS_WINDOW_SEC = 4.0      # 관측창 길이
SAFETY_GAP_SEC = 1.0      # obs_end 와 target_start 사이 안전 간격
N_FRAMES = 8              # 4s / 8frames = 2 fps
FRAME_SHORT_SIDE = 336    # step2(vlm.py)와 동일 — 공정성·토큰 비용
GAP_CAP_SEC = 5.0         # anticipation gap 상한 (plan §4 D2)
DEFAULT_HORIZONS = (3, 4)

REPO = Path(__file__).resolve().parents[5]
TAXONOMY = REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_verbnoun_taxonomy.json"
LABELS_CSV = REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_step_labels.csv"


def normalize_label(text: str) -> str:
    """라벨 표면 노이즈 제거 — GT·어휘·모델 예측에 동일 적용해야 채점이 공정하다."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .;:\t\n\r")


def observation_window(target_start_sec: float) -> tuple[float, float]:
    """예측 대상 action의 시작 시각 → (obs_start, obs_end). 경계-안전 보장."""
    obs_end = target_start_sec - SAFETY_GAP_SEC
    return max(0.0, obs_end - OBS_WINDOW_SEC), obs_end


def action_vocab() -> list[str]:
    """GoalStep verb×noun action 어휘(293). context_val(step2)과 동일 계보 —
    라벨 계보를 통일해야 step2 모델과 표본·정답이 맞는다 (plan §4 D4)."""
    import csv

    tax = json.loads(TAXONOMY.read_text())
    verbs, nouns = tax["verbs"], tax["nouns"]
    acts = set()
    with open(LABELS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            acts.add(normalize_label(f"{verbs[int(row['verb_label'])]} {nouns[int(row['noun_label'])]}"))
    return sorted(acts)


def candidates_for(sample: dict, vocab: list[str], mode: str) -> list[str]:
    """프롬프트에 실을 후보 목록 (mode='vocab' 기본).

    mode='vocab'      : 전체 행동 어휘(293). 원본 VPA 정의(전체 행동 공간에서의 계획)에 맞고,
                        전 arm이 같은 라벨 공간을 쓰므로 표 안에서 직접 비교가 성립한다.
    mode='wm10_first' : **EGO 배포 형태** — 1번째 action 만 WM top-10 에서 고르고,
                        2번째 이후는 전체 어휘에서 자유 추론. WM 이 바로 다음 action 하나만
                        예측하는 구조를 그대로 반영한다. 후보 목록 자체는 전체 어휘를 주고,
                        1스텝 제약은 build_prompt 가 별도 문장으로 건다.
    """
    return vocab


def build_prompt(sample: dict, vocab: list[str], horizon: int, *, with_frames: bool,
                 candidate_mode: str = "vocab") -> tuple[str, str]:
    """(system, user) — 전 arm 공통. with_frames=False 는 blind control 전용으로
    프레임 관련 문장만 제거하고 나머지 텍스트는 **완전히 동일**하게 유지한다
    (프레임 기여분만 분리하기 위해 다른 변인을 두지 않는다)."""
    goal = sample.get("goal_text") or "(not specified)"
    observed = sample.get("observed_actions") or []
    obs_txt = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(observed[-15:])) or "  (none recorded)"
    cands = candidates_for(sample, vocab, candidate_mode)
    vocab_txt = "\n".join(f"- {v}" for v in cands)

    if with_frames:
        source = (
            f"{N_FRAMES} frames sampled at {N_FRAMES / OBS_WINDOW_SEC:.0f} fps from the last "
            f"{OBS_WINDOW_SEC:.0f} seconds of first-person video, ending "
            f"{SAFETY_GAP_SEC:.0f} second before the next action begins"
        )
    else:
        source = "no video (text context only)"

    system = (
        "You are a procedural planning assistant for egocentric cooking videos. "
        f"You are given: {source}; the goal; and the actions the camera-wearer has already "
        f"COMPLETED. Predict the next {horizon} actions, in temporal order, starting with the "
        "action that begins immediately after the observation ends. Each action is "
        "'<verb> <noun>'. You MUST copy labels verbatim from the candidate list and output "
        f"EXACTLY {horizon} of them as a JSON array of strings, and nothing else."
    )
    wm_block = ""
    if candidate_mode == "wm10_first":
        # EGO 배포 형태: world model 이 '바로 다음' action 후보 10개를 제시하고 LM 이 그중 하나를
        # 고른다. WM 은 그 이후 스텝을 예측하지 못하므로 2번째부터는 전체 어휘에서 자유 추론.
        wm = "\n".join(f"- {c}" for c in (sample.get("wm_candidates") or []))
        if wm:
            wm_block = (
                "\nA video world model has ranked these 10 candidates for the FIRST next action "
                f"only:\n{wm}\n"
                "Your 1st predicted action MUST be one of these 10. "
                "Actions 2 onward are unconstrained — pick them from the full candidate list.\n"
            )

    user = (
        f"GOAL: {goal}\n\n"
        f"ACTIONS ALREADY COMPLETED (in order):\n{obs_txt}\n\n"
        f"CANDIDATE ACTION LABELS (choose only from these):\n{vocab_txt}\n"
        f"{wm_block}\n"
        f"Predict the next {horizon} actions as a JSON array of exactly {horizon} labels "
        "copied verbatim from the candidate list, in order. Output ONLY the JSON array."
    )
    return system, user


def parse_prediction(content: str, horizon: int) -> list[str]:
    """모델 응답에서 JSON 배열 추출 — 코드펜스·잡설을 관용한다."""
    if not content:
        return []
    txt = content.strip()
    if "```" in txt:
        for seg in txt.split("```"):
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
    except json.JSONDecodeError:
        return []
    return [str(x) for x in arr][:horizon] if isinstance(arr, list) else []


def map_to_vocab(label: str, vocab_set: set[str], vocab_list: list[str], cutoff: float = 0.6) -> str:
    """어휘 밖 예측 매핑: 정확일치 → 정규화 → difflib 최근접. 전 arm 동일 적용."""
    import difflib

    norm = normalize_label(label)
    if norm in vocab_set:
        return norm
    near = difflib.get_close_matches(norm, vocab_list, n=1, cutoff=cutoff)
    return near[0] if near else norm


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
