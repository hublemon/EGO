"""공용 VLM 래퍼 — Qwen3-VL 로딩·프레임 추출·3태그 프롬프트·파싱.

base_trace / teacher / projection / eval 이 공유한다. step2_vlm_alignment 미참조.
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import torch

from ego.step2_retrospection.contracts import SupportSample

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
N_FRAMES = 8          # 관측창 8s에서 1fps
FRAME_SHORT_SIDE = 336

# OOM_OPT_GUARDS: 프레임 사전추출 캐시 (tools/oom_opt/frame_extractor.py 가 생성).
# 히트 시 decord 디코드를 건너뛰어 RAM 진동·reshape 크래시를 원천 제거. 미스는 on-the-fly 폴백.
import os as _os_cache
FRAME_CACHE_DIR = _os_cache.environ.get(
    "FRAME_CACHE_DIR",
    _os_cache.path.join(_os_cache.environ.get("RETRO3_RUNS", "runs/cesft_v2"), "frame_cache"))


def _load_cached(rec: dict, n_frames: int):
    """프레임 캐시 히트 → PIL 리스트, 미스/불량 → None (on-the-fly 폴백)."""
    from PIL import Image
    sid = rec.get("sample_id"); uid = rec.get("video_uid")
    if not sid or not uid:
        return None
    d = _os_cache.path.join(FRAME_CACHE_DIR, uid, sid)
    if not _os_cache.path.isdir(d):
        return None
    paths = [_os_cache.path.join(d, f"f{i}.jpg") for i in range(n_frames)]
    if not all(_os_cache.path.isfile(p) for p in paths):
        return None
    imgs = [Image.open(p).convert("RGB") for p in paths]
    if len({im.size for im in imgs}) != 1:
        return None
    return imgs

# 시간 계약별 '다음 행동 시점' 문구 — retro3(start−1s)는 기본값, retro4(end−1s 가변
# horizon)는 체인이 RETRO_NEXT_GAP_TEXT 환경변수로 주입 (config contract.next_gap_text).
import os as _os
NEXT_GAP_TEXT = _os.environ.get("RETRO_NEXT_GAP_TEXT", "starting 1 second after the last frame")

SYSTEM_PROMPT = (
    "You are an egocentric activity assistant. You see frames from the last 8 seconds of a "
    "first-person video, a list of actions the person already COMPLETED, and a shuffled list "
    "of candidate next actions. Each action is 'verb noun'. Exactly ONE candidate is what the "
    f"person does next ({NEXT_GAP_TEXT}).\n"
    "Respond in EXACTLY this format:\n"
    "<reasoning>\nCompare the candidates against the visual scene and the completed-action "
    "history. 3-6 sentences.\n</reasoning>\n"
    "<task_belief>\nOne sentence: the local procedure or subgoal the person is currently in. "
    "Do NOT name the chosen next action verbatim.\n</task_belief>\n"
    "<action>\nverb noun\n</action>\n"
    "The <action> line must copy one candidate EXACTLY as written."
)

TAG_RE = {
    "reasoning": re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.S),
    "task_belief": re.compile(r"<task_belief>\s*(.*?)\s*</task_belief>", re.S),
    "action": re.compile(r"<action>\s*(.*?)\s*</action>", re.S),
}


def load_model(model_name: str = MODEL_NAME, dtype=torch.bfloat16, device_map: str = "cuda"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype, device_map=device_map)
    model.eval()
    return model, processor


@lru_cache(maxsize=4)  # 16→4: 1920×1440 장영상 리더가 무거워 호스트 RAM 스파이크 방지 (S3 77% OOM-kill 대응)
# 2026-07-24: num_threads 2→4 시도했으나 decord가 프레임을 잘못된 크기로 디코드 → vision reshape 크래시.
#   known-good 값(num_threads=2, workers=4)으로 복원. decode 병렬화는 안정성 리스크로 폐기.
def _video_reader(path: str):
    import decord
    decord.bridge.set_bridge("torch")
    return decord.VideoReader(path, num_threads=2)


# decord VideoReader는 스레드 안전이 아님 — 영상(path)별 락으로 직렬화하고,
# 서로 다른 영상은 병렬 디코딩 허용 (extract_frames_parallel).
_vid_locks: dict[str, threading.Lock] = {}
_vid_locks_guard = threading.Lock()


def _video_lock(path: str) -> threading.Lock:
    with _vid_locks_guard:
        return _vid_locks.setdefault(path, threading.Lock())


def close_readers() -> None:
    """decord 소멸자 스레딩 크래시("pure virtual method called") 방지 —
    torch teardown 전에 명시적으로 리더를 해제한다. 각 러너 main() 끝에서 호출."""
    _video_reader.cache_clear()
    import gc
    gc.collect()


def extract_frames(video_root: Path, video_uid: str, start_sec: float, end_sec: float,
                   n_frames: int = N_FRAMES):
    """관측창 [start,end]에서 n_frames 균등 샘플 → PIL 이미지 리스트 (짧은 변 리사이즈)."""
    import decord
    from PIL import Image
    decord.bridge.set_bridge("torch")  # thread-local — 워커 스레드에서도 매번 보장
    path = str(video_root / f"{video_uid}.mp4")
    with _video_lock(path):  # 같은 영상 동시 접근 직렬화 (리더 비스레드안전)
        vr = _video_reader(path)
        fps = vr.get_avg_fps()
        n_total = len(vr)
        t0, t1 = max(0.0, start_sec), max(start_sec + 1e-3, end_sec)
        idxs = [min(n_total - 1, max(0, round((t0 + (t1 - t0) * i / max(1, n_frames - 1)) * fps)))
                for i in range(n_frames)]
        batch = vr.get_batch(idxs)  # (n, H, W, 3) uint8 torch
    imgs = []  # PIL 변환/리사이즈는 락 밖 (병렬 허용)
    for i in range(batch.shape[0]):
        im = Image.fromarray(batch[i].numpy())
        w, h = im.size
        s = FRAME_SHORT_SIDE / min(w, h)
        if s < 1.0:
            im = im.resize((round(w * s), round(h * s)))
        imgs.append(im)
    # OOM_OPT_GUARDS: 개수(짝수)·크기 동일 검증 — 불량 디코드를 model forward 앞에서 raise→skip_decode
    if len(imgs) != n_frames or len({im.size for im in imgs}) != 1:
        raise ValueError(f"frame validation failed: n={len(imgs)} sizes={len({im.size for im in imgs})}")
    return imgs


def extract_frames_parallel(video_root: Path, recs: list[dict], n_frames: int = N_FRAMES,
                            workers: int = 2) -> list[tuple]:  # OOM_OPT_GUARDS 4→2
    """chunk 내 샘플들의 프레임을 스레드 병렬 추출. recs[i] ↔ 반환 [i] 짝 유지.
    반환 원소: (imgs, None) 성공 | (None, exc) 실패 (영상 깨짐 등 — 호출부에서 태깅)."""
    if not recs:
        return []

    def one(rec: dict):
        try:
            cached = _load_cached(rec, n_frames)  # OOM_OPT_GUARDS: 캐시 히트 시 디코드 생략
            if cached is not None:
                return cached, None
            return extract_frames(video_root, rec["video_uid"],
                                  rec["obs_start_sec"], rec["obs_end_sec"], n_frames), None
        except Exception as e:
            return None, e

    with ThreadPoolExecutor(max_workers=min(workers, len(recs))) as ex:
        return list(ex.map(one, recs))


def prefetch_chunks(video_root: Path, todo: list[dict], batch_size: int,
                    n_frames: int = N_FRAMES, workers: int = 2):  # OOM_OPT_GUARDS 4→2
    """todo를 batch_size chunk로 잘라 (chunk, frames)를 yield하는 더블 버퍼 제너레이터.
    현재 chunk를 소비(GPU 생성)하는 동안 다음 chunk의 프레임 추출이 백그라운드로 진행 —
    CPU 디코딩 시간이 GPU 생성 뒤로 숨는다. frames[i] = (imgs|None, err|None)."""
    chunks = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    if not chunks:
        return
    with ThreadPoolExecutor(max_workers=1) as pf:
        fut = pf.submit(extract_frames_parallel, video_root, chunks[0], n_frames, workers)
        for ci, chunk in enumerate(chunks):
            frames = fut.result()
            if ci + 1 < len(chunks):
                fut = pf.submit(extract_frames_parallel, video_root, chunks[ci + 1], n_frames, workers)
            yield chunk, frames


def fmt_history(s: SupportSample | dict, max_items: int = 15) -> str:
    hist = s["history"] if isinstance(s, dict) else [a.__dict__ for a in s.history]
    if not hist:
        return "(no completed actions yet)"
    tail = hist[-max_items:]
    lines = [f"- {a['verb']} {a['noun']}" for a in tail]
    if len(hist) > max_items:
        lines.insert(0, f"(... {len(hist) - max_items} earlier actions omitted)")
    return "\n".join(lines)


def fmt_candidates(cands: list[str]) -> str:
    return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))


def user_prompt(rec: dict) -> str:
    return (
        f"Completed actions so far (oldest to newest):\n{fmt_history(rec)}\n\n"
        f"Candidate next actions (shuffled):\n{fmt_candidates(rec['candidates'])}\n\n"
        "Which candidate is the next action? Follow the required format."
    )


def build_messages(rec: dict, images) -> list[dict]:
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": user_prompt(rec)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def parse_trace(text: str) -> dict | None:
    out = {}
    for tag, rx in TAG_RE.items():
        m = rx.search(text)
        if not m:
            return None
        out[tag] = m.group(1).strip()
    return out


def match_candidate(action_text: str, candidates: list[str]) -> str | None:
    """exact → 정규화(소문자·공백 축약) → 유일 부분일치 순. 실패 시 None."""
    a = action_text.strip()
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
    # 토큰 집합 겹침 최대가 유일하면 채택 (taxonomy 접미사 "_(...)" 누락 흡수)
    strip = lambda x: re.sub(r"_\([^)]*\)", "", norm(x))  # noqa: E731
    sa = set(strip(a).split())
    scores = [(len(sa & set(strip(c).split())), c) for c in candidates]
    best = max(s for s, _ in scores)
    top = [c for s, c in scores if s == best and best > 0]
    if len(top) == 1:
        return top[0]
    return None


@torch.no_grad()
def generate(model, processor, messages: list[dict], max_new_tokens: int = 320,
             do_sample: bool = False, temperature: float = 1.0) -> str:
    return generate_batch(model, processor, [messages], max_new_tokens, do_sample, temperature)[0]


@torch.no_grad()
def generate_batch(model, processor, messages_list: list[list[dict]], max_new_tokens: int = 320,
                   do_sample: bool = False, temperature: float = 1.0) -> list[str]:
    """좌측 패딩 배치 생성 — 처리율 핵심."""
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    images = [[c["image"] for msg in m for c in msg["content"]
               if isinstance(c, dict) and c.get("type") == "image"] for m in messages_list]
    has_img = any(images)
    tok = processor.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        inputs = processor(text=texts, images=images if has_img else None,
                           return_tensors="pt", padding=True).to(model.device)
    finally:
        tok.padding_side = old_side
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                         temperature=temperature if do_sample else None,
                         pad_token_id=tok.eos_token_id)
    new = gen[:, inputs["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)
