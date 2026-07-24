#!/usr/bin/env python3
"""vlm.py 가드 + 프레임 캐시 로더 분기를 멱등 적용 (핸드오프 §3 가드 3개 + §3-1 캐시).

**theta_ce 완료 후에만** 실행해야 안전 (실행 중이던 프로세스는 이미 vlm 을 import 했으므로
디스크 수정은 그 프로세스에 무영향; 이후 orchestrator 가 새로 띄우는 스테이지(r0/eval 등)만
패치된 vlm 을 import → 크래시-프루프). post_theta_ce_hook.sh 가 호출한다.

적용 항목:
  1. extract_frames_parallel / prefetch_chunks 의 기본 workers 4→2 (디코드 동시성↓ = RAM 진동 상단↓).
  2. extract_frames 반환 전 프레임 검증(개수 짝수·크기 동일) → 불량 시 raise
     → 호출부 try/except 가 skip_decode 로 태깅 (reshape 크래시를 model forward 앞에서 차단).
  3. 프레임 캐시 히트 분기(_load_cached): frame_cache/<uid>/<sid>/f{i}.jpg 존재 시 디코드 생략,
     미스는 on-the-fly 폴백 (무손실 점진 전환).

멱등: 파일에 'OOM_OPT_GUARDS' 센티넬이 있으면 재적용하지 않음. 백업 vlm.py.oomopt.bak 생성.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

VLM = Path(__file__).resolve().parents[2] / "src" / "ego" / "step2_retrospection" / "vlm.py"
SENTINEL = "OOM_OPT_GUARDS"

CACHE_BLOCK = '''FRAME_SHORT_SIDE = 336

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
    return imgs'''

# (old, new) 치환 목록 — 현재 vlm.py 정확 매칭
EDITS = [
    # 1. 캐시 블록 삽입 (FRAME_SHORT_SIDE 정의 지점)
    ("FRAME_SHORT_SIDE = 336", CACHE_BLOCK),
    # 2. workers 기본 4→2
    ("def extract_frames_parallel(video_root: Path, recs: list[dict], n_frames: int = N_FRAMES,\n"
     "                            workers: int = 4) -> list[tuple]:",
     "def extract_frames_parallel(video_root: Path, recs: list[dict], n_frames: int = N_FRAMES,\n"
     "                            workers: int = 2) -> list[tuple]:  # OOM_OPT_GUARDS 4→2"),
    ("def prefetch_chunks(video_root: Path, todo: list[dict], batch_size: int,\n"
     "                    n_frames: int = N_FRAMES, workers: int = 4):",
     "def prefetch_chunks(video_root: Path, todo: list[dict], batch_size: int,\n"
     "                    n_frames: int = N_FRAMES, workers: int = 2):  # OOM_OPT_GUARDS 4→2"),
    # 3. 캐시 히트 분기 (one(rec))
    ('''    def one(rec: dict):
        try:
            return extract_frames(video_root, rec["video_uid"],
                                  rec["obs_start_sec"], rec["obs_end_sec"], n_frames), None
        except Exception as e:
            return None, e''',
     '''    def one(rec: dict):
        try:
            cached = _load_cached(rec, n_frames)  # OOM_OPT_GUARDS: 캐시 히트 시 디코드 생략
            if cached is not None:
                return cached, None
            return extract_frames(video_root, rec["video_uid"],
                                  rec["obs_start_sec"], rec["obs_end_sec"], n_frames), None
        except Exception as e:
            return None, e'''),
    # 4. 프레임 검증 (extract_frames 반환 전)
    ('''        imgs.append(im)
    return imgs''',
     '''        imgs.append(im)
    # OOM_OPT_GUARDS: 개수(짝수)·크기 동일 검증 — 불량 디코드를 model forward 앞에서 raise→skip_decode
    if len(imgs) != n_frames or len({im.size for im in imgs}) != 1:
        raise ValueError(f"frame validation failed: n={len(imgs)} sizes={len({im.size for im in imgs})}")
    return imgs'''),
]


def main():
    text = VLM.read_text()
    if SENTINEL in text:
        print(f"[apply_vlm_guards] 이미 적용됨(센티넬 존재) — skip: {VLM}")
        return 0
    for old, new in EDITS:
        if old not in text:
            print(f"[apply_vlm_guards] ERROR: 매칭 실패 — vlm.py 가 예상과 다름. 중단.\n---\n{old[:120]}...")
            return 2
        if text.count(old) != 1:
            print(f"[apply_vlm_guards] ERROR: 중복 매칭({text.count(old)}) — 중단: {old[:80]}")
            return 3
    shutil.copy2(VLM, VLM.with_suffix(".py.oomopt.bak"))
    for old, new in EDITS:
        text = text.replace(old, new, 1)
    # 컴파일 검증 후 기록
    try:
        compile(text, str(VLM), "exec")
    except SyntaxError as e:
        print(f"[apply_vlm_guards] ERROR: 패치 후 SyntaxError — 미적용: {e}")
        return 4
    VLM.write_text(text)
    print(f"[apply_vlm_guards] 적용 완료 (백업 {VLM.with_suffix('.py.oomopt.bak').name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
