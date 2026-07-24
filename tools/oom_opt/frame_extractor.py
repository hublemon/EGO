#!/usr/bin/env python3
"""RAM-경계 프레임 사전 추출기 (핸드오프 §3-1).

학습/평가 풀의 8프레임@336 을 오프라인 1회 추출·검증해 디스크(JPEG)에 저장한다.
학습/평가 로더(vlm.extract_frames 의 캐시-히트 분기)가 디코드 없이 이미지를 읽어
decord 진동(±25G RAM)·reshape 크래시를 원천 제거한다.

절대 안전 규약 (theta_ce 등 GPU 잡과 공존 — CPU-only, RAM 게이트):
  - worker 1, 영상/샘플 1개씩 → 저장 → 즉시 del/gc (per-sample RAM 상수).
  - 매 샘플 전 cgroup 여유 RAM 확인, < RAM_FLOOR_GB(기본 60G)면 여유 회복까지 일시정지.
  - torch/CUDA 미사용 (decord native bridge + PIL) — GPU 무접촉.
  - 프레임 개수(짝수=N_FRAMES)·크기 동일 검증 1회. 불량 → manifest 에 기록, 스킵.

출력:
  runs/cesft_v2/frame_cache/<video_uid>/<sample_id>/f{0..N-1}.jpg
  runs/cesft_v2/frame_cache/manifest.jsonl   (sample_id → {dir, n, ok, reason})

멱등: 이미 ok 로 캐시된 sample_id 는 스킵. 중단 후 재기동 무손실.
사용: PYTHON tools/oom_opt/frame_extractor.py [--pool train|val|both] [--limit N]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
RUN = Path(os.environ.get("RETRO3_RUNS", "runs/cesft_v2"))
CACHE = RUN / "frame_cache"
MANIFEST = CACHE / "manifest.jsonl"
N_FRAMES = 8
SHORT = 336
RAM_FLOOR_GB = float(os.environ.get("FRAME_RAM_FLOOR_GB", "60"))
CFG = "configs/step2_retrospection/cesft_v2.yaml"


def log(msg: str):
    line = f"[frame_extract {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    with open(RUN / "logs" / "frame_extract.log", "a") as f:
        f.write(line + "\n")


# cgroup (limit, memory.stat, current) — v2 우선, v1 폴백.
_CG_PATHS = (("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory.stat",
              "/sys/fs/cgroup/memory.current"),
             ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
              "/sys/fs/cgroup/memory/memory.stat",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes"))
# 회수 불가(= 실제로 압박을 만드는) 항목. dirty/writeback 은 flush 전까지 못 비우므로 포함.
_HARD_KEYS_V2 = ("anon", "unevictable", "slab_unreclaimable", "file_dirty", "file_writeback")
_HARD_KEYS_V1 = ("total_rss", "total_unevictable", "total_dirty", "total_writeback")


def _read_mem_stat(path: str) -> dict:
    out = {}
    try:
        with open(path) as f:
            for ln in f:
                k, _, v = ln.strip().partition(" ")
                try:
                    out[k] = int(v)
                except ValueError:
                    continue
    except Exception:
        pass
    return out


def _hard_used_bytes(stat: dict, current: int) -> int:
    """current(= page cache 포함) 대신 회수 불가 사용량만 합산."""
    if not stat:
        return current  # stat 을 못 읽으면 종전대로 보수적 동작
    keys = _HARD_KEYS_V2 if "anon" in stat else _HARD_KEYS_V1
    hard = sum(stat.get(k, 0) for k in keys)
    if hard <= 0:
        return current
    return min(hard, current) if current > 0 else hard


def cgroup_ram_free_gb() -> float:
    """cgroup 여유 RAM(GB) = limit − '회수 불가' 사용량 (orchestrator 와 동일 로직).

    limit − memory.current 를 쓰면 안 된다: current 는 page cache(file) 를 포함해서,
    이 추출기가 frame_cache 로 쓴 JPEG 만으로도 수백 G 가 쌓인다
    (2026-07-24 20:23 관측: anon 4.2G / file 227.7G / current 232.9G).
    그러면 추출기가 자기 write-back 캐시로 자기 RAM 게이트를 막아 free=0G 로 영구
    대기하고(19:57~20:23 교착), 커널은 reclaim 만 돌며 동시 학습(sft_r0)을 스톨시켰다.
    page cache 는 회수 가능하므로 게이트에서 제외한다.
    """
    for mx, st, cur in _CG_PATHS:
        try:
            lim = Path(mx).read_text().strip()
            if lim == "max":
                break
            limit = int(lim)
            if limit <= 0 or limit >= (1 << 62):
                break
            try:
                used = int(Path(cur).read_text().strip())
            except Exception:
                used = 0
            hard = _hard_used_bytes(_read_mem_stat(st), used)
            free = max(0.0, (limit - hard) / 1e9)
            # 호스트가 실제로 더 빡빡하면 그쪽이 진짜 상한.
            try:
                import psutil
                free = min(free, psutil.virtual_memory().available / 1e9)
            except Exception:
                pass
            return free
        except Exception:
            continue
    return 1e9


def wait_for_ram():
    """여유 RAM 이 floor 이상 될 때까지 대기 (theta_ce/orchestrator 우선)."""
    waited = 0
    while cgroup_ram_free_gb() < RAM_FLOOR_GB:
        if waited % 300 == 0:
            log(f"RAM 대기: free {cgroup_ram_free_gb():.0f}G < floor {RAM_FLOOR_GB:.0f}G — 일시정지")
        time.sleep(30)
        waited += 30


def read_jsonl(p: Path):
    out = []
    if not Path(p).is_file():
        return out
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def build_pool(which: str) -> list[dict]:
    data = RUN / "data"
    subset = None
    sp = data / "train_subset.json"
    if sp.is_file():
        j = json.loads(sp.read_text())
        subset = set(j["sample_ids"]) if isinstance(j, dict) and "sample_ids" in j else None
    pool = []
    if which in ("train", "both"):
        for r in read_jsonl(data / "context_train.jsonl"):
            if r.get("gt_rank", 99) <= 10 and (subset is None or r["sample_id"] in subset):
                pool.append(r)
    if which in ("val", "both"):
        for r in read_jsonl(data / "context_val.jsonl"):
            if r.get("gt_rank", 99) <= 10:
                pool.append(r)
    # 같은 sample_id 중복 제거
    seen, uniq = set(), []
    for r in pool:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        uniq.append(r)
    return uniq


def frame_indices(fps: float, n_total: int, start: float, end: float) -> list[int]:
    """vlm.extract_frames 와 동일 — 캐시가 on-the-fly 와 정확히 일치해야 함."""
    t0, t1 = max(0.0, start), max(start + 1e-3, end)
    return [min(n_total - 1, max(0, round((t0 + (t1 - t0) * i / max(1, N_FRAMES - 1)) * fps)))
            for i in range(N_FRAMES)]


def already_ok(done_map: dict, sid: str, sdir: Path) -> bool:
    if done_map.get(sid) == "ok" and sdir.is_dir():
        return all((sdir / f"f{i}.jpg").is_file() for i in range(N_FRAMES))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="both", choices=["train", "val", "both"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import yaml
    from PIL import Image
    import decord
    decord.bridge.set_bridge("native")  # torch/CUDA 무접촉

    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    CACHE.mkdir(parents=True, exist_ok=True)

    done_map = {r["sample_id"]: r.get("reason", "ok") if r.get("ok") else "bad"
                for r in read_jsonl(MANIFEST)}
    pool = build_pool(args.pool)
    if args.limit:
        pool = pool[: args.limit]
    # 영상별 정렬 → 같은 VideoReader 연속 사용 (리더 재생성 최소화)
    pool.sort(key=lambda r: (r["video_uid"], r["obs_start_sec"]))
    log(f"pool={len(pool)} (이미 ok={sum(v=='ok' for v in done_map.values())}) floor={RAM_FLOOR_GB:.0f}G")

    n_ok = n_bad = n_skip = 0
    cur_uid = None
    vr = None
    for i, r in enumerate(pool):
        sid, uid = r["sample_id"], r["video_uid"]
        sdir = CACHE / uid / sid
        if already_ok(done_map, sid, sdir):
            n_skip += 1
            continue
        wait_for_ram()
        try:
            if uid != cur_uid:
                del vr
                gc.collect()
                vr = decord.VideoReader(str(video_root / f"{uid}.mp4"), num_threads=2)
                cur_uid = uid
            fps = vr.get_avg_fps()
            n_total = len(vr)
            idxs = frame_indices(fps, n_total, r["obs_start_sec"], r["obs_end_sec"])
            batch = vr.get_batch(idxs).asnumpy()  # (N,H,W,3) uint8
            imgs, sizes = [], set()
            for k in range(batch.shape[0]):
                im = Image.fromarray(batch[k])
                w, h = im.size
                s = SHORT / min(w, h)
                if s < 1.0:
                    im = im.resize((round(w * s), round(h * s)))
                imgs.append(im)
                sizes.add(im.size)
            # ── 검증: 개수(짝수) · 크기 동일 ──
            if len(imgs) != N_FRAMES or N_FRAMES % 2 != 0 or len(sizes) != 1:
                reason = f"bad_frames n={len(imgs)} sizes={len(sizes)}"
                with open(MANIFEST, "a") as f:
                    f.write(json.dumps({"sample_id": sid, "video_uid": uid,
                                        "ok": False, "reason": reason}) + "\n")
                n_bad += 1
                del batch, imgs
                continue
            sdir.mkdir(parents=True, exist_ok=True)
            for k, im in enumerate(imgs):
                im.save(sdir / f"f{k}.jpg", quality=95)
            with open(MANIFEST, "a") as f:
                f.write(json.dumps({"sample_id": sid, "video_uid": uid, "ok": True,
                                    "reason": "ok", "dir": str(sdir), "n": N_FRAMES}) + "\n")
            n_ok += 1
            del batch, imgs
        except Exception as e:
            with open(MANIFEST, "a") as f:
                f.write(json.dumps({"sample_id": sid, "video_uid": uid, "ok": False,
                                    "reason": f"exc:{str(e)[:120]}"}) + "\n")
            n_bad += 1
        finally:
            gc.collect()
        if (i + 1) % 200 == 0:
            log(f"{i+1}/{len(pool)} ok={n_ok} bad={n_bad} skip={n_skip} free={cgroup_ram_free_gb():.0f}G")

    del vr
    gc.collect()
    (CACHE / "EXTRACT_DONE").write_text(json.dumps(
        {"ts": time.time(), "ok": n_ok, "bad": n_bad, "skip": n_skip, "pool": len(pool)}))
    log(f"DONE ok={n_ok} bad={n_bad} skip={n_skip} pool={len(pool)}")


if __name__ == "__main__":
    main()
