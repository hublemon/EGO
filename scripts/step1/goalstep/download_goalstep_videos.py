#!/usr/bin/env python3
"""GoalStep 영상 병렬 다운로드 (540ss 우선 + full_scale 폴백, 재개 안전).

입력: goalstep_download_plan.json — [{video_uid, s3_path, source}, ...]
  (video_540ss/full_scale v2_1 매니페스트에서 파생. 540ss 우선, 없으면 full_scale.)

동작:
  1) HEAD 패스 — 전 객체 ContentLength 수집 → plan 에 size_bytes 기록 (정확한 총량/ETA 분모)
  2) 다운로드 — .part 임시파일 → 완료 시 rename. 최종 파일이 존재하고 크기가 일치하면 스킵
     → 중단/재실행 안전. AWS 키는 ~/.aws/credentials 만 사용 (인자/환경변수 금지).
  3) 진행상태 — <out>/.download_status.json 을 파일 단위 완료 시마다 갱신 (대시보드가 읽음)
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config


def parse_s3(uri: str):
    rest = uri[len("s3://"):]
    bucket, key = rest.split("/", 1)
    return bucket, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text())
    s3 = boto3.client("s3", config=Config(max_pool_connections=args.workers * 4,
                                          retries={"max_attempts": 10, "mode": "adaptive"}))

    # 1) HEAD 패스 — 크기 확보 (plan 에 캐싱, 재실행 시 스킵)
    if any("size_bytes" not in it for it in plan):
        def head(it):
            b, k = parse_s3(it["s3_path"])
            it["size_bytes"] = s3.head_object(Bucket=b, Key=k)["ContentLength"]
            return it
        with ThreadPoolExecutor(max_workers=32) as ex:
            done = 0
            for _ in as_completed([ex.submit(head, it) for it in plan]):
                done += 1
                if done % 100 == 0:
                    print(f"[head] {done}/{len(plan)}", flush=True)
        Path(args.plan).write_text(json.dumps(plan, indent=1))
    total_bytes = sum(it["size_bytes"] for it in plan)
    print(f"[plan] {len(plan)} videos, {total_bytes/2**30:.1f} GiB total", flush=True)
    if args.dry_run:
        return

    status_path = out / ".download_status.json"
    lock = threading.Lock()
    state = {"n_total": len(plan), "total_bytes": total_bytes,
             "n_done": 0, "done_bytes": 0, "failed": [],
             "started_at": time.time(), "updated_at": time.time(), "finished": False}
    # 재개: 이미 완료된 파일 반영. resumed_bytes 는 속도 계산의 기준점(대시보드용).
    todo = []
    for it in plan:
        dst = out / f"{it['video_uid']}.mp4"
        if dst.exists() and dst.stat().st_size == it["size_bytes"]:
            state["n_done"] += 1; state["done_bytes"] += it["size_bytes"]
        else:
            todo.append(it)
    state["resumed_bytes"] = state["done_bytes"]

    def write_status():
        state["updated_at"] = time.time()
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(status_path)

    write_status()
    print(f"[resume] already done {state['n_done']}/{len(plan)}, remaining {len(todo)}", flush=True)

    def fetch(it):
        b, k = parse_s3(it["s3_path"])
        dst = out / f"{it['video_uid']}.mp4"
        part = out / f"{it['video_uid']}.mp4.part"
        s3.download_file(b, k, str(part))
        if part.stat().st_size != it["size_bytes"]:
            raise IOError(f"size mismatch {it['video_uid']}")
        os.replace(part, dst)
        return it

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, it): it for it in todo}
        for fu in as_completed(futs):
            it = futs[fu]
            with lock:
                try:
                    fu.result()
                    state["n_done"] += 1; state["done_bytes"] += it["size_bytes"]
                    gb = state["done_bytes"] / 2**30
                    print(f"[ok] {it['video_uid']} ({it['source']}) "
                          f"{state['n_done']}/{state['n_total']} {gb:.1f} GiB", flush=True)
                except Exception as e:  # noqa: BLE001
                    state["failed"].append({"video_uid": it["video_uid"], "error": str(e)})
                    print(f"[FAIL] {it['video_uid']}: {e}", flush=True)
                write_status()
    state["finished"] = True
    write_status()
    print(f"[DONE] {state['n_done']}/{state['n_total']} ok, {len(state['failed'])} failed, "
          f"{state['done_bytes']/2**30:.1f} GiB", flush=True)
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
