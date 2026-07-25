"""공통 런타임 — 진행 상태/처리율/ETA 기록. 대시보드(tools/retro3_dashboard.py)가 읽는다.

모든 stage 모듈은 StatusWriter 하나를 열고 update()를 주기 호출한다.
파일 스키마 (runs/retro3/status/<stage>.json):
{
  "stage": str, "state": "running"|"done"|"failed",
  "done": int, "total": int, "pct": float,
  "started_at": epoch, "updated_at": epoch,
  "rate_per_s": float, "eta_sec": float|null, "elapsed_sec": float,
  "metrics": {...},   # stage별 실측값 (coverage, gate율, loss, margin 등)
  "message": str
}
원자적 쓰기(tmp → rename) — 대시보드가 절대 깨진 json을 읽지 않도록.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

RUNS_ROOT = Path(os.environ.get("RETRO3_RUNS", "runs/retro3"))


def runs_root() -> Path:
    return RUNS_ROOT


def status_dir() -> Path:
    d = RUNS_ROOT / "status"
    d.mkdir(parents=True, exist_ok=True)
    return d


def marker_dir() -> Path:
    d = RUNS_ROOT / "markers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_marker(name: str, payload: dict | None = None) -> None:
    p = marker_dir() / name
    p.write_text(json.dumps({"ts": time.time(), **(payload or {})}, ensure_ascii=False))


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(path)


class StatusWriter:
    """stage 진행 상태 기록기. update()는 매 샘플 호출해도 min_interval로 스로틀."""

    def __init__(self, stage: str, total: int | None = None, min_interval: float = 2.0):
        self.stage = stage
        self.total = total
        self.done = 0
        self.t0 = time.time()
        self._last_write = 0.0
        self.min_interval = min_interval
        self.metrics: dict = {}
        self.path = status_dir() / f"{stage}.json"
        self._flush("running", force=True)

    def update(self, done: int | None = None, inc: int = 0, metrics: dict | None = None,
               message: str = "", force: bool = False) -> None:
        if done is not None:
            self.done = done
        else:
            self.done += inc
        if metrics:
            self.metrics.update(metrics)
        now = time.time()
        if force or (now - self._last_write) >= self.min_interval:
            self._flush("running", message=message)

    def finish(self, metrics: dict | None = None, message: str = "") -> None:
        if metrics:
            self.metrics.update(metrics)
        self._flush("done", message=message, force=True)

    def fail(self, message: str) -> None:
        self._flush("failed", message=message, force=True)

    def _flush(self, state: str, message: str = "", force: bool = False) -> None:
        now = time.time()
        elapsed = now - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = None
        if self.total and rate > 0 and state == "running":
            eta = max(0.0, (self.total - self.done) / rate)
        _atomic_write(self.path, {
            "stage": self.stage, "state": state,
            "done": self.done, "total": self.total,
            "pct": (100.0 * self.done / self.total) if self.total else None,
            "started_at": self.t0, "updated_at": now,
            "rate_per_s": round(rate, 4), "eta_sec": eta, "elapsed_sec": round(elapsed, 1),
            "metrics": self.metrics, "message": message,
        })
        self._last_write = now


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
