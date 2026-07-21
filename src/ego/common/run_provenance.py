"""run_provenance.py — 실행 출처를 산출물로 남기는 최소 헬퍼 (핸드오프 §7 개선 0).

배경(2026-07-20 retro 핸드오프 §6 RC6): 그날 5개 실행의 하이퍼파라미터가 **셸 스크립트의
CLI 플래그로만** 존재했다. `.pid`·tmux·wandb·설정 YAML 이 전부 없어서, 재현 가능한 것은
체인 로그뿐이었다. 그래서 어떤 러너든 출력 디렉터리에 `run_config.json` 한 장을 남긴다:

  전체 argv · git SHA(+dirty) · 타임스탬프(UTC/로컬) · 입력 데이터 경로와 크기/mtime/해시

설계 원칙: **절대 실행을 깨뜨리지 않는다.** 모든 실패는 경고 한 줄로 삼키고 None 을 반환한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 해시를 계산할 최대 파일 크기 (그 이상은 sha256=None — 크기/mtime 만으로 식별)
DEFAULT_HASH_MAX_BYTES = 256 * 1024 * 1024
# run_config 에 남길 환경변수 (전체 environ 은 토큰이 섞일 수 있어 화이트리스트만)
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE", "PYTHONPATH",
            "EGO_ROOT", "WORLD_SIZE", "LOCAL_RANK")


def _git(repo: Path, *cmd: str):
    try:
        # `-c safe.directory=*` : 마운트 이동으로 소유자가 달라진 체크아웃(dubious ownership)
        #   에서도 조회는 성공하도록 — 기록 실패가 실행 중단으로 이어져선 안 된다.
        r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), *cmd],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def git_info(repo: Path | str | None = None) -> dict:
    """git SHA·브랜치·dirty 여부. git 이 없거나 repo 가 아니어도 예외를 내지 않는다."""
    repo = Path(repo) if repo else Path(__file__).resolve().parents[3]
    sha = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    return {"repo": str(repo), "sha": sha,
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": (bool(status) if status is not None else None),
            "dirty_files": (len(status.splitlines()) if status else 0)}


def file_fingerprint(path: str | Path, hash_max_bytes: int = DEFAULT_HASH_MAX_BYTES) -> dict:
    """파일/디렉터리 지문: 크기·mtime·sha256(작은 파일만). 없으면 exists=False."""
    p = Path(path)
    info: dict = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return info
    try:
        st = p.stat()
        info["is_dir"] = p.is_dir()
        info["mtime"] = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(
            timespec="seconds")
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            info["num_files"] = len(files)
            info["bytes"] = sum(f.stat().st_size for f in files)
            info["sha256"] = None
            return info
        info["bytes"] = st.st_size
        if st.st_size <= hash_max_bytes:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            info["sha256"] = h.hexdigest()
        else:
            info["sha256"] = None   # 너무 큼 — 크기/mtime 으로만 식별
        if p.suffix == ".jsonl" and st.st_size <= hash_max_bytes:
            with p.open("rb") as f:
                info["num_lines"] = sum(1 for _ in f)
    except Exception as e:   # 권한/경합 — 지문 실패가 실행을 막아선 안 된다
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _as_dict(args) -> dict:
    if args is None:
        return {}
    if isinstance(args, dict):
        d = dict(args)
    else:
        d = dict(vars(args))
    return {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict))
                else str(v)) for k, v in d.items()}


def _auto_data_paths(args_dict: dict) -> list[str]:
    """args 값 중 '실제로 존재하는 경로'를 입력 데이터 후보로 자동 수집."""
    found = []
    for k, v in args_dict.items():
        if not isinstance(v, str) or not v or v.startswith("-"):
            continue
        if len(v) > 4096 or "\n" in v:
            continue
        try:
            if Path(v).exists():
                found.append(v)
        except OSError:
            continue
    return found


def write_run_config(output_dir: str | Path,
                     args=None,
                     data_paths=None,
                     extra: dict | None = None,
                     filename: str = "run_config.json",
                     repo: str | Path | None = None,
                     hash_max_bytes: int = DEFAULT_HASH_MAX_BYTES):
    """`output_dir/run_config.json` 을 쓰고 경로를 반환한다 (실패 시 None).

    args        : argparse.Namespace 또는 dict — 전량 기록
    data_paths  : 입력 데이터 경로 목록. None 이면 args 값 중 존재하는 경로를 자동 수집
    extra       : 러너별 추가 필드(예: n_pairs, 유효 배치 크기)
    """
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        a = _as_dict(args)
        paths = list(data_paths) if data_paths is not None else _auto_data_paths(a)
        now = datetime.now(timezone.utc)
        cfg = {
            "schema": "ego.run_config/1",
            "script": (sys.argv[0] if sys.argv else None),
            "argv": list(sys.argv),
            "cmdline": " ".join(sys.argv),
            "args": a,
            "git": git_info(repo),
            "time_utc": now.isoformat(timespec="seconds"),
            "time_local": datetime.now().astimezone().isoformat(timespec="seconds"),
            # 서버는 UTC, 보고는 KST — 두 시각을 같이 남겨 로그 대조 시 환산 오류를 막는다
            "time_kst": now.astimezone(timezone(timedelta(hours=9))).isoformat(
                timespec="seconds"),
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "env": {k: os.environ.get(k) for k in ENV_KEYS if os.environ.get(k) is not None},
            "packages": _pkg_versions(),
            "inputs": [file_fingerprint(p, hash_max_bytes) for p in paths],
        }
        if extra:
            cfg["extra"] = extra
        dst = out / filename
        dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[provenance] → {dst}", flush=True)
        return dst
    except Exception as e:   # 기록 실패가 학습/평가를 죽이면 안 된다
        print(f"[provenance] 기록 실패(무시): {type(e).__name__}: {e}", flush=True)
        return None


def _pkg_versions() -> dict:
    """이미 import 된 핵심 패키지의 버전만 기록 (없으면 생략 — import 부작용 금지)."""
    out = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        mod = sys.modules.get(name)
        v = getattr(mod, "__version__", None) if mod is not None else None
        if v:
            out[name] = str(v)
    if "torch" in out:
        try:
            import torch   # 이미 로드된 경우에만 도달
            out["cuda"] = torch.version.cuda
            out["gpu"] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else None)
        except Exception:
            pass
    return out
