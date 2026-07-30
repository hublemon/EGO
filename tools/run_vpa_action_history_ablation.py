#!/usr/bin/env python3
"""Resume-safe VPA action-history ablation queue.

Runs exactly these six methods:
  ours_wm1st, ours_full, qwen_backbone, frontier,
  wm_top1_repeat, wm_topk_rank

Phase order is fixed:
  1) T=3, action history removed
  2) T=4, action history included
  3) T=4, action history removed

Every model arm always receives the same 4-second / 8-frame observation.  Only
the completed-action text is ablated.  Frontier runs in parallel with the
single-GPU local queue inside each phase; phases themselves never overlap.
All runners append records and are safe to resume after interruption.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/vpa_v2"
OUT = RUN / "action_history_ablation"
STATE_PATH = OUT / "pipeline_state.json"
FILE_LOCK_PATH = OUT / "pipeline.lock"
LOG_DIR = OUT / "logs"
ADAPTER = Path(os.environ.get(
    "VPA_SFT_ADAPTER",
    REPO.parent / "EGO_jihun3/outputs/step2_retrospection/cesft_v2/sft_r15/adapter",
))
PYTHON = sys.executable

METHODS = (
    "ours_wm1st",
    "ours_full",
    "qwen_backbone",
    "frontier",
    "wm_top1_repeat",
    "wm_topk_rank",
)
LOCAL_ARMS = (
    ("ours_wm1st", ("--adapter", str(ADAPTER), "--candidates", "wm10_first")),
    ("ours_full", ("--adapter", str(ADAPTER), "--candidates", "vocab")),
    ("qwen_backbone", ("--candidates", "vocab")),
)
PHASES = (
    {"key": "T3_nohist", "horizon": 3, "history": "none", "total": 915},
    {"key": "T4_full", "horizon": 4, "history": "full", "total": 504},
    {"key": "T4_nohist", "horizon": 4, "history": "none", "total": 504},
)

LOCK = threading.RLock()
FILE_LOCK_HANDLE = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_singleton_lock() -> None:
    """Prevent duplicate GPU/API queues from writing the same resume files."""
    global FILE_LOCK_HANDLE
    FILE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = FILE_LOCK_PATH.open("a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fh.close()
        raise RuntimeError(
            f"another VPA ablation orchestrator holds {FILE_LOCK_PATH}"
        ) from exc
    os.set_inheritable(fh.fileno(), False)
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    FILE_LOCK_HANDLE = fh


def initial_state() -> dict:
    old = load_json(STATE_PATH, {}) or {}
    state = {
        "schema_version": 1,
        "title": "VPA action-history ablation",
        "pipeline_state": old.get("pipeline_state", "queued"),
        "pid": os.getpid(),
        "started_at": old.get("started_at"),
        "finished_at": old.get("finished_at"),
        "updated_at": now(),
        "methods": list(METHODS),
        "phase_order": [p["key"] for p in PHASES],
        "schedule": "gpu_parallel_first_then_frontier",
        "queue_stage": old.get("queue_stage", "gpu"),
        "frame_contract": {
            "with_frames": True,
            "blind_arms": False,
            "window_sec": 4,
            "safety_gap_sec": 1,
            "n_frames": 8,
            "short_side": 336,
        },
        "provenance": old.get("provenance", {}),
        "notes": [
            "Only completed action-history text is ablated; video frames are unchanged.",
            "WM-only baselines are recomputed per phase but are mathematically history-invariant.",
            "The three local GPU arms run concurrently per condition; all T3/T4 GPU work "
            "finishes before any remaining frontier API calls.",
            "A phase is complete only at 100% prediction coverage.",
        ],
        "phases": old.get("phases", {}),
        "events": old.get("events", [])[-100:],
        "error": None,
    }
    for p in PHASES:
        phase = state["phases"].setdefault(p["key"], {})
        phase.update({
            "key": p["key"],
            "horizon": p["horizon"],
            "history": p["history"],
            "total": p["total"],
        })
        phase.setdefault("state", "queued")
        phase.setdefault("started_at", None)
        phase.setdefault("finished_at", None)
        arms = phase.setdefault("arms", {})
        for method in METHODS:
            arms.setdefault(method, {
                "state": "queued",
                "started_at": None,
                "finished_at": None,
                "initial_done": 0,
                "pid": None,
                "attempt": 0,
                "returncode": None,
                "error": None,
            })
    return state


STATE = initial_state()


def save_state() -> None:
    with LOCK:
        STATE["updated_at"] = now()
        atomic_json(STATE_PATH, STATE)


def event(message: str) -> None:
    with LOCK:
        STATE["events"].append({"at": now(), "message": message})
        STATE["events"] = STATE["events"][-100:]
        save_state()


def set_pipeline(value: str, **extra) -> None:
    with LOCK:
        STATE["pipeline_state"] = value
        STATE.update(extra)
        save_state()


def set_phase(key: str, value: str, **extra) -> None:
    with LOCK:
        STATE["phases"][key]["state"] = value
        STATE["phases"][key].update(extra)
        save_state()


def set_arm(phase_key: str, arm: str, value: str, **extra) -> None:
    with LOCK:
        rec = STATE["phases"][phase_key]["arms"][arm]
        rec["state"] = value
        rec.update(extra)
        save_state()


def read_records(prefix: Path) -> tuple[dict[str, dict], int]:
    records = {}
    failures = 0
    path = prefix.with_suffix(".records.jsonl")
    if not path.is_file():
        return records, failures
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                records[row["sample_id"]] = row
            else:
                failures += 1
    return records, failures


def done_count(prefix: Path) -> int:
    return len(read_records(prefix)[0])


def expected_ids(phase: dict) -> set[str]:
    samples = load_json(RUN / f"vpa_v2_T{phase['horizon']}.json", [])
    return {s["sample_id"] for s in samples}


def expected_arm_contract(phase: dict, arm: str) -> dict:
    contract = {
        "history": phase["history"],
        "with_frames": True,
        "horizon": phase["horizon"],
    }
    if arm == "frontier":
        contract["model"] = os.environ.get("FRONTIER_MODEL")
    else:
        contract["model"] = "Qwen/Qwen3-VL-8B-Instruct"
        contract["candidate_mode"] = (
            "wm10_first" if arm == "ours_wm1st" else "vocab"
        )
        contract["adapter"] = (
            str(ADAPTER) if arm in {"ours_wm1st", "ours_full"} else None
        )
    return contract


def validate_partial_records(prefix: Path, phase: dict, arm: str) -> int:
    """Validate every resumable success before the runner is allowed to skip it."""
    records, _ = read_records(prefix)
    expected = expected_ids(phase)
    if not set(records).issubset(expected):
        alien = sorted(set(records) - expected)[:5]
        raise RuntimeError(f"{prefix.name}: alien sample IDs in resume file: {alien}")
    contract = expected_arm_contract(phase, arm)
    bad = []
    for sid, row in records.items():
        if len(row.get("pred") or []) != phase["horizon"]:
            bad.append((sid, "prediction_length"))
            continue
        for key, value in contract.items():
            if row.get(key) != value:
                bad.append((sid, key))
                break
    if bad:
        raise RuntimeError(
            f"{prefix.name}: {len(bad)} resumable records violate arm contract; "
            f"examples={bad[:5]}"
        )
    return len(records)


def run_command(cmd: list[str], log_path: Path, phase_key: str | None = None,
                arm: str | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] $ {shlex.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if phase_key and arm:
            set_arm(phase_key, arm, STATE["phases"][phase_key]["arms"][arm]["state"],
                    pid=proc.pid)
        rc = proc.wait()
        if phase_key and arm:
            set_arm(phase_key, arm, STATE["phases"][phase_key]["arms"][arm]["state"],
                    pid=None, returncode=rc)
        log.write(f"[{now()}] returncode={rc}\n")
        log.flush()
        return rc


def prompt_contract_check() -> None:
    from ego.step3_results.vpa.v2.common import build_prompt

    sample = {
        "goal_text": "make tea",
        "observed_actions": ["take cup", "pour water"],
        "wm_candidates": ["take cup", "pour water"],
    }
    vocab = ["pour water", "take cup"]
    expected = {
        "vocab": (
            "24111901f197b29c02b1556961e44936a9f616540ab6aa45f40af93a73ff4ef5",
            "ab3ab5cdf2e617c1868897d91bbdd60304d0a5a6c7aa9ab4fffee70f59d97c9b",
        ),
        "wm10_first": (
            "24111901f197b29c02b1556961e44936a9f616540ab6aa45f40af93a73ff4ef5",
            "5ab2ed46b494299484f3614f86ffc1764057557cc02123229af37b8a9191bb6c",
        ),
    }
    for mode, wanted in expected.items():
        full = build_prompt(
            sample, vocab, 3, with_frames=True,
            candidate_mode=mode, history="full",
        )
        got = tuple(hashlib.sha256(x.encode()).hexdigest() for x in full)
        if got != wanted:
            raise RuntimeError(f"history=full prompt drift for {mode}: {got}")
        system, user = build_prompt(
            sample, vocab, 3, with_frames=True,
            candidate_mode=mode, history="none",
        )
        if "8 frames sampled" not in system or "no video" in system:
            raise RuntimeError("no-history prompt lost the video-frame contract")
        if "already COMPLETED" in system:
            raise RuntimeError("no-history system prompt still describes completed actions")
        block = user.split(
            "ACTIONS ALREADY COMPLETED (in order):\n", 1
        )[1].split("\n\nCANDIDATE ACTION LABELS", 1)[0]
        if block != "  (not provided)":
            raise RuntimeError(f"unexpected no-history block: {block!r}")
        if "  1. take cup" in user or "  2. pour water" in user:
            raise RuntimeError("completed action text leaked into no-history prompt")


def prepare_subsets() -> None:
    from ego.step3_results.vpa.v2 import frames as F

    prompt_contract_check()
    if not ADAPTER.is_dir():
        raise FileNotFoundError(f"missing adapter: {ADAPTER}")
    adapter_config_path = ADAPTER / "adapter_config.json"
    adapter_weights_path = ADAPTER / "adapter_model.safetensors"
    adapter_config = load_json(adapter_config_path, {})
    base_model = adapter_config.get("base_model_name_or_path")
    if base_model != "Qwen/Qwen3-VL-8B-Instruct":
        raise RuntimeError(
            f"unexpected adapter base model: {base_model!r}"
        )
    if not adapter_weights_path.is_file():
        raise FileNotFoundError(f"missing adapter weights: {adapter_weights_path}")
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip())
    except subprocess.SubprocessError:
        git_head, git_dirty = None, None
    with LOCK:
        STATE["provenance"] = {
            "git_head": git_head,
            "git_dirty": git_dirty,
            "python": PYTHON,
            "base_model": base_model,
            "adapter_path": str(ADAPTER.resolve()),
            "adapter_config_sha256": sha256_file(adapter_config_path),
            "adapter_weights_sha256": sha256_file(adapter_weights_path),
            "frontier_model": os.environ.get("FRONTIER_MODEL"),
            "frontier_endpoint": os.environ.get("FRONTIER_BASE_URL"),
        }
        save_state()
    for p in PHASES:
        gt = RUN / f"vpa_v2_T{p['horizon']}.json"
        samples = load_json(gt)
        if not isinstance(samples, list) or len(samples) != p["total"]:
            raise RuntimeError(
                f"{gt}: expected {p['total']} samples, got "
                f"{len(samples) if isinstance(samples, list) else 'invalid'}"
            )
        if len({s["sample_id"] for s in samples}) != p["total"]:
            raise RuntimeError(f"{gt}: duplicate sample IDs")
        cache = RUN / F.cache_dirname()
        eligible = [
            s["sample_id"]
            for s in samples
            if all(x.is_file() for x in F.frame_paths(cache, s))
        ]
        if len(eligible) != p["total"]:
            raise RuntimeError(
                f"T={p['horizon']}: frames {len(eligible)}/{p['total']} "
                "(pipeline will not run on a partial frame set)"
            )
        atomic_json(
            RUN / f"frames_subset_T{p['horizon']}.json",
            {
                "sample_ids": eligible,
                "n": len(eligible),
                "history": "condition-specific",
                "with_frames": True,
                "frame_contract": STATE["frame_contract"],
            },
        )


def local_prefix(phase: dict, arm: str) -> Path:
    return OUT / phase["key"] / "preds" / f"{arm}_T{phase['horizon']}"


def base_local_cmd(phase: dict, arm_args: tuple[str, ...], prefix: Path) -> list[str]:
    return [
        PYTHON, "-u", "-m", "ego.step3_results.vpa.v2.run_local_vlm",
        "--gt", str(RUN / f"vpa_v2_T{phase['horizon']}.json"),
        "--frames-dir", str(RUN),
        "--mode", "frames",
        "--history", phase["history"],
        "--out", str(prefix),
        *arm_args,
    ]


def verify_model_records(prefix: Path, phase: dict) -> None:
    arm = prefix.stem.rsplit(f"_T{phase['horizon']}", 1)[0]
    count = validate_partial_records(prefix, phase, arm)
    records, _ = read_records(prefix)
    if set(records) != expected_ids(phase):
        raise RuntimeError(
            f"{prefix.name}: exact-ID coverage failed "
            f"({count}/{phase['total']})"
        )
    preds = load_json(prefix.with_suffix(".json"), {})
    if set(preds) != expected_ids(phase):
        raise RuntimeError(
            f"{prefix.with_suffix('.json')}: {len(preds)}/{phase['total']}"
        )
    if any(len(preds[sid]) != phase["horizon"] for sid in preds):
        raise RuntimeError(f"{prefix.name}: final prediction length mismatch")


def run_local_arm(phase: dict, arm: str, arm_args: tuple[str, ...]) -> None:
    prefix = local_prefix(phase, arm)
    log = LOG_DIR / f"{phase['key']}_{arm}.log"
    initial = validate_partial_records(prefix, phase, arm)
    if initial == phase["total"]:
        verify_model_records(prefix, phase)
        set_arm(
            phase["key"], arm, "completed",
            initial_done=initial,
            finished_at=STATE["phases"][phase["key"]]["arms"][arm].get("finished_at") or now(),
        )
        return

    set_arm(
        phase["key"], arm, "running",
        started_at=now(), finished_at=None, initial_done=initial,
        error=None, attempt=0,
    )
    cmd = base_local_cmd(phase, arm_args, prefix)

    # Required parse/adapter/frame smoke. It writes into the real prefix, so the
    # subsequent full command resumes rather than discarding the three calls.
    if phase["key"] == "T3_nohist" and arm == "ours_wm1st" and initial == 0:
        set_arm(phase["key"], arm, "smoke", attempt=1)
        rc = run_command(cmd + ["--limit", "3"], log, phase["key"], arm)
        if rc != 0 or done_count(prefix) < 3:
            raise RuntimeError(
                f"{phase['key']}/{arm}: 3-sample smoke failed (rc={rc}, "
                f"done={done_count(prefix)})"
            )
        set_arm(phase["key"], arm, "running", initial_done=3)
        event("T3 no-history local 3-sample smoke passed with frames enabled")

    previous = done_count(prefix)
    for attempt in range(1, 4):
        set_arm(phase["key"], arm, "running", attempt=attempt)
        rc = run_command(cmd, log, phase["key"], arm)
        current = done_count(prefix)
        if rc == 0 and current == phase["total"]:
            verify_model_records(prefix, phase)
            set_arm(
                phase["key"], arm, "completed",
                finished_at=now(), pid=None, returncode=0,
            )
            event(f"{phase['key']} {arm} completed {current}/{phase['total']}")
            return
        if current <= previous:
            raise RuntimeError(
                f"{phase['key']}/{arm}: no progress on attempt {attempt} "
                f"(rc={rc}, {current}/{phase['total']})"
            )
        previous = current
    raise RuntimeError(
        f"{phase['key']}/{arm}: incomplete after retries "
        f"({done_count(prefix)}/{phase['total']})"
    )


def frontier_cmd(phase: dict, prefix: Path) -> list[str]:
    return [
        PYTHON, "-u", "-m", "ego.step3_results.vpa.v2.run_frontier",
        "--gt", str(RUN / f"vpa_v2_T{phase['horizon']}.json"),
        "--subset", str(RUN / f"frames_subset_T{phase['horizon']}.json"),
        "--frames-dir", str(RUN),
        "--history", phase["history"],
        "--out", str(prefix),
        "--max-calls", str(phase["total"]),
    ]


def run_frontier_phase(phase: dict, errors: list[str]) -> None:
    arm = "frontier"
    prefix = local_prefix(phase, arm)
    log = LOG_DIR / f"{phase['key']}_{arm}.log"
    try:
        for key in ("FRONTIER_API_KEY", "FRONTIER_BASE_URL", "FRONTIER_MODEL"):
            if not os.environ.get(key):
                raise RuntimeError(f"{key} is not set")
        initial = validate_partial_records(prefix, phase, arm)
        if initial == phase["total"]:
            verify_model_records(prefix, phase)
            set_arm(
                phase["key"], arm, "completed",
                initial_done=initial,
                finished_at=STATE["phases"][phase["key"]]["arms"][arm].get("finished_at") or now(),
            )
            return

        set_arm(
            phase["key"], arm, "running",
            started_at=now(), finished_at=None, initial_done=initial,
            error=None, attempt=0,
        )
        cmd = frontier_cmd(phase, prefix)
        if initial == 0:
            set_arm(phase["key"], arm, "smoke", attempt=1)
            rc = run_command(cmd + ["--probe"], log, phase["key"], arm)
            if rc != 0 or done_count(prefix) < 1:
                raise RuntimeError(
                    f"{phase['key']}/frontier vision probe failed "
                    f"(rc={rc}, done={done_count(prefix)})"
                )
            set_arm(phase["key"], arm, "running", initial_done=1)
            event(f"{phase['key']} frontier vision probe passed with 8 images")

        previous = done_count(prefix)
        for attempt in range(1, 4):
            set_arm(phase["key"], arm, "running", attempt=attempt)
            rc = run_command(cmd, log, phase["key"], arm)
            current = done_count(prefix)
            if rc == 0 and current == phase["total"]:
                verify_model_records(prefix, phase)
                status = load_json(
                    prefix.parent / f"{prefix.stem}.status.json", {}
                )
                if not status.get("complete"):
                    raise RuntimeError("frontier status did not mark complete=true")
                set_arm(
                    phase["key"], arm, "completed",
                    finished_at=now(), pid=None, returncode=0,
                )
                event(
                    f"{phase['key']} frontier completed "
                    f"{current}/{phase['total']}"
                )
                return
            if current <= previous:
                raise RuntimeError(
                    f"{phase['key']}/frontier: no progress on attempt {attempt} "
                    f"(rc={rc}, {current}/{phase['total']})"
                )
            previous = current
        raise RuntimeError(
            f"{phase['key']}/frontier incomplete after retries "
            f"({done_count(prefix)}/{phase['total']})"
        )
    except Exception as exc:  # surface thread failures to the main queue
        message = f"{type(exc).__name__}: {exc}"
        set_arm(phase["key"], arm, "failed", finished_at=now(), pid=None, error=message)
        errors.append(message)


def run_baselines(phase: dict) -> None:
    names = ("wm_top1_repeat", "wm_topk_rank")
    for name in names:
        set_arm(
            phase["key"], name, "running",
            started_at=now(), initial_done=0, error=None,
        )
    out_dir = OUT / phase["key"] / "metrics"
    cmd = [
        PYTHON, "-u", "-m", "ego.step3_results.vpa.v2.evaluate",
        "--gt", str(RUN / f"vpa_v2_T{phase['horizon']}.json"),
        "--subset", str(RUN / f"frames_subset_T{phase['horizon']}.json"),
        "--out-dir", str(out_dir),
        "--baselines",
        "--baseline-names", *names,
    ]
    rc = run_command(cmd, LOG_DIR / f"{phase['key']}_baselines.log")
    metrics_path = out_dir / (
        f"metrics_T{phase['horizon']}_frames_subset_T{phase['horizon']}.json"
    )
    metrics = load_json(metrics_path, {})
    for name in names:
        rec = metrics.get(name, {})
        if rc != 0 or rec.get("n") != phase["total"] or not rec.get("reportable"):
            raise RuntimeError(
                f"{phase['key']}/{name}: baseline coverage validation failed"
            )
        set_arm(
            phase["key"], name, "completed",
            finished_at=now(), initial_done=0, returncode=0,
        )
    event(
        f"{phase['key']} WM-only baselines completed "
        "(history-invariant, no other baselines generated)"
    )


def mark_frontier_deferred(phase: dict) -> None:
    prefix = local_prefix(phase, "frontier")
    done = validate_partial_records(prefix, phase, "frontier")
    if done >= phase["total"]:
        verify_model_records(prefix, phase)
        set_arm(
            phase["key"], "frontier", "completed",
            finished_at=STATE["phases"][phase["key"]]["arms"]["frontier"].get("finished_at") or now(),
            pid=None,
        )
        return
    row = STATE["phases"][phase["key"]]["arms"]["frontier"]
    observed = row.get("observed_sec_per_sample")
    started = row.get("started_at")
    processed = max(0, done - int(row.get("initial_done") or 0))
    if started and processed:
        try:
            elapsed = (
                datetime.now(timezone.utc) - datetime.fromisoformat(started)
            ).total_seconds()
            observed = min(max(elapsed / processed, 0.15), 120.0)
        except ValueError:
            pass
    set_arm(
        phase["key"], "frontier", "deferred",
        pid=None, observed_sec_per_sample=observed,
    )


def score_phase(phase: dict) -> None:
    out_dir = OUT / phase["key"] / "metrics"
    subset = RUN / f"frames_subset_T{phase['horizon']}.json"
    gt = RUN / f"vpa_v2_T{phase['horizon']}.json"
    log = LOG_DIR / f"{phase['key']}_evaluate.log"
    for arm in ("ours_wm1st", "ours_full", "qwen_backbone", "frontier"):
        cmd = [
            PYTHON, "-u", "-m", "ego.step3_results.vpa.v2.evaluate",
            "--gt", str(gt),
            "--subset", str(subset),
            "--pred", str(local_prefix(phase, arm).with_suffix(".json")),
            "--run-name", arm,
            "--out-dir", str(out_dir),
        ]
        if run_command(cmd, log) != 0:
            raise RuntimeError(f"{phase['key']}: evaluate failed for {arm}")

    comparisons = (
        ("ours_full", "qwen_backbone"),
        ("ours_full", "frontier"),
        ("ours_wm1st", "ours_full"),
    )
    paired_results = {}
    for a, b in comparisons:
        cmd = [
            PYTHON, "-u", "-m", "ego.step3_results.vpa.v2.paired",
            "--gt", str(gt),
            "--subset", str(subset),
            "--a", str(local_prefix(phase, a).with_suffix(".json")),
            "--a-name", a,
            "--b", str(local_prefix(phase, b).with_suffix(".json")),
            "--b-name", b,
            "--out-dir", str(out_dir),
        ]
        if run_command(cmd, log) != 0:
            raise RuntimeError(f"{phase['key']}: paired failed for {a} vs {b}")
        paired_path = out_dir / (
            f"paired_{a}_vs_{b}_T{phase['horizon']}.json"
        )
        paired_results[f"{a}_vs_{b}"] = load_json(paired_path, {})

    metrics_path = out_dir / (
        f"metrics_T{phase['horizon']}_frames_subset_T{phase['horizon']}.json"
    )
    metrics = load_json(metrics_path, {})
    if set(METHODS) - set(metrics):
        raise RuntimeError(
            f"{phase['key']}: final metrics missing "
            f"{sorted(set(METHODS) - set(metrics))}"
        )
    if any(
        metrics[name].get("n") != phase["total"]
        or not metrics[name].get("reportable")
        for name in METHODS
    ):
        raise RuntimeError(f"{phase['key']}: final metrics are not 100% reportable")
    # Do not retain stale arms if this directory is ever reused.
    metrics = {name: metrics[name] for name in METHODS}
    atomic_json(metrics_path, metrics)
    atomic_json(
        OUT / phase["key"] / "summary.json",
        {
            "phase": phase["key"],
            "horizon": phase["horizon"],
            "history": phase["history"],
            "with_frames": True,
            "methods": list(METHODS),
            "qualified_methods": [
                f"{name}_{'nohist' if phase['history'] == 'none' else 'fullhist'}"
                for name in METHODS
            ],
            "metrics": metrics,
            "paired": paired_results,
            "provenance": STATE.get("provenance", {}),
            "completed_at": now(),
        },
    )


def summary_is_valid(phase: dict) -> bool:
    summary = load_json(OUT / phase["key"] / "summary.json", {})
    if not isinstance(summary, dict):
        return False
    if (
        summary.get("phase") != phase["key"]
        or summary.get("horizon") != phase["horizon"]
        or summary.get("history") != phase["history"]
        or summary.get("with_frames") is not True
        or summary.get("methods") != list(METHODS)
    ):
        return False
    metrics = summary.get("metrics") or {}
    if set(metrics) != set(METHODS):
        return False
    if any(
        metrics[name].get("n") != phase["total"]
        or not metrics[name].get("reportable")
        for name in METHODS
    ):
        return False
    try:
        for arm in ("ours_wm1st", "ours_full", "qwen_backbone", "frontier"):
            verify_model_records(local_prefix(phase, arm), phase)
    except Exception:
        return False
    return True


def run_gpu_phase(phase: dict) -> None:
    """Finish baselines and three concurrent local GPU arms; never call frontier here."""
    phase_state = STATE["phases"][phase["key"]]
    if summary_is_valid(phase):
        set_phase(
            phase["key"], "completed",
            finished_at=phase_state.get("finished_at") or now(),
        )
        return

    set_phase(
        phase["key"], "gpu_running",
        started_at=phase_state.get("started_at") or now(),
        finished_at=None,
    )
    event(
        f"GPU-first stage {phase['key']} started: T={phase['horizon']}, "
        f"history={phase['history']}, frames=8; frontier deferred"
    )
    mark_frontier_deferred(phase)
    run_baselines(phase)
    local_errors: list[str] = []

    def worker(arm: str, arm_args: tuple[str, ...]) -> None:
        try:
            run_local_arm(phase, arm, arm_args)
        except Exception as exc:
            with LOCK:
                local_errors.append(f"{arm}: {type(exc).__name__}: {exc}")

    workers = [
        threading.Thread(
            target=worker,
            args=(arm, arm_args),
            name=f"local-{phase['key']}-{arm}",
            daemon=False,
        )
        for arm, arm_args in LOCAL_ARMS
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if local_errors:
        raise RuntimeError(
            f"{phase['key']} local GPU arms failed: {'; '.join(local_errors)}"
        )
    set_phase(phase["key"], "waiting_frontier")
    event(f"GPU-first stage {phase['key']} completed; frontier remains deferred")


def run_frontier_and_score_phase(phase: dict) -> None:
    """After every GPU arm is done, finish frontier and score this phase."""
    phase_state = STATE["phases"][phase["key"]]
    if summary_is_valid(phase):
        set_phase(
            phase["key"], "completed",
            finished_at=phase_state.get("finished_at") or now(),
        )
        return

    set_phase(
        phase["key"], "frontier_running",
        started_at=phase_state.get("started_at") or now(),
        finished_at=None,
    )
    event(f"deferred frontier stage {phase['key']} started")
    frontier_errors: list[str] = []
    run_frontier_phase(phase, frontier_errors)
    if frontier_errors:
        raise RuntimeError(
            f"{phase['key']} frontier failed: {'; '.join(frontier_errors)}"
        )

    set_phase(phase["key"], "evaluating")
    score_phase(phase)
    set_phase(phase["key"], "completed", finished_at=now())
    event(f"phase {phase['key']} completed with six methods at 100% coverage")


def score_t4_cross_history() -> None:
    """Paired full-history minus no-history deltas on the identical T=4 set."""
    full = next(p for p in PHASES if p["key"] == "T4_full")
    nohist = next(p for p in PHASES if p["key"] == "T4_nohist")
    out_dir = OUT / "T4_cross_history"
    log = LOG_DIR / "T4_cross_history.log"
    results = {}
    for arm in ("ours_wm1st", "ours_full", "qwen_backbone", "frontier"):
        a_name = f"{arm}_fullhist"
        b_name = f"{arm}_nohist"
        cmd = [
            PYTHON, "-u", "-m", "ego.step3_results.vpa.v2.paired",
            "--gt", str(RUN / "vpa_v2_T4.json"),
            "--subset", str(RUN / "frames_subset_T4.json"),
            "--a", str(local_prefix(full, arm).with_suffix(".json")),
            "--a-name", a_name,
            "--b", str(local_prefix(nohist, arm).with_suffix(".json")),
            "--b-name", b_name,
            "--out-dir", str(out_dir),
        ]
        if run_command(cmd, log) != 0:
            raise RuntimeError(f"T4 cross-history paired failed for {arm}")
        path = out_dir / f"paired_{a_name}_vs_{b_name}_T4.json"
        result = load_json(path, {})
        if result.get("n_paired") != 504 or result.get("n_clusters") != 54:
            raise RuntimeError(f"T4 cross-history coverage failed for {arm}")
        results[arm] = result

    full_summary = load_json(OUT / "T4_full/summary.json", {})
    nohist_summary = load_json(OUT / "T4_nohist/summary.json", {})
    baseline_equal = {
        arm: full_summary.get("metrics", {}).get(arm)
        == nohist_summary.get("metrics", {}).get(arm)
        for arm in ("wm_top1_repeat", "wm_topk_rank")
    }
    if not all(baseline_equal.values()):
        raise RuntimeError("history-invariant WM baselines unexpectedly differ at T=4")
    atomic_json(
        out_dir / "summary.json",
        {
            "comparison": "T4 full history minus T4 no history",
            "with_frames": True,
            "n_paired": 504,
            "n_clusters": 54,
            "paired": results,
            "wm_baselines_identical": baseline_equal,
            "note": "DiD is optional per the handoff and is not required for the primary table.",
            "completed_at": now(),
        },
    )
    event("T4 paired full-history minus no-history analysis completed")


def main() -> int:
    acquire_singleton_lock()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if STATE.get("pipeline_state") == "running":
        event("pipeline resumed after a previous running state")
    set_pipeline(
        "preparing",
        started_at=STATE.get("started_at") or now(),
        finished_at=None,
        pid=os.getpid(),
        error=None,
    )
    try:
        prepare_subsets()
        event("dataset, prompt, adapter, and 100% frame-cache contracts verified")
        set_pipeline(
            "running",
            schedule="gpu_parallel_first_then_frontier",
            queue_stage="gpu",
        )
        for phase in PHASES:
            run_gpu_phase(phase)
        event("all local GPU arms completed; starting deferred frontier queue")
        set_pipeline("running", queue_stage="frontier")
        for phase in PHASES:
            run_frontier_and_score_phase(phase)
        score_t4_cross_history()
        set_pipeline("completed", finished_at=now(), pid=None)
        event("all requested VPA action-history ablations completed")
        return 0
    except KeyboardInterrupt:
        set_pipeline(
            "interrupted", finished_at=now(), pid=None,
            error="KeyboardInterrupt",
        )
        return 130
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        set_pipeline(
            "failed", finished_at=now(), pid=None,
            error=message,
            traceback=traceback.format_exc(),
        )
        event(f"pipeline failed: {message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
