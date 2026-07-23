#!/usr/bin/env python3
"""Unified, dependency-free dashboard for the active GoalStep experiment queue."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "datasets" / "Ego4D"

EXPERIMENTS = [
    {
        "id": "history-context-k8",
        "kind": "history_context",
        "p0b_policy": "diagnostic_only",
        "title": "A2.end−1s · visual history K=8",
        "subtitle": "현재 A2 시각 증거 + same-level 과거 8개 시각 요약 → strict-future A3",
        "run": "z1_history_context_k8_vna_ep10",
        "cache": "goalstep_feature_cache_end_m1_lobs8_vna",
        "config": "z1_history_context_k8_vna_ep10.yaml",
        "train_total": 30374,
        "val_total": 7214,
        "eligible_train": 29293,
        "eligible_val": 6960,
        "epochs": 10,
        "idle_state": "queued",
        "queue_note": "개정: P0-b는 진단용, P0-a 28.41을 champion으로 두고 K=8 Phase-1/2 실행",
    },
    {
        "id": "history-probe-zoo",
        "kind": "history_zoo",
        "title": "Phase 2 · history probe zoo (12-arm)",
        "subtitle": "Phase 1 기본 arm + LR×WD 11개 arm → fold-safe field ensemble",
        "run": "z1_history_context_probe_zoo_ep10",
        "cache": "goalstep_feature_cache_end_m1_lobs8_vna",
        "config": "z1_history_context_probe_zoo_ep10.yaml",
        "train_total": 30374,
        "val_total": 7214,
        "eligible_train": 29293,
        "eligible_val": 6960,
        "epochs": 10,
        "idle_state": "queued",
        "queue_note": "Phase 1 직후 자동 실행 · 11개 신규 arm, 기본 arm은 Phase 1 결과 재사용",
    },
    {
        "id": "start-m1-lobs8",
        "title": "action_start−1s · 8s",
        "subtitle": "A2 시작 1초 전까지 관찰 → A2 예측",
        "run": "z1_start_m1_lobs8_vna",
        "cache": "goalstep_feature_cache_start_m1_lobs8_vna",
        "config": "z1_start_m1_lobs8_vna.yaml",
        "train_total": 30374,
        "val_total": 7214,
        "eligible_train": 30374,
        "eligible_val": 7214,
        "epochs": 10,
        "idle_state": "completed",
        "queue_note": "완료 · best_action_top5.pt export 완료",
    },
    {
        "id": "end-m1-next",
        "title": "A2.end−1s · 8s → 다음 A3",
        "subtitle": "A2 종료 직전 관찰 → strict-future same-level A3 예측",
        "run": "z1_end_m1_lobs8_next_action_vna_ep10",
        "cache": "goalstep_feature_cache_end_m1_lobs8_vna",
        "config": "z1_end_m1_lobs8_next_action_vna_ep10.yaml",
        "train_total": 30374,
        "val_total": 7214,
        "eligible_train": 29293,
        "eligible_val": 6960,
        "epochs": 10,
        "idle_state": "stopped",
        "queue_note": "사용자 판단으로 epoch 9 도중 중단 · epoch 8까지 checkpoint 보존",
    },
    {
        "id": "start-m1-lobs16",
        "title": "action_start−1s · 16s",
        "subtitle": "A2 시작 1초 전까지 16초 관찰 → A2 예측",
        "run": "z1_start_m1_lobs16_vna",
        "cache": "goalstep_feature_cache_start_m1_lobs16_vna",
        "config": "z1_start_m1_lobs16_vna.yaml",
        "train_total": 30374,
        "val_total": 7214,
        "eligible_train": 30374,
        "eligible_val": 7214,
        "epochs": 10,
        "idle_state": "paused",
        "queue_note": "우선순위 변경으로 중단 · 생성된 cache 보존 · 재개 가능",
    },
    {
        "id": "adaptive-transition-mr24x8",
        "title": "adaptive A1 boundary · MR24+8",
        "subtitle": "A1 종료 직전까지 가변 관찰 → 가까운 same-level 다음 A2 예측",
        "run": "z1_adaptive_transition_mr24x8_vna_ep10",
        "cache": "goalstep_feature_cache_adaptive_transition_mr24x8_vna",
        "config": "z1_adaptive_transition_mr24x8_vna_ep10.yaml",
        "train_total": 18962,
        "val_total": 4458,
        "eligible_train": 18962,
        "eligible_val": 4458,
        "epochs": 10,
        "idle_state": "paused",
        "queue_note": "Phase 2 최종 리포트 완료 후 20:45 UTC 재개 · 기존 cache skip 후 남은 train 피처 추출 중",
    },
    {
        "id": "end-m1-baseline",
        "title": "action_end−1s · 8s · ep10",
        "subtitle": "A2 종료 1초 전까지 관찰 → A2 예측 (recognition 성격 baseline)",
        "run": "z1_end_m1_lobs8_vna_ep10",
        "cache": "goalstep_feature_cache_end_m1_lobs8_vna",
        "config": "z1_end_m1_lobs8_vna_ep10.yaml",
        "train_total": 30374,
        "val_total": 7214,
        "eligible_train": 30374,
        "eligible_val": 7214,
        "epochs": 10,
        "idle_state": "queued",
        "queue_note": "feature cache 완료 · 이전 시도는 epoch 1 전에 종료 · 재실행 예정",
    },
    {
        "id": "end-m6-lobs8",
        "title": "action_end−6s · 8s",
        "subtitle": "A2 종료 6초 전까지 관찰 → A2 예측",
        "run": "z1_end_m6_lobs8_vna_ep10",
        "cache": "goalstep_feature_cache_end_m6_lobs8_vna",
        "config": "z1_end_m6_lobs8_vna_ep10.yaml",
        "train_total": 30374,
        "val_total": 7214,
        "eligible_train": 30374,
        "eligible_val": 7214,
        "epochs": 10,
        "idle_state": "queued",
        "queue_note": "feature extraction부터 시작 예정",
    },
]


def tail(path: Path, lines: int = 18) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def read_history(run_dir: Path) -> list[dict]:
    path = run_dir / "training_history.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def process_snapshot() -> str:
    try:
        return subprocess.check_output(["ps", "-eo", "cmd"], text=True, timeout=3)
    except Exception:
        return ""


def gpu_stats() -> list[dict]:
    try:
        output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=3)
        keys = ("index", "name", "util", "memory_used", "memory_total", "temperature")
        return [dict(zip(keys, (x.strip() for x in row.split(",")))) for row in output.splitlines()]
    except Exception as exc:
        return [{"error": str(exc)}]


def cache_counts(cache_name: str, memo: dict[str, dict[str, int]]) -> dict[str, int]:
    if cache_name not in memo:
        root = DATA / cache_name
        memo[cache_name] = {
            split: sum(1 for _ in (root / split).glob("*.pt")) if (root / split).is_dir() else 0
            for split in ("train", "val")
        }
    return memo[cache_name]


def final_action_metrics(final: dict | None) -> dict | None:
    if not final:
        return None
    full = final.get("val_full", {}).get("metrics")
    if not full:
        return None
    return {
        "cmr5": full.get("overall_cmr5", {}).get("action"),
        "top1": full.get("accuracy_top1", {}).get("action"),
        "top5": full.get("accuracy_top5", {}).get("action"),
        "top10": full.get("accuracy_top10", {}).get("action"),
        "top15": full.get("accuracy_top15", {}).get("action"),
        "scope": "full validation",
        "best_epoch": final.get("best_epoch"),
    }


def history_final_action_metrics(final: dict | None) -> dict | None:
    if not final:
        return None
    action = final.get("best_val", {}).get("overall", {}).get("fused", {}).get("action")
    if not action:
        return None
    return {
        "cmr5": action.get("cmr@5"),
        "top1": action.get("top1"),
        "top5": action.get("top5"),
        "top10": action.get("top10"),
        "top15": action.get("top15"),
        "scope": "full validation · fused",
        "best_epoch": final.get("best_epoch"),
    }


def crossfit_action_metrics(result: dict | None, preferred: tuple[str, ...]) -> dict | None:
    metrics = (result or {}).get("metrics_percent", {})
    for name in preferred:
        action = metrics.get(name, {}).get("action")
        if action:
            return {
                "cmr5": action.get("cmr@5", action.get("cmr5")),
                "top1": action.get("top1"),
                "top5": action.get("top5"),
                "top10": action.get("top10"),
                "top15": action.get("top15"),
                "scope": f"video-disjoint OOF · {name}",
                "best_epoch": "fold-selected",
            }
    return None


def _derived_store_count(root: Path, split: str, total: int, shard_size: int = 1024) -> int:
    manifest = read_json(root / "manifest.json") or {}
    completed = manifest.get("splits", {}).get(split, {}).get("rows")
    if completed is not None:
        return min(total, int(completed))
    directory = root / split
    shards = sum(1 for _ in directory.glob("shard_*.pt")) if directory.is_dir() else 0
    return min(total, shards * shard_size)


def history_context_status(exp: dict, processes: str, cache_memo: dict) -> dict:
    run_dir = REPO / "outputs" / "goalstep" / "runs" / exp["run"]
    phase0_dir = REPO / "outputs" / "goalstep" / "runs" / "history_context_phase0"
    store_root = DATA / "goalstep_history_context_store"
    raw_history = read_history(run_dir)
    history = []
    for row in raw_history:
        adapted = dict(row)
        for key in ("top1", "top5", "top10", "top15"):
            adapted[f"action_{key}"] = row.get(f"fused_action_{key}", "")
        adapted["action_cmr@5"] = row.get("fused_action_cmr@5", "")
        history.append(adapted)

    final = read_json(run_dir / "final_metrics.json")
    crossfit = read_json(run_dir / "history_context_vs_p0a_results.json")
    gate_result = read_json(phase0_dir / "p0b_results.json")
    ensemble_result = read_json(phase0_dir / "p0a_primary_same_decision_results.json")
    gate = (gate_result or {}).get("gate", {})
    process_lines = processes.splitlines()
    phase0_running = any("run_history_phase0.py" in line for line in process_lines)
    store_running = any("prepare_history_context_store.py" in line for line in process_lines)
    training = any("train_goalstep_history_context.py" in line for line in process_lines)
    evaluating = any("evaluate_history_context_vs_p0a.py" in line for line in process_lines)

    if crossfit:
        phase = "completed"
    elif evaluating or final:
        phase = "crossfit_eval"
    elif training:
        phase = "training"
    elif store_running:
        phase = "derived_store"
    elif phase0_running:
        phase = "phase0"
    elif gate and not gate.get("passed", False) and exp.get("p0b_policy") != "diagnostic_only":
        phase = "gate_failed"
    else:
        phase = "queued"

    latest = history[-1] if history else None
    valid = [row for row in history if row.get("action_top5") not in (None, "")]
    best = max(valid, key=lambda row: float(row["action_top5"])) if valid else None
    ensemble_action = (
        (ensemble_result or {}).get("oof_fieldwise_ensemble", {}).get("action")
    )
    if best is None and ensemble_action:
        best = {
            "epoch": "P0-a",
            "action_cmr@5": ensemble_action.get("cmr5"),
            "action_top1": ensemble_action.get("top1"),
            "action_top5": ensemble_action.get("top5"),
            "action_top10": ensemble_action.get("top10"),
            "action_top15": ensemble_action.get("top15"),
        }
    final_metrics = crossfit_action_metrics(
        crossfit, ("final_blend", "phase1", "p0a")
    ) or history_final_action_metrics(final)
    raw_counts = cache_counts(exp["cache"], cache_memo)
    store_train = _derived_store_count(store_root, "train", exp["train_total"])
    store_val = _derived_store_count(store_root, "val", exp["val_total"])

    if phase == "completed":
        progress = 100.0
    elif phase == "crossfit_eval":
        progress = 98.0
    elif phase == "training":
        epoch = float(latest["epoch"]) if latest else 0.0
        progress = 45.0 + 55.0 * epoch / exp["epochs"]
    elif phase == "derived_store":
        progress = 15.0 + 30.0 * (store_train + store_val) / (
            exp["train_total"] + exp["val_total"]
        )
    elif phase == "gate_failed":
        progress = 15.0
    elif gate:
        progress = 15.0
    else:
        gate_log_text = "\n".join(tail(phase0_dir / "logs/gate.log", 80))
        matches = re.findall(r"cache pass (\d+)/(\d+)", gate_log_text)
        gate_fraction = (int(matches[-1][0]) / int(matches[-1][1])) if matches else 0.0
        progress = 1.0 + 13.0 * gate_fraction if phase == "phase0" else 0.0

    gate_note = exp["queue_note"]
    if gate:
        observed = gate.get("observed_percent")
        verdict = "PASS" if gate.get("passed") else "FAIL"
        role = "diagnostic" if exp.get("p0b_policy") == "diagnostic_only" else "gate"
        gate_note = f"P0-b {role} {verdict} · Action OOF Top-5 {observed:.2f}"
        if ensemble_action:
            gate_note += f" · P0-a same-decision ensemble {ensemble_action['top5']:.2f}"
    if crossfit:
        blend_action = (
            crossfit.get("metrics_percent", {}).get("final_blend", {}).get("action", {})
        )
        paired_blend = (
            crossfit.get("paired_action_top5", {}).get("final_blend_vs_p0a", {})
        )
        if blend_action.get("top5") is not None:
            gate_note += f" · Phase-1 blend OOF {float(blend_action['top5']):.2f}"
        if paired_blend.get("delta_top5_pp") is not None:
            interval = paired_blend.get("video_bootstrap_95ci_pp", [None, None])
            gate_note += f" (Δ {float(paired_blend['delta_top5_pp']):+.2f}pp"
            if interval[0] is not None:
                gate_note += f", CI low {float(interval[0]):+.2f}"
            gate_note += ")"
    logs = []
    for path, lines in (
        (phase0_dir / "logs/gate.log", 12),
        (run_dir / "logs/store.log", 8),
        (run_dir / "logs/train.log", 18),
        (run_dir / "logs/champion_eval.log", 12),
    ):
        logs.extend(tail(path, lines))

    return {
        **exp,
        "phase": phase,
        "progress": round(min(progress, 100.0), 2),
        "cache": {
            "train": {"done": raw_counts["train"], "total": exp["train_total"]},
            "val": {"done": raw_counts["val"], "total": exp["val_total"]},
        },
        "derived_store": {
            "train": {"done": store_train, "total": exp["train_total"]},
            "val": {"done": store_val, "total": exp["val_total"]},
        },
        "latest": latest,
        "best": best,
        "final_action": final_metrics,
        "history": history,
        "queue_note": gate_note,
        "logs": logs[-28:],
    }


def history_zoo_status(exp: dict, processes: str, cache_memo: dict) -> dict:
    run_dir = REPO / "outputs" / "goalstep" / "runs" / exp["run"]
    phase1_dir = REPO / "outputs" / "goalstep" / "runs" / "z1_history_context_k8_vna_ep10"
    raw_history = read_history(run_dir)
    by_epoch: dict[int, dict] = {}
    for row in raw_history:
        try:
            epoch = int(row["epoch"])
            value = float(row["fused_action_top5"])
        except (KeyError, TypeError, ValueError):
            continue
        if epoch not in by_epoch or value > float(by_epoch[epoch]["action_top5"]):
            by_epoch[epoch] = {
                "epoch": epoch,
                "arm_id": row.get("arm_id"),
                "action_cmr@5": row.get("fused_action_cmr@5"),
                "action_top1": row.get("fused_action_top1"),
                "action_top5": row.get("fused_action_top5"),
                "action_top10": row.get("fused_action_top10"),
                "action_top15": row.get("fused_action_top15"),
            }
    history = [by_epoch[key] for key in sorted(by_epoch)]
    latest = history[-1] if history else None
    best = max(history, key=lambda row: float(row["action_top5"])) if history else None
    zoo_final = read_json(run_dir / "final_metrics.json")
    selector_candidates = (
        run_dir / "history_probe_zoo_vs_p0a_results.json",
        run_dir / "history_context_probe_zoo_vs_p0a_results.json",
        run_dir / "phase2_vs_p0a_results.json",
    )
    selected_path = next((path for path in selector_candidates if path.is_file()), None)
    selected = read_json(selected_path) if selected_path else None
    process_lines = processes.splitlines()
    training = any("train_goalstep_history_probe_zoo.py" in line for line in process_lines)
    evaluating = any(
        "evaluate_history" in line and "zoo" in line for line in process_lines
    )
    phase1_complete = (phase1_dir / "history_context_vs_p0a_results.json").is_file()
    if selected:
        phase = "completed"
    elif evaluating or zoo_final:
        phase = "crossfit_eval"
    elif training:
        phase = "probe_zoo"
    else:
        phase = "queued"

    if phase == "completed":
        progress = 100.0
    elif phase == "crossfit_eval":
        progress = 98.0
    elif phase == "probe_zoo":
        progress = 5.0 + 90.0 * (float(latest["epoch"]) if latest else 0.0) / exp["epochs"]
    elif phase1_complete:
        progress = 5.0
    else:
        progress = 0.0

    promotion = (selected or {}).get("phase2_promotion", {})
    champion_after = promotion.get("champion_after_phase2")
    preferred_metrics = (
        ("phase1_incumbent", "final_blend", "p0a")
        if promotion.get("retained") is True
        or champion_after == "phase1_final_blend_incumbent_retained"
        else ("final_blend", "phase1_incumbent", "p0a")
    )
    final_metrics = crossfit_action_metrics(selected, preferred_metrics)
    counts = cache_counts(exp["cache"], cache_memo)
    store_root = DATA / "goalstep_history_context_store"
    logs = []
    for path, lines in (
        (run_dir / "logs/train.log", 22),
        (run_dir / "logs/champion_eval.log", 14),
        (run_dir / "logs/selection.log", 14),
    ):
        logs.extend(tail(path, lines))
    note = exp["queue_note"]
    if phase == "probe_zoo" and latest:
        note = (
            f"11-arm shared-loader 학습 중 · epoch {latest['epoch']} · "
            f"현재 full-val 최고 arm {latest.get('arm_id')}"
        )
    elif phase == "crossfit_eval":
        note = "학습 완료 · P0-a 기준 video-disjoint OOF 선택/검정 중"
    elif selected:
        comparison = promotion.get("paired_action_top5", {})
        delta = comparison.get("delta_top5_pp")
        interval = comparison.get("video_bootstrap_95ci_pp", [None, None])
        verdict = "Phase 2 잠정 승격" if promotion.get("promoted") else "Phase 1 incumbent 유지"
        note = verdict
        if delta is not None:
            note += f" · Δ {float(delta):+.2f}pp vs Phase 1"
        if interval[0] is not None:
            note += f" · CI low {float(interval[0]):+.2f}"
    return {
        **exp,
        "phase": phase,
        "progress": round(min(progress, 100.0), 2),
        "cache": {
            "train": {"done": counts["train"], "total": exp["train_total"]},
            "val": {"done": counts["val"], "total": exp["val_total"]},
        },
        "derived_store": {
            "train": {
                "done": _derived_store_count(store_root, "train", exp["train_total"]),
                "total": exp["train_total"],
            },
            "val": {
                "done": _derived_store_count(store_root, "val", exp["val_total"]),
                "total": exp["val_total"],
            },
        },
        "latest": latest,
        "best": best,
        "final_action": final_metrics,
        "history": history,
        "queue_note": note,
        "logs": logs[-30:],
    }


def experiment_status(exp: dict, processes: str, cache_memo: dict) -> dict:
    run_dir = REPO / "outputs" / "goalstep" / "runs" / exp["run"]
    history = read_history(run_dir)
    final = read_json(run_dir / "final_metrics.json")
    counts = cache_counts(exp["cache"], cache_memo)
    process_lines = processes.splitlines()
    extracting = any(
        exp["config"] in line and "extract_features.py" in line for line in process_lines
    )
    training = any(
        exp["config"] in line and "train_goalstep_z1.py" in line for line in process_lines
    )
    queue_log = tail(run_dir / "logs/queue.log", 20)
    queue_failed = any("ERROR:" in line for line in queue_log)

    if final:
        phase = "completed"
    elif training:
        phase = "training"
    elif extracting:
        phase = "feature_extraction"
    elif queue_failed:
        phase = "interrupted"
    else:
        phase = exp["idle_state"]

    latest = history[-1] if history else None
    best = None
    if history:
        valid = [row for row in history if row.get("action_top5") not in (None, "")]
        if valid:
            best = max(valid, key=lambda row: float(row["action_top5"]))
    final_metrics = final_action_metrics(final)

    total_cache = exp["train_total"] + exp["val_total"]
    done_cache = counts["train"] + counts["val"]
    if phase == "completed":
        progress = 100.0
    elif phase == "training" and latest:
        progress = 50.0 + 50.0 * float(latest["epoch"]) / exp["epochs"]
    elif phase == "training":
        progress = 50.0
    elif phase == "stopped" and latest:
        progress = 50.0 + 50.0 * float(latest["epoch"]) / exp["epochs"]
    else:
        progress = 50.0 * done_cache / max(1, total_cache)

    logs = []
    for filename, lines in (
        ("logs/queue.log", 5), ("logs/pipeline.log", 5), ("logs/train.log", 12),
        ("logs/extract_train.log", 4), ("logs/extract_val.log", 4),
    ):
        logs.extend(tail(run_dir / filename, lines))

    return {
        **exp,
        "phase": phase,
        "progress": round(min(progress, 100.0), 2),
        "cache": {
            "train": {"done": counts["train"], "total": exp["train_total"]},
            "val": {"done": counts["val"], "total": exp["val_total"]},
        },
        "latest": latest,
        "best": best,
        "final_action": final_metrics,
        "history": history,
        "logs": logs[-24:],
    }


def status() -> dict:
    processes = process_snapshot()
    memo: dict[str, dict[str, int]] = {}
    experiments = [
        history_context_status(exp, processes, memo)
        if exp.get("kind") == "history_context"
        else history_zoo_status(exp, processes, memo)
        if exp.get("kind") == "history_zoo"
        else experiment_status(exp, processes, memo)
        for exp in EXPERIMENTS
    ]
    order = [
        "training", "probe_zoo", "crossfit_eval", "derived_store", "phase0", "paused", "queued", "stopped",
        "interrupted", "gate_failed", "completed",
    ]
    counts = {state: sum(exp["phase"] == state for exp in experiments) for state in order}
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": counts,
        "experiments": experiments,
        "gpus": gpu_stats(),
    }


HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GoalStep Experiment Board</title><style>
:root{color-scheme:dark;--bg:#071018;--card:#101c27;--card2:#0a151e;--line:#243443;--text:#edf6ff;--muted:#91a3b5;--blue:#60a5fa;--mint:#5eead4;--amber:#fbbf24;--rose:#fb7185;--violet:#c084fc}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 12% -5%,#123047 0,transparent 32%),var(--bg);font:15px system-ui;color:var(--text)}.wrap{max-width:1180px;margin:auto;padding:30px 18px 70px}.top{position:sticky;top:0;z-index:5;background:#071018e8;backdrop-filter:blur(14px);border-bottom:1px solid var(--line);padding:16px 0;margin-bottom:22px}h1{margin:0;font-size:28px}.sub{color:var(--muted);margin-top:6px}.summary,.gpu,.metrics{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}.chip,.metric{background:var(--card2);border:1px solid var(--line);border-radius:11px;padding:9px 12px}.chip b{margin-left:7px}.run{background:#101c27e8;border:1px solid var(--line);border-radius:18px;margin:0 0 18px;padding:20px;box-shadow:0 16px 42px #0003}.runhead{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.run h2{font-size:21px;margin:0}.badge{white-space:nowrap;border-radius:99px;padding:7px 11px;font-size:12px;font-weight:750;text-transform:uppercase;letter-spacing:.08em}.completed{color:var(--mint);border:1px solid #2d6a62}.training,.probe_zoo{color:var(--blue);border:1px solid #335f87}.feature_extraction,.derived_store,.phase0,.crossfit_eval{color:var(--violet);border:1px solid #654a78}.paused{color:var(--amber);border:1px solid #725d2d}.queued{color:#c7d2fe;border:1px solid #495473}.stopped,.interrupted,.gate_failed{color:var(--rose);border:1px solid #713947}.bar{height:8px;background:#253440;border-radius:99px;overflow:hidden;margin:17px 0 8px}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--mint));transition:width .5s}.row{display:grid;grid-template-columns:1.1fr 1fr;gap:14px;margin-top:14px}.panel{background:var(--card2);border-radius:13px;padding:14px}.label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.11em}.metrics{display:grid;grid-template-columns:repeat(5,1fr)}.metric b{display:block;font-size:18px;margin-top:4px}.note{color:#c9d7e2;margin-top:9px}canvas{width:100%;height:190px;margin-top:8px}details{margin-top:13px}summary{cursor:pointer;color:var(--muted)}pre{white-space:pre-wrap;max-height:260px;overflow:auto;background:#061019;border:1px solid #1b2b38;border-radius:10px;padding:12px;color:#b9cedd;font:12px ui-monospace}.empty{color:var(--muted);padding:12px 0}@media(max-width:760px){.runhead{display:block}.badge{display:inline-block;margin-top:10px}.row{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap"><header class="top"><h1>GoalStep Experiment Board</h1><div class="sub">완료 · 진행 · 일시중단 · 예정 실험을 5초마다 갱신합니다. 아래로 스크롤해 전체 큐를 확인하세요.</div><div id="summary" class="summary"></div><div id="gpu" class="gpu"></div></header><main id="runs"><div class="empty">loading…</div></main></div>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>v==null||v===''?'—':Number(v).toFixed(2);const names={completed:'완료',training:'학습 중',probe_zoo:'Phase 2 zoo 학습 중',crossfit_eval:'OOF 평가 중',feature_extraction:'피처 추출 중',derived_store:'기존 피처 요약 중',phase0:'Phase-0 진단',paused:'일시중단',queued:'예정',stopped:'중단',interrupted:'오류 중단',gate_failed:'게이트 미달'};
function chart(canvas,rows,epochs){const x=canvas.getContext('2d'),W=canvas.width,H=canvas.height;x.clearRect(0,0,W,H);x.strokeStyle='#243443';x.fillStyle='#91a3b5';x.font='11px system-ui';for(let y=0;y<=100;y+=25){let py=H-22-y*(H-35)/100;x.beginPath();x.moveTo(32,py);x.lineTo(W-8,py);x.stroke();x.fillText(y,3,py+3)}[['action_top5','#60a5fa'],['action_top10','#c084fc'],['action_top15','#fb7185'],['action_top1','#5eead4']].forEach(([k,col],si)=>{x.strokeStyle=col;x.lineWidth=2.5;x.beginPath();rows.forEach((r,i)=>{let px=32+(Number(r.epoch)-1)*(W-45)/Math.max(1,epochs-1),py=H-22-Number(r[k])*(H-35)/100;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle=col;x.fillText(k.replace('action_',''),W-230+si*57,12)})}
function metricBlock(m,title){if(!m)return '<div class="empty">측정값 없음</div>';return `<div class="label">${esc(title)}</div><div class="metrics"><div class="metric"><span class="label">CMR@5</span><b>${num(m.cmr5??m['action_cmr@5'])}</b></div><div class="metric"><span class="label">Top-1</span><b>${num(m.top1??m.action_top1)}</b></div><div class="metric"><span class="label">Top-5</span><b>${num(m.top5??m.action_top5)}</b></div><div class="metric"><span class="label">Top-10</span><b>${num(m.top10??m.action_top10)}</b></div><div class="metric"><span class="label">Top-15</span><b>${num(m.top15??m.action_top15)}</b></div></div>`}
function runHTML(r){let shown=r.final_action||r.best||r.latest;let title=r.final_action?`best epoch ${r.final_action.best_epoch} · full validation`:r.best?`현재 best epoch ${r.best.epoch} · full validation`:'아직 평가 없음';let epoch=r.latest?`${r.latest.epoch} / ${r.epochs}`:`0 / ${r.epochs}`;let store=r.derived_store?`<div class="label" style="margin-top:12px">Derived history store</div><div class="metrics" style="grid-template-columns:1fr 1fr"><div class="metric"><span class="label">Train</span><b>${r.derived_store.train.done.toLocaleString()} / ${r.derived_store.train.total.toLocaleString()}</b></div><div class="metric"><span class="label">Val</span><b>${r.derived_store.val.done.toLocaleString()} / ${r.derived_store.val.total.toLocaleString()}</b></div></div>`:'';return `<article class="run"><div class="runhead"><div><h2>${esc(r.title)}</h2><div class="sub">${esc(r.subtitle)}</div></div><span class="badge ${r.phase}">${esc(names[r.phase]||r.phase)}</span></div><div class="bar"><div class="fill" style="width:${r.progress}%"></div></div><div class="sub">진행률 ${num(r.progress)}% · epoch ${epoch} · eligible train ${r.eligible_train.toLocaleString()} / val ${r.eligible_val.toLocaleString()}</div><div class="note">${esc(r.queue_note)}</div><div class="row"><section class="panel">${metricBlock(shown,title)}</section><section class="panel"><div class="label">Existing endpoint feature cache</div><div class="metrics" style="grid-template-columns:1fr 1fr"><div class="metric"><span class="label">Train</span><b>${r.cache.train.done.toLocaleString()} / ${r.cache.train.total.toLocaleString()}</b></div><div class="metric"><span class="label">Val</span><b>${r.cache.val.done.toLocaleString()} / ${r.cache.val.total.toLocaleString()}</b></div></div>${store}</section></div><section class="panel" style="margin-top:14px"><div class="label">Action accuracy curve</div>${r.history.length?`<canvas id="c-${r.id}" width="1080" height="190"></canvas>`:'<div class="empty">epoch 결과가 생기면 그래프가 표시됩니다.</div>'}</section><details><summary>최근 로그 보기</summary><pre>${esc(r.logs.join('\n')||'로그 없음')}</pre></details></article>`}
async function refresh(){try{const d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());document.getElementById('summary').innerHTML=Object.entries(d.summary).filter(([,n])=>n).map(([s,n])=>`<span class="chip ${s}">${esc(names[s]||s)} <b>${n}</b></span>`).join('')+`<span class="chip">UTC ${esc(d.updated_at.slice(11,19))}</span>`;document.getElementById('gpu').innerHTML=d.gpus.map(g=>g.error?`<span class="chip">${esc(g.error)}</span>`:`<span class="chip">GPU ${esc(g.index)} · ${esc(g.util)}% · ${Number(g.memory_used).toLocaleString()} / ${Number(g.memory_total).toLocaleString()} MiB · ${esc(g.temperature)}°C</span>`).join('');document.getElementById('runs').innerHTML=d.experiments.map(runHTML).join('');d.experiments.forEach(r=>{let c=document.getElementById('c-'+r.id);if(c)chart(c,r.history,r.epochs)})}catch(e){document.getElementById('summary').innerHTML='<span class="chip interrupted">reconnecting</span>'}}refresh();setInterval(refresh,5000);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(status(), ensure_ascii=False).encode()
            content_type = "application/json; charset=utf-8"
        elif self.path == "/" or self.path.startswith("/?"):
            body = HTML.encode()
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=17867)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
