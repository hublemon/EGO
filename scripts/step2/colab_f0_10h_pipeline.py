#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_f0_10h_pipeline.py — F0 v2 10시간 무인 파이프라인 (Colab A100 40GB, 단일 GPU).

final plan(docs/experiments/2026-07-18_f0_final_plan.md) §5 확정 실행 계획 중
**단일 A100 세션에서 정직하게 가능한 범위**를 시간예산 기반으로 자동 실행한다:

  Phase A  환경·데이터 점검 (분 단위)
  Phase B  4f-base held-out 평가          ← §5-#1 멀티프레임 게이트 (교란 분리 축 1)
  Phase B' (옵션) 1f-base 평가             ← v1 heldout jsonl 제공 시 (3중 비교 완성)
  Phase C  500-step 검증 run (r16·4f·gen8·T1.0·mask 0.0)   ← §5-#5. 데드라인 초과 시
           프로세스 중단 — 125 step 체크포인트가 살아남아 마지막 것을 평가
  Phase D  4f-trained 평가 (최신 체크포인트)  ← 교란 분리 축 2 (G1)
  Phase E  (시간 잔여 시) --no_memory 평가   ← handoff 배터리 ① (히스토리 기여도)
  Phase F  세션 리포트 (markdown + json) → out_root 에 저장

이 세션에서 **하지 않는 것** (계획상 다음 세션 — 교란 분리 원칙):
  - r64 ablation: 4f 효과 확정 후 분리 실행 (§0-3 "동시에 바꾸지 말 것")
  - L2-a mask 0.15~0.2: full run 에서 프록시 재악화 시에만
  - full-data ≥1 epoch (freeze 게이트): 단일 A100 에서 ≈5000 스텝 = 수 일 → 서버(2×H200) 몫
  - judge 곡선: GPU 불필요 (API) — 리포트에 실행 커맨드 기록, 별도 실행

사용 (Colab 셀 하나, VSCode Colab 커널에서도 동일):
    !cd /content/EGO && nohup python scripts/step2/colab_f0_10h_pipeline.py \
        --train_jsonl  /content/drive/MyDrive/ego/grpo_train.jsonl \
        --heldout_jsonl /content/drive/MyDrive/ego/grpo_heldout.jsonl \
        --path_map "/server/EGO/data=/content/drive/MyDrive/ego/data" \
        --out_root /content/drive/MyDrive/ego/runs/f0_v2_session1 \
        --budget_hours 10 --copy_frames_local > /content/pipeline.log 2>&1 &
    # 진행 확인: !tail -n 30 /content/pipeline.log
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_train_f0_full as base  # hr/sh/install_deps/prepare_jsonl 재사용

# v1 기록 앵커 (2026-07-17 f0_final, 1f·EK100 held-out 500) — 리포트 비교용 참조값
V1_ANCHORS = {"qwen3_1f_base_acc": 0.230, "qwen3_1f_step500_acc": 0.258,
              "wm_top1_ref_acc": 0.374, "candidate_recall_ceiling": 0.620}


def now_h(t0: float) -> float:
    return (time.time() - t0) / 3600


def run_eval(tag: str, jsonl: Path, out_dir: Path, model: str, limit: int,
             batch_size: int, adapter: str | None = None, no_memory: bool = False) -> dict | None:
    """evaluate.py 1회 실행 → 요약 dict (실패 시 None)."""
    out = out_dir / f"eval_{tag}.json"
    cmd = [sys.executable, str(REPO / "src/ego/step2_vlm_alignment/evaluate.py"),
           "--jsonl", str(jsonl), "--model_name", model,
           "--reward_mode", "wm_likelihood_joint", "--limit", str(limit),
           "--batch_size", str(batch_size), "--max_new_tokens", "384",
           "--out", str(out)]
    if adapter:
        cmd += ["--adapter", adapter]
    if no_memory:
        cmd += ["--no_memory"]
    t = time.time()
    r = base.sh(cmd, cwd=str(REPO))
    dt = (time.time() - t) / 60
    if r.returncode != 0 or not out.exists():
        print(f"✗ eval[{tag}] 실패 (code={r.returncode}, {dt:.0f}min)")
        return None
    d = json.loads(out.read_text(encoding="utf-8"))
    d["_eval_minutes"] = round(dt, 1)
    print(f"[eval:{tag}] {dt:.0f}min → {out.name}")
    return d


def key_metrics(d: dict | None) -> dict:
    """평가 요약에서 핵심 지표만 추출 (키 이름 편차 허용 — 부분 문자열 매칭)."""
    if not d:
        return {"status": "failed"}
    flat = {}

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(obj, (int, float)):
            flat[prefix[:-1]] = obj
    walk(d)
    out = {}
    for want in ("acc", "g2", "wm_follow", "in_joint", "invalid", "verb", "noun"):
        for k, v in flat.items():
            if want in k.lower():
                out[k] = v
    out["_eval_minutes"] = d.get("_eval_minutes")
    return out


def train_with_deadline(a, train_jsonl: Path, out_dir: Path, deadline_ts: float) -> tuple[int, bool]:
    """검증 run 을 데드라인과 함께 실행. (마지막 스텝 추정, 데드라인 중단 여부) 반환."""
    cmd = [sys.executable, str(REPO / "scripts/step2/colab_train_f0_full.py"),
           "--train_jsonl", str(train_jsonl), "--out_dir", str(out_dir),
           "--run_mode", "validation", "--model", a.model,
           "--per_device_batch", str(a.per_device_batch), "--grad_accum", str(a.grad_accum),
           "--no_install"]
    print(f"[train] deadline: {datetime.fromtimestamp(deadline_ts):%H:%M} "
          f"(잔여 {(deadline_ts - time.time()) / 3600:.1f}h)")
    proc = subprocess.Popen(cmd, cwd=str(REPO))
    killed = False
    while proc.poll() is None:
        if time.time() > deadline_ts:
            print("[train] ⏰ 데드라인 도달 — SIGTERM (체크포인트는 125 step 마다 저장돼 있음)")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.kill()
            killed = True
            break
        time.sleep(30)
    ckpts = sorted(out_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    last = int(ckpts[-1].name.split("-")[1]) if ckpts else 0
    print(f"[train] 종료 (killed={killed}) 최신 체크포인트: {last or '없음'}")
    return last, killed


def main() -> None:
    p = argparse.ArgumentParser(description="F0 v2 10h unattended pipeline (A100 40GB)")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--heldout_jsonl", required=True, help="v2(4f) held-out jsonl")
    p.add_argument("--heldout_1f_jsonl", default=None,
                   help="(옵션) v1(1f) held-out — 제공 시 3중 비교 완성")
    p.add_argument("--path_map", action="append")
    p.add_argument("--out_root", required=True, help="세션 산출물 루트 — Drive 권장")
    p.add_argument("--budget_hours", type=float, default=10.0)
    p.add_argument("--eval_reserve_hours", type=float, default=1.7,
                   help="trained 평가+리포트용 예약 시간 (학습 데드라인 = 예산 - 소비 - 예약)")
    p.add_argument("--eval_limit", type=int, default=500)
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--copy_frames_local", action="store_true")
    p.add_argument("--no_install", action="store_true")
    p.add_argument("--skip_base_eval", action="store_true",
                   help="4f-base 평가 생략 (이미 이전 세션에서 확보한 경우)")
    a = p.parse_args()

    t0 = time.time()
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train_out = out_root / "train_val500"
    work = Path(os.environ.get("EGO_FULL_WORK", "/content/f0_work"))
    report: dict = {"started": datetime.now().isoformat(timespec="seconds"),
                    "budget_hours": a.budget_hours, "anchors_v1": V1_ANCHORS,
                    "plan": "final_plan §5: #1 gate eval → #5 validation run → G1 eval → battery①",
                    "phases": {}}

    base.hr("F0 v2 — 10h 무인 파이프라인")
    print(f"budget={a.budget_hours}h  eval_limit={a.eval_limit}  out={out_root}")

    # Phase A — 환경 + 데이터
    base.install_deps(a.no_install)
    ns = argparse.Namespace(train_jsonl=a.train_jsonl, path_map=a.path_map,
                            copy_frames_local=a.copy_frames_local)
    train_jsonl = base.prepare_jsonl(ns, work)
    ns_h = argparse.Namespace(train_jsonl=a.heldout_jsonl, path_map=a.path_map,
                              copy_frames_local=a.copy_frames_local)
    heldout_jsonl = base.prepare_jsonl(ns_h, work / "heldout")
    heldout_1f = None
    if a.heldout_1f_jsonl:
        ns_1f = argparse.Namespace(train_jsonl=a.heldout_1f_jsonl, path_map=a.path_map,
                                   copy_frames_local=a.copy_frames_local)
        heldout_1f = base.prepare_jsonl(ns_1f, work / "heldout_1f")
    report["phases"]["A_setup_hours"] = round(now_h(t0), 2)

    # Phase B — 4f-base 게이트 평가 (§5-#1 최우선)
    if not a.skip_base_eval:
        base.hr("Phase B — 4f-base held-out 평가 (멀티프레임 게이트)")
        d = run_eval("4f_base", heldout_jsonl, out_root, a.model, a.eval_limit, a.eval_batch)
        report["phases"]["B_4f_base"] = key_metrics(d)
    if heldout_1f is not None:
        base.hr("Phase B' — 1f-base 평가 (3중 비교)")
        d = run_eval("1f_base", heldout_1f, out_root, a.model, a.eval_limit, a.eval_batch)
        report["phases"]["B1_1f_base"] = key_metrics(d)

    # Phase C — 500-step 검증 run (남은 시간 - 예약)
    base.hr("Phase C — 500-step 검증 run")
    deadline = t0 + (a.budget_hours - a.eval_reserve_hours) * 3600
    if deadline - time.time() < 3600:
        print("⚠ 학습 가용시간 < 1h — 그래도 시작한다 (125 step 체크포인트 단위 회수)")
    last_step, deadline_killed = train_with_deadline(a, train_jsonl, train_out, deadline)
    report["phases"]["C_train"] = {"last_checkpoint_step": last_step,
                                   "deadline_killed": deadline_killed,
                                   "elapsed_hours": round(now_h(t0), 2),
                                   "steps_per_hour": None}
    if last_step:
        train_hours = report["phases"]["C_train"]["elapsed_hours"] - report["phases"]["A_setup_hours"]
        report["phases"]["C_train"]["steps_per_hour"] = round(last_step / max(train_hours, 0.01), 1)

    # Phase D — trained 평가 (최신 체크포인트)
    if last_step:
        base.hr(f"Phase D — 4f-trained 평가 (checkpoint-{last_step})")
        d = run_eval(f"4f_trained_step{last_step}", heldout_jsonl, out_root, a.model,
                     a.eval_limit, a.eval_batch, adapter=str(train_out / f"checkpoint-{last_step}"))
        report["phases"]["D_4f_trained"] = key_metrics(d)
    else:
        print("✗ 체크포인트 없음 — Phase D 생략")
        report["phases"]["D_4f_trained"] = {"status": "no_checkpoint"}

    # Phase E — 잔여 시간에 배터리 ① (--no_memory)
    remain_h = a.budget_hours - now_h(t0)
    if last_step and remain_h > 0.8:
        base.hr(f"Phase E — --no_memory 평가 (잔여 {remain_h:.1f}h, 배터리 ①)")
        d = run_eval(f"4f_trained_step{last_step}_nomem", heldout_jsonl, out_root, a.model,
                     a.eval_limit, a.eval_batch,
                     adapter=str(train_out / f"checkpoint-{last_step}"), no_memory=True)
        report["phases"]["E_trained_no_memory"] = key_metrics(d)
        remain_h = a.budget_hours - now_h(t0)
        if remain_h > 0.8:
            d = run_eval("4f_base_nomem", heldout_jsonl, out_root, a.model,
                         a.eval_limit, a.eval_batch, no_memory=True)
            report["phases"]["E_base_no_memory"] = key_metrics(d)
    else:
        print(f"[E] 잔여 {remain_h:.1f}h — no_memory 평가는 다음 세션으로")

    # Phase F — 리포트
    base.hr("Phase F — 세션 리포트")
    report["finished"] = datetime.now().isoformat(timespec="seconds")
    report["total_hours"] = round(now_h(t0), 2)
    report["next_session"] = [
        "r64 ablation 500-step (4f 효과 확정 후 — 교란 분리, plan §0-3)",
        "L2-a mask 0.15~0.2 run (reasoning_proxy.jsonl 재악화 시에만)",
        "judge 곡선: judge_reasoning.py --judge_model gemini-2.5-pro, ckpt 125/250/375/500 (~$0.3, GPU 불필요)",
        "full-data ≥1 epoch (freeze 게이트) — 서버(2×H200) 실행: RUN_MODE=full train_f0_final_v2.sh",
        "1f-base 평가 미실시 시 v1 heldout 로 보완 (3중 비교)",
    ]
    (out_root / "session_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# F0 v2 세션 리포트 (Colab A100)", "",
          f"- 시작 {report['started']} · 종료 {report['finished']} · 총 {report['total_hours']}h",
          f"- 학습: checkpoint-{last_step} (deadline_killed={deadline_killed}, "
          f"{report['phases']['C_train'].get('steps_per_hour')} steps/h)",
          "", "## 지표 (앵커: v1 1f-base 0.230 · 1f-step500 0.258 · WM-ref 0.374)", ""]
    for ph, m in report["phases"].items():
        md.append(f"### {ph}")
        if isinstance(m, dict):
            for k, v in m.items():
                md.append(f"- {k}: {v}")
        md.append("")
    md += ["## 다음 세션", ""] + [f"- {x}" for x in report["next_session"]]
    (out_root / "session_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"리포트 → {out_root}/session_report.md")
    print(f"프록시 로그 → {train_out}/reasoning_proxy.jsonl (히스토리 참조율·belief 재진술율)")
    print("[DONE]")


if __name__ == "__main__":
    main()
