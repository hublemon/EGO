#!/usr/bin/env python3
"""Render the action_end-1s GoalStep run artifacts as a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_history(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def tunnel_url(path: Path) -> str | None:
    if not path.is_file():
        return None
    matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", path.read_text(errors="replace"))
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--index-stats", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", choices=["running", "completed", "failed"], default="running")
    args = parser.parse_args()

    run = Path(args.run_dir).resolve()
    stats = load_json(Path(args.index_stats).resolve()) or {}
    final = load_json(run / "final_metrics.json")
    metadata = load_json(run / "run_metadata.json") or {}
    history = load_history(run / "training_history.csv")
    url = tunnel_url(run / "logs/cloudflared.log")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        "# GoalStep action_end−1s / 8초 V/N/A 실험",
        "",
        f"- 마지막 자동 갱신: `{now}`",
        f"- 상태: **{args.status}**",
        f"- 실시간 UI: {url or '터널 URL 생성 대기 중'}",
        f"- tmux: `ego_goalstep_end_m1_lobs8_vna`",
        "",
        "## 실험 정의",
        "",
        "기존 `action_start−1s` GoalStep index의 행 순서와 V/N/A label을 고정하고, "
        "observation endpoint만 `action_end−1s`로 바꾼 공개 V-JEPA EK100 loader 진단 실험이다. "
        "관측 길이는 최대 8초이며 32 frame을 균일 샘플링하므로 실효 4fps다.",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| train / val sample | {stats.get('train', {}).get('samples', 30374):,} / {stats.get('val', {}).get('samples', 7214):,} |",
        "| label space | verb 81 / noun 140 / action 293 |",
        "| endpoint | `target_end_sec - 1.0s` |",
        "| observation | 최대 8초, 32 frames, 4fps |",
        "| backbone | frozen V-JEPA2 ViT-L/16, 256 |",
        "| probe | depth 4, 16 heads |",
        "| supervision | verb + noun + action focal loss |",
        "| sampler | random, 전체 sample 1회/epoch |",
        "| precision | train BF16 autocast / eval FP32 |",
        f"| epochs | {metadata.get('epochs', 15)} |",
        f"| batch | {metadata.get('batch_size', 32)} |",
        "| LR / WD | 3e-4 / 1e-4 |",
        "",
        "## Endpoint 변화의 실측 특성",
        "",
        f"- train target action 일부가 보이는 비율: `{100 * stats.get('train', {}).get('target_action_visible_fraction', 0):.3f}%`",
        f"- val target action 일부가 보이는 비율: `{100 * stats.get('val', {}).get('target_action_visible_fraction', 0):.3f}%`",
        f"- train target action 관측량 중앙값: `{stats.get('train', {}).get('target_action_visible_seconds_median', 0):.3f}s`",
        f"- val target action 관측량 중앙값: `{stats.get('val', {}).get('target_action_visible_seconds_median', 0):.3f}s`",
        "",
        "이 수치는 진짜 anticipation 난도보다 recognition 성격이 강한 의도적인 대조군이다.",
        "",
        "## Epoch별 validation 실측값",
        "",
        "| Epoch | Loss | V CMR@5 | V Top-5 | N CMR@5 | N Top-5 | A CMR@5 | A Top-1 | A Top-5 | sec |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if history:
        for row in history:
            lines.append("| " + " | ".join([
                str(row.get("epoch", "-")), fmt(row.get("train_loss")),
                fmt(row.get("verb_cmr@5")), fmt(row.get("verb_top5")),
                fmt(row.get("noun_cmr@5")), fmt(row.get("noun_top5")),
                fmt(row.get("action_cmr@5")), fmt(row.get("action_top1")),
                fmt(row.get("action_top5")), fmt(row.get("seconds")),
            ]) + " |")
    else:
        lines.append("| - | feature 추출/첫 epoch 대기 중 | - | - | - | - | - | - | - | - |")

    lines.extend(["", "## 최종 full-validation 결과", ""])
    if final and final.get("val_full"):
        metrics = final["val_full"]["metrics"]
        cmr = metrics.get("overall_cmr5", {})
        top1 = metrics.get("accuracy_top1", {})
        top5 = metrics.get("accuracy_top5", {})
        lines.extend([
            f"- 선택 epoch: **{final.get('best_epoch')}** "
            f"(`{final.get('checkpoint_selection_metric', 'legacy_action_cmr5')}` 기준)",
            f"- full-val size: **{final['val_full'].get('size')}**",
            "",
            "| Head | CMR@5 | Top-1 | Top-5 |",
            "|---|---:|---:|---:|",
        ])
        for head in ("verb", "noun", "action"):
            lines.append(f"| {head} | {fmt(cmr.get(head))} | {fmt(top1.get(head))} | {fmt(top5.get(head))} |")
    else:
        lines.append("15 epoch 완료 후 자동으로 채워진다.")

    lines.extend([
        "",
        "## 산출물",
        "",
        f"- config: `{Path(args.config)}`",
        "- index: `src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8/`",
        "- feature cache: `../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna/`",
        "- run: `outputs/goalstep/runs/z1_end_m1_lobs8_vna/`",
        "- epoch checkpoints: `outputs/goalstep/runs/z1_end_m1_lobs8_vna/checkpoints/epoch_XX.pt`",
        "- final selection: `best.pt`, `best_action_top5.pt`, `latest.pt`, `final_metrics.json`",
        "",
        "## 운영 명령",
        "",
        "```bash",
        "tmux list-windows -t ego_goalstep_end_m1_lobs8_vna",
        "tmux attach -t ego_goalstep_end_m1_lobs8_vna",
        "tail -f outputs/goalstep/runs/z1_end_m1_lobs8_vna/logs/pipeline.log",
        "```",
        "",
        "SSH/VS Code/GPT 세션 종료는 tmux 내부의 feature 추출, 학습, UI, tunnel, reporter에 영향을 주지 않는다. "
        "서버 재부팅 또는 tmux server 종료는 별도 예외다. Cloudflare quick-tunnel URL은 프로세스 재시작 시 바뀔 수 있다.",
        "",
    ])

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
