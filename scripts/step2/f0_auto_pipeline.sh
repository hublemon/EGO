#!/usr/bin/env bash
# f0_auto_pipeline.sh — 사전 검증 배터리(①②⑤) → v2 데이터 빌드 → 500-step 검증 run
# → 체크포인트 heldout 평가 → 결과 md 생성까지 무인 진행하는 오케스트레이터.
#
# 실행 (VSCode/세션 독립 — setsid nohup):
#   setsid nohup bash scripts/step2/f0_auto_pipeline.sh > $EGO_ROOT/runs/f0_auto.log 2>&1 < /dev/null &
#
# 설계:
#  - 전 단계 멱등: 산출물이 있으면 건너뜀 → 중단 시 그냥 재실행하면 이어서 진행.
#  - 4f 채택 게이트: 배터리 ⑤(4f-base acc > 0.30, plan §5-#1) 실패 시
#    NEEDS_DECISION 마커를 남기고 학습 전에 정지한다 (연구 결정 사항 — 자동화 금지).
#  - leakage 게이트(train+heldout) FAIL 시 학습 전에 정지 (freeze 게이트, plan §0-4).
set -uo pipefail

# ── 환경 ──────────────────────────────────────────────────────────────
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export EK100_VIDEOS="${EK100_VIDEOS:-/mnt/ddn/prod-shared/datasets/EK100/videos}"
export HF_HOME=/mnt/nvme/cache
export TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"
PY=$ENVBIN/python

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
GD="$EGO_ROOT/data/grpo_dataset"
BAT="$EGO_ROOT/runs/f0_battery"
SB1="${SB1:-$BAT/heldout_1f_root}"          # 1f-strict 대조용 EGO_ROOT (심링크 샌드박스)
mkdir -p "$BAT"
DATA=src/ego/step2_vlm_alignment/data

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { say "✗ $*"; echo "$*" > "$BAT/PIPELINE_FAILED"; exit 1; }

say "=== F0 auto pipeline 시작 (repo=$REPO EGO_ROOT=$EGO_ROOT) ==="

# ── [0] heldout 추출 경합 방지 (train 추출 대기는 [5] 직전 — 배터리와 병행) ──
while pgrep -f "extract_frame_train.py --split validation" > /dev/null; do
  say "대기: heldout 프레임 추출 진행 중..."; sleep 60; done

# ── [1] heldout 4f 프레임 추출 (멱등 — 기하 일치 jpg 는 스킵) ──────────
say "[1] heldout 4f 프레임 추출 (resume)"
$PY $DATA/extract_frame_train.py --split validation --num_frames 4 --resume || die "heldout 프레임 추출 실패"

# ── [2] heldout v2(4f-strict) 빌드 ─────────────────────────────────────
say "[2] heldout 4f-strict 빌드"
[ -s "$GD/memory_heldout.jsonl" ] || $PY $DATA/extract_memory_train.py --split validation --future-k 5 || die "memory heldout 실패"
$PY $DATA/assemble_train.py --split validation || die "assemble heldout 실패"
$PY $DATA/convert_to_train_format.py --input "$GD/grpo_dataset_heldout.jsonl" \
    --output "$GD/grpo_heldout.jsonl" || die "convert heldout 실패"

# ── [3] 1f-strict 대조 셋 (backup 프레임 + 합성 1f manifest) ───────────
if [ ! -s "$SB1/data/grpo_dataset/grpo_heldout_1f.jsonl" ]; then
  say "[3] 1f-strict 대조 셋 구성"
  mkdir -p "$SB1/data/grpo_dataset" "$SB1/src"
  for f in selected_heldout.jsonl predictions_heldout.jsonl memory_heldout.jsonl; do
    ln -sfn "$GD/$f" "$SB1/data/grpo_dataset/$f"; done
  ln -sfn "$GD/frames_1f_backup" "$SB1/data/grpo_dataset/frames"
  ln -sfn "$EGO_ROOT/src/epic-kitchens-100-annotations" "$SB1/src/epic-kitchens-100-annotations"
  $PY - "$GD" "$SB1" <<'PYEOF'
import json, sys
gd, sb = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{gd}/selected_heldout.jsonl")]
with open(f"{sb}/data/grpo_dataset/frames_manifest_heldout.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"sample_id": r["sample_id"], "video_id": r["video_id"],
                            "trigger_frame": int(r["trigger_frame"]),
                            "n_frames": 1, "offsets_sec": [0.0]}) + "\n")
PYEOF
  EGO_ROOT="$SB1" $PY $DATA/assemble_train.py --split validation || die "assemble 1f 실패"
  EGO_ROOT="$SB1" $PY $DATA/convert_to_train_format.py \
      --input "$SB1/data/grpo_dataset/grpo_dataset_heldout.jsonl" \
      --output "$SB1/data/grpo_dataset/grpo_heldout_1f.jsonl" || die "convert 1f 실패"
fi
J1F="$SB1/data/grpo_dataset/grpo_heldout_1f.jsonl"
J4F="$GD/grpo_heldout.jsonl"

# ── [4] 배터리 평가 (①②⑤) — 2 GPU 페어 2라운드, 멱등 ────────────────
run_eval() {  # $1 out-name  $2 device  $3.. extra args
  local name=$1 dev=$2; shift 2
  if [ -s "$BAT/$name.json" ] && [ -s "$BAT/$name.records.jsonl" ]; then
    say "  skip $name (완료)"; return 0; fi
  say "  eval $name (${dev})"
  $PY scripts/step2/eval_battery.py --device "$dev" --out "$BAT/$name.json" "$@" \
      > "$BAT/$name.log" 2>&1
}
say "[4] 배터리 라운드 1: 1f-base + ①no-memory"
run_eval base_1f_strict       cuda:0 --jsonl "$J1F" &
run_eval base_1f_strict_nomem cuda:1 --jsonl "$J1F" --no_memory &
wait
say "[4] 배터리 라운드 2: ②history-only + ⑤4f-base"
run_eval base_1f_histonly cuda:0 --jsonl "$J1F" --history_only &
run_eval base_4f_strict   cuda:1 --jsonl "$J4F" &
wait
for n in base_1f_strict base_1f_strict_nomem base_1f_histonly base_4f_strict; do
  [ -s "$BAT/$n.json" ] || die "배터리 $n 실패 — $BAT/$n.log 확인"
done

# ── [5] train v2 빌드 + leakage 게이트 ────────────────────────────────
while pgrep -f "extract_frame_train.py --split train" > /dev/null; do
  say "대기: train 프레임 추출 진행 중..."; sleep 60; done
say "[5] train 4f 프레임 추출 마무리(resume) + 빌드 + leakage 게이트"
$PY $DATA/extract_frame_train.py --split train --num_frames 4 --resume || die "train 프레임 추출 실패"
$PY $DATA/extract_memory_train.py --split train --future-k 5 || die "memory train 실패"
$PY $DATA/assemble_train.py --split train || die "assemble train 실패"
$PY $DATA/convert_to_train_format.py --input "$GD/grpo_dataset.jsonl" \
    --output "$GD/grpo_train.jsonl" || die "convert train 실패"
$PY scripts/step2/check_leakage.py \
    --memory "$GD/memory_train.jsonl" --train_jsonl "$GD/grpo_train.jsonl" \
    --b0_meta "$GD/grpo_train_b0meta.jsonl" \
    --csv "$EGO_ROOT/src/epic-kitchens-100-annotations/EPIC_100_train.csv" \
    --selected "$GD/selected_train.jsonl" || die "leakage 게이트 FAIL (train)"
$PY scripts/step2/check_leakage.py \
    --memory "$GD/memory_heldout.jsonl" --train_jsonl "$GD/grpo_heldout.jsonl" \
    --b0_meta "$GD/grpo_heldout_b0meta.jsonl" \
    --csv "$EGO_ROOT/src/epic-kitchens-100-annotations/EPIC_100_validation.csv" \
    --selected "$GD/selected_heldout.jsonl" || die "leakage 게이트 FAIL (heldout)"

# ── [6] 입력 계약 결정 (배터리 ⑤ 게이트, plan §5-#1) ──────────────────
# 2026-07-18 게이트 결과: 4f-base acc 0.226 ≤ 0.30 → 미달. 담당자 결정 = 1f 유지 (기본값).
INPUT_CONTRACT="${INPUT_CONTRACT:-1f}"      # 1f | 4f (4f 는 게이트 통과 또는 FORCE_4F=1 시)
if [ "$INPUT_CONTRACT" = "4f" ]; then
  ACC4F=$($PY -c "import json; print(json.load(open('$BAT/base_4f_strict.json'))['acc'])")
  say "[6] 4f-base acc = $ACC4F (게이트 0.30)"
  GATE=$($PY -c "print(int(float('$ACC4F') > 0.30))")
  if [ "$GATE" != "1" ] && [ "${FORCE_4F:-0}" != "1" ]; then
    die "4f 게이트 미달 — INPUT_CONTRACT=1f 로 실행하거나 FORCE_4F=1 로 강행"
  fi
  TRAIN_JSONL="$GD/grpo_train.jsonl"; EVAL_JSONL="$J4F"; NF=4; SUF=4f
else
  say "[6] 입력 계약 = 1f (게이트 미달 → 담당자 결정 반영)"
  SBT="$BAT/train_1f_root"
  if [ ! -s "$SBT/data/grpo_dataset/grpo_train_1f.jsonl" ]; then
    mkdir -p "$SBT/data/grpo_dataset" "$SBT/src"
    for f in selected_train.jsonl predictions_train.jsonl memory_train.jsonl; do
      ln -sfn "$GD/$f" "$SBT/data/grpo_dataset/$f"; done
    ln -sfn "$GD/frames_1f_backup" "$SBT/data/grpo_dataset/frames"
    ln -sfn "$EGO_ROOT/src/epic-kitchens-100-annotations" "$SBT/src/epic-kitchens-100-annotations"
    $PY - "$GD" "$SBT" <<'PYEOF'
import json, sys
gd, sb = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{gd}/selected_train.jsonl")]
with open(f"{sb}/data/grpo_dataset/frames_manifest.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"sample_id": r["sample_id"], "video_id": r["video_id"],
                            "trigger_frame": int(r["trigger_frame"]),
                            "n_frames": 1, "offsets_sec": [0.0]}) + "\n")
PYEOF
    EGO_ROOT="$SBT" $PY $DATA/assemble_train.py --split train || die "assemble train 1f 실패"
    EGO_ROOT="$SBT" $PY $DATA/convert_to_train_format.py \
        --input "$SBT/data/grpo_dataset/grpo_dataset.jsonl" \
        --output "$SBT/data/grpo_dataset/grpo_train_1f.jsonl" || die "convert train 1f 실패"
    $PY scripts/step2/check_leakage.py \
        --memory "$GD/memory_train.jsonl" \
        --train_jsonl "$SBT/data/grpo_dataset/grpo_train_1f.jsonl" \
        --b0_meta "$SBT/data/grpo_dataset/grpo_train_1f_b0meta.jsonl" \
        --csv "$EGO_ROOT/src/epic-kitchens-100-annotations/EPIC_100_train.csv" \
        --selected "$GD/selected_train.jsonl" || die "leakage 게이트 FAIL (train 1f)"
  fi
  TRAIN_JSONL="$SBT/data/grpo_dataset/grpo_train_1f.jsonl"; EVAL_JSONL="$J1F"; NF=1; SUF=1f
fi

# ── [7] 500-step 검증 run (2×H200) ────────────────────────────────────
CKPT_DIR="$REPO/outputs/step2/f0_final_v2_val_${SUF}"
if [ ! -d "$CKPT_DIR/checkpoint-500" ]; then
  say "[7] 500-step 검증 run 시작 (${SUF})"
  RUN_MODE=validation NUM_FRAMES=$NF EGO_TRAIN_JSONL="$TRAIN_JSONL" \
    bash scripts/step2/train_f0_final_v2.sh || die "500-step run 실패"
else
  say "[7] skip — checkpoint-500 존재"
fi
[ -d "$CKPT_DIR/checkpoint-500" ] || die "checkpoint-500 미생성"

# ── [8] 체크포인트 heldout 평가 (G1 일치율·G2 동시 추적) ──────────────
say "[8] 체크포인트 평가 (125/250/375/500, ${SUF})"
run_eval "step125_${SUF}" cuda:0 --jsonl "$EVAL_JSONL" --adapter "$CKPT_DIR/checkpoint-125" &
run_eval "step250_${SUF}" cuda:1 --jsonl "$EVAL_JSONL" --adapter "$CKPT_DIR/checkpoint-250" &
wait
run_eval "step375_${SUF}" cuda:0 --jsonl "$EVAL_JSONL" --adapter "$CKPT_DIR/checkpoint-375" &
run_eval "step500_${SUF}" cuda:1 --jsonl "$EVAL_JSONL" --adapter "$CKPT_DIR/checkpoint-500" &
wait

# ── [9] 결과 md 자동 생성 ─────────────────────────────────────────────
say "[9] 결과 md 생성"
$PY - "$BAT" <<'PYEOF'
import json, sys
from pathlib import Path
bat = Path(sys.argv[1])
names = ["base_1f_strict", "base_1f_strict_nomem", "base_1f_histonly", "base_4f_strict",
         "step125_1f", "step250_1f", "step375_1f", "step500_1f",
         "step125_4f", "step250_4f", "step375_4f", "step500_4f"]
label = {"base_1f_strict": "1f-base (strict)", "base_1f_strict_nomem": "① no-memory",
         "base_1f_histonly": "② history-only", "base_4f_strict": "⑤ 4f-base (strict)",
         "step125_1f": "step125 (1f)", "step250_1f": "step250 (1f)",
         "step375_1f": "step375 (1f)", "step500_1f": "step500 (1f)",
         "step125_4f": "step125 (4f)", "step250_4f": "step250 (4f)",
         "step375_4f": "step375 (4f)", "step500_4f": "step500 (4f)"}
rows = []
for n in names:
    p = bat / f"{n}.json"
    if p.exists():
        d = json.loads(p.read_text())
        rows.append((label[n], d))
md = ["# F0 사전 검증 배터리 + 500-step 검증 run — 자동 실행 결과", "",
      "| run | acc | G2 | in_joint5 | wm_follow | parse | belief 존재 | 추론 단어수 |",
      "|---|---|---|---|---|---|---|---|"]
for lab, d in rows:
    md.append(f"| {lab} | {d['acc']} | {d['g2_acc']} | {d['in_joint5']} | "
              f"{d['wm_follow']} | {d['parse_rate']} | {d['belief_present_rate']} | "
              f"{d['mean_reasoning_words']} |")
md += ["", f"- WM top-1 참조선: {rows[0][1]['wm_top1_gt_acc']}, "
           f"candidate-recall oracle: {rows[0][1]['gt_in_top5_rate']}, G2 n={rows[0][1]['g2_n']}",
       "- 게이트: 4f 채택 = 4f-base acc > 0.30 · G2 chance=0.20",
       "- 생성: greedy(do_sample=False) · max_new_tokens 384 · n=500 (v1 heldout 동일 표본)"]
(bat / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
PYEOF
cp "$BAT/RESULTS.md" "$REPO/docs/experiments/2026-07-18_f0_battery_results.md" || true

say "=== PIPELINE DONE ==="
touch "$BAT/PIPELINE_DONE"
