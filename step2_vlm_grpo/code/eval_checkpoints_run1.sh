#!/bin/bash
# Run 1 종료 후: 전 체크포인트 held-out 평가 → G1(곡선 상승)/G2(disagreement 구간) 판정 데이터.
# base 모델(step 0 참조점) 평가 포함. GPU 1장, 체크포인트당 held-out 500샘플 기준 수십 분.
set -euo pipefail
cd ~/work/jihun/EGO
source activate.sh

RUN_DIR="${1:-runs/grpo_run1_wmonly}"
LIMIT="${2:-500}"
EVAL_DIR="$RUN_DIR/heldout_eval"
mkdir -p "$EVAL_DIR"

if [ ! -f "$EVAL_DIR/step0.json" ]; then
  echo "=== base model (step 0) ==="
  python eval_heldout.py --limit "$LIMIT" --out "$EVAL_DIR/step0.json"
fi

for ckpt in $(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  step=$(basename "$ckpt" | cut -d- -f2)
  out="$EVAL_DIR/step${step}.json"
  if [ -f "$out" ]; then
    echo "skip step $step (exists)"; continue
  fi
  echo "=== checkpoint step $step ==="
  python eval_heldout.py --adapter "$ckpt" --limit "$LIMIT" --out "$out"
done

echo "=== G1/G2 곡선 요약 ==="
python - <<'EOF'
import json, glob, re
rows = []
for p in sorted(glob.glob("runs/grpo_run1_wmonly/heldout_eval/step*.json"),
                key=lambda x: int(re.search(r"step(\d+)", x).group(1))):
    s = json.load(open(p))
    rows.append((int(re.search(r"step(\d+)", p).group(1)), s))
print(f"{'step':>6} {'action_acc(fz)':>14} {'g2_acc':>7} {'g2_n':>5} {'escape':>7} {'wm_follow':>9} {'think_w':>8}")
for step, s in rows:
    print(f"{step:>6} {s.get('gt_action_acc_fuzzy'):>14} {s.get('g2_vlm_acc'):>7} "
          f"{s.get('g2_n'):>5} {s.get('candidate_escape_rate'):>7} "
          f"{s.get('wm_follow_rate'):>9} {s.get('think_words_mean'):>8}")
print("\nWM top-1 참조선 (sample-level GT acc):", rows[0][1].get("wm_top1_gt_action_acc") if rows else "?")
print("G2 chance = 0.20")
EOF
