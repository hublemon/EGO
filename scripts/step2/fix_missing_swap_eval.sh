#!/usr/bin/env bash
# 누락된 ③ belief-swap 인과민감도 측정을 뒤늦게 채운다.
#
# retro_overnight_gpu1_v2.sh:121 의 eval_belief_swap 호출에 --records 가 빠져 있어
# 이번 라운드의 **목표 지표**(③ > 0.05)가 rc=2 로 측정되지 못했다. 생성 평가는 정상
# 완료돼 있고 개입 평가가 요구하는 records 파일도 이미 만들어져 있으므로, 학습을 다시
# 돌릴 필요 없이 이 스크립트로 채울 수 있다.
#
# 돌고 있는 체인 스크립트는 건드리지 않는다 — bash 는 스크립트를 바이트 오프셋으로
# 읽어들이므로 실행 중 파일을 고치면 이후 실행이 깨진다. 대신 체인이 각 tag 의 records
# 를 내놓는 대로 여기서 이어 붙인다.
#
# GPU 0 을 쓴다 (체인은 GPU 1 에서 계속 돈다).

set -uo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"; PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

BAT=$EGO_ROOT/runs/f0_battery
OUT=$EGO_ROOT/runs/retro_overnight
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
DEV=cuda:0
LOG=$OUT/fix_swap.log
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# tag -> 어댑터 경로 (체인이 쓰는 이름 그대로)
declare -A ADAPTERS=(
  [belief_sum_gt]="$REPO/outputs/step2/retro_belief_sum_gt_1f"
  [belief_sum_wm]="$REPO/outputs/step2/retro_belief_sum_wm_1f"
)

say "===== 누락된 ③ 측정 시작 (GPU 0) ====="

# GPU 0 이 Step-1 격차 측정으로 아직 바쁘면 비켜준다.
while pgrep -f "measure_gap.py" >/dev/null; do sleep 30; done
say "GPU 0 확보"

for TAG in belief_sum_gt belief_sum_wm; do
  REC="$OUT/eval_${TAG}.records.jsonl"

  # 체인이 이 tag 의 생성 평가를 끝낼 때까지 기다린다 (최대 60분).
  waited=0
  while [ ! -s "$REC" ] && [ $waited -lt 3600 ]; do sleep 60; waited=$((waited+60)); done
  if [ ! -s "$REC" ]; then say "SKIP $TAG — records 없음 ($REC)"; continue; fi
  if [ -s "$OUT/swap_${TAG}.json" ]; then say "SKIP $TAG — 이미 측정됨"; continue; fi

  # 어댑터 경로는 반드시 대기 **후에** 정한다. 대기 전에 정하면 아직 학습이 끝나지
  # 않은 tag 의 checkpoint-final 이 없어서 부모 디렉터리로 굳고, PeftModel 이
  # adapter_config.json 을 못 찾아 rc=1 로 죽는다.
  AD="${ADAPTERS[$TAG]}"
  [ -d "$AD/checkpoint-final" ] && AD="$AD/checkpoint-final"

  say "--- ③ 측정: $TAG (records $(wc -l < "$REC")건) ---"
  $PY scripts/step2/eval_belief_swap.py \
    --jsonl "$J1F" --records "$REC" --adapter "$AD" \
    --limit 0 --device $DEV --out "$OUT/swap_${TAG}.json" \
    > "$OUT/swap_${TAG}.log" 2>&1
  rc=$?
  say "  rc=$rc"
  [ $rc -eq 0 ] && $PY -c "
import json
d=json.load(open('$OUT/swap_${TAG}.json'))
cs=d.get('causal_sensitivity', d.get('3', d.get('score')))
print(f'  [$TAG] ③ causal_sensitivity = {cs}  (n={d.get(\"n\")})  기준 > 0.05, 어제 0.006')
" 2>&1 | tee -a "$LOG"
done

say "===== 완료 ====="
