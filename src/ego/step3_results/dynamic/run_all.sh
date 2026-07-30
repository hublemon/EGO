#!/usr/bin/env bash
# Closed-Loop Dynamic Planning — 3 arm 순차 실행 + 채점. tmux 세션 dynplan 에서 돌린다.
# 스텝 단위 재개가 되므로 중단 후 같은 명령을 다시 실행하면 이어서 진행한다.
set -u
cd /home/hogun/Project/EGO
PY=~/ml_env/bin/python
export PYTHONPATH=src
LOG=/tmp/claude-1002/-home-hogun/754f0145-ce4d-4e63-9ea1-a05977cb6dac/scratchpad
BS=${BS:-16}
say() { echo "[$(date +%F\ %H:%M:%S)] $*"; }

for ARM in ego_closed ego_nobelief oracle_gt_hist; do
  say "=== arm=$ARM (batch=$BS) 시작"
  $PY -u -m ego.step3_results.dynamic.run_closed_loop \
      --arm "$ARM" --batch-size "$BS" \
      --episodes runs/dynamic_v1/episodes.json \
      --out-dir runs/dynamic_v1/preds > "$LOG/run_$ARM.log" 2>&1
  say "=== arm=$ARM 종료 (rc=$?)"
done

say "채점"
$PY -u -m ego.step3_results.dynamic.evaluate \
    --pred-dir runs/dynamic_v1/preds --out-dir runs/dynamic_v1/metrics \
    > "$LOG/evaluate.log" 2>&1
tail -40 "$LOG/evaluate.log"
say "ALL DONE"
