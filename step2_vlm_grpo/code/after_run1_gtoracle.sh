#!/bin/bash
# after_run1_gtoracle.sh — Run 1 held-out 평가가 끝난 직후 GT-oracle 참조선 자동 생성 (Run 3 대비)
#
# 대기 조건: watch_run1.sh 가 diagnostics.log 에 "watch_run1 종료"를 기록할 때까지
# (= Run 1 학습 + held-out 체크포인트 평가 완료 시점, GPU 유휴 보장)
# 실행 내용: Exp.14(runs/grpo_final/checkpoint-1250, GT-primary)를 같은 held-out
# 500샘플에서 평가 → "GT를 알 때의 상한" 참조선. Run 3 §3 결과표의 3번 항목을 미리 확보.
#
# 실행: setsid nohup bash after_run1_gtoracle.sh > /dev/null 2>&1 &
# 결과: runs/grpo_final/heldout_eval/gtoracle_step1250.json + diagnostics.log 에 기록

set -uo pipefail
cd ~/work/jihun/EGO
source activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

DIAG="runs/grpo_run1_wmonly/diagnostics.log"
log() { echo "[$(date '+%F %T')] [gtoracle] $*" >> "$DIAG"; }

# watch_run1 종료 대기 (최대 12h)
for i in $(seq 1 1440); do
  grep -q "watch_run1 종료" "$DIAG" 2>/dev/null && break
  sleep 30
done
if ! grep -q "watch_run1 종료" "$DIAG" 2>/dev/null; then
  log "⚠️ 12h 내 watch_run1 종료 미감지 — GT-oracle 평가 수동 실행 필요"
  exit 1
fi

log "GT-oracle(Exp.14 checkpoint-1250) held-out 평가 시작"
python eval_heldout.py \
  --jsonl data/grpo_dataset/grpo_heldout.jsonl \
  --adapter runs/grpo_final/checkpoint-1250 \
  --limit 500 --batch_size 16 \
  --out runs/grpo_final/heldout_eval/gtoracle_step1250.json >> "$DIAG" 2>&1
log "GT-oracle 평가 완료 → runs/grpo_final/heldout_eval/gtoracle_step1250.json"

# 곡선 figure 도 자동 생성 (/opt/conda python3 — matplotlib 은 eve-cu124/시스템 python 에 없음)
/opt/conda/bin/python3 plot_run1_curves.py >> "$DIAG" 2>&1 || \
  log "plot 실패 — 수동: /opt/conda/bin/python3 plot_run1_curves.py"
log "after_run1_gtoracle 종료"
