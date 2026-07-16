#!/bin/bash
# watch_run1.sh — Run 1 독립 감시자 (세션/VSCode 종료와 무관하게 동작)
#
# 하는 일:
#   1. 250/500/750/1000/1250 step 도달 시마다 reward·think 진단을
#      runs/grpo_run1_wmonly/diagnostics.log 에 기록
#   2. 학습 종료(정상/비정상) 감지 후 자동으로 held-out 체크포인트 일괄 평가
#      (eval_checkpoints_run1.sh → heldout_eval/step*.json + G1/G2 곡선 요약)
#   3. 모든 결과는 파일로 남으므로 세션이 없어도 사후 확인 가능
#
# 실행: setsid nohup bash watch_run1.sh > /dev/null 2>&1 &
# 확인: tail -f runs/grpo_run1_wmonly/diagnostics.log

set -uo pipefail
cd ~/work/jihun/EGO
source activate.sh

RUN_DIR="runs/grpo_run1_wmonly"
DIAG="$RUN_DIR/diagnostics.log"
RL="$RUN_DIR/reward_log.jsonl"

log() { echo "[$(date '+%F %T')] $*" >> "$DIAG"; }

last_step() {
  [ -f "$RL" ] || { echo 0; return; }
  tail -1 "$RL" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("step",0))' 2>/dev/null || echo 0
}

train_alive() { pgrep -f "train_qwen25vl_grpo_ek100.py" > /dev/null; }

diagnose() {
  local upto=$1
  if [ ! -f "$RL" ]; then
    log "reward_log 없음 — step ${upto} 진단 생략"
    return
  fi
  log "===== STEP ${upto} 진단 ====="
  python - "$upto" <<'EOF' >> "$DIAG" 2>&1
import json, statistics, sys
upto = int(sys.argv[1])
rows = [json.loads(l) for l in open('runs/grpo_run1_wmonly/reward_log.jsonl')]
rows = [r for r in rows if r['step'] <= upto + 5]
if not rows:
    print("no rows"); raise SystemExit
print(f"steps logged: {len(rows)} (last={rows[-1]['step']})")
print(f"{'step':>5} {'P1_wmlik':>9} {'P4_conv':>8} {'gate':>7} {'total':>7} {'ds_filt':>7} {'zstd':>6}")
for r in rows[::max(1, len(rows)//15)]:
    print(f"{r['step']:>5} {r['reward_wm_likelihood_reward']:>9.4f} {r['reward_think_convergence_reward']:>8.4f} "
          f"{r['reward_candidate_gate_reward_think']:>7.4f} {r['reward_total']:>7.4f} "
          f"{str(r.get('ds_frac_groups_filtered')):>7} {str(r.get('frac_reward_zero_std')):>6}")
p1 = [r['reward_wm_likelihood_reward'] for r in rows]
tot = [r['reward_total'] for r in rows]
ds = [r.get('ds_frac_groups_filtered', 0) or 0 for r in rows]
q = max(1, len(rows)//4)
seg = lambda a, i, j: statistics.mean(a[i:j]) if a[i:j] else float('nan')
print(f"\nP1 4분위 궤적:    {seg(p1,0,q):.4f} → {seg(p1,q,2*q):.4f} → {seg(p1,2*q,3*q):.4f} → {seg(p1,3*q,len(p1)):.4f}")
print(f"total 4분위 궤적: {seg(tot,0,q):.4f} → {seg(tot,q,2*q):.4f} → {seg(tot,2*q,3*q):.4f} → {seg(tot,3*q,len(tot)):.4f}")
print(f"P1 최근25 std: {statistics.pstdev(p1[-25:]):.4f} (0 근접 = advantage 고갈 경고)")
print(f"ds_filt: 초반 {seg(ds,0,q):.3f} → 최근 {seg(ds,3*q,len(ds)):.3f} (급증 = 무신호 그룹 증가 경고)")
try:
    ta = [json.loads(l) for l in open('runs/grpo_run1_wmonly/think_analysis.jsonl')]
    ta = [r for r in ta if r['step'] <= upto + 5]
    if ta:
        print(f"\nthink: {'step':>5} {'words':>7} {'diversity':>9} {'mention':>8}")
        for r in ta[::max(1, len(ta)//8)]:
            print(f"       {r['step']:>5} {r.get('think_word_count_mean'):>7} "
                  f"{r.get('generation_diversity'):>9} {r.get('candidate_mention_rate'):>8}")
        w = ta[-1].get('think_word_count_mean', 0)
        d = ta[-1].get('generation_diversity', 1)
        if w and w <= 5: print("⚠️ RED FLAG: think 단어수 ≤5 — format-only 붕괴 신호")
        if d is not None and d == 0: print("⚠️ RED FLAG: 생성 다양성 0 — collapse 신호")
except FileNotFoundError:
    pass
EOF
  log "===== STEP ${upto} 진단 끝 ====="
}

log "watch_run1 시작 (pid $$, 학습 pid: $(pgrep -f train_qwen25vl_grpo_ek100.py | head -1 || echo '?'))"

for MILESTONE in 250 500 750 1000 1250; do
  while true; do
    cur=$(last_step)
    [ "$cur" -ge "$MILESTONE" ] && break
    if ! train_alive; then
      log "⚠️ 학습 프로세스 종료 감지 (step ~${cur}, milestone ${MILESTONE} 미도달)"
      diagnose "$cur"
      break 2
    fi
    sleep 30
  done
  [ "$(last_step)" -ge "$MILESTONE" ] && diagnose "$MILESTONE"
done

# 학습 종료 대기 (1250 진단 후 저장 마무리까지)
while train_alive; do sleep 30; done
log "학습 프로세스 종료. 최종 step: $(last_step)"

# held-out 자동 평가 (체크포인트가 있으면)
if ls "$RUN_DIR"/checkpoint-* >/dev/null 2>&1; then
  log "held-out 체크포인트 일괄 평가 시작 (limit 500)"
  bash eval_checkpoints_run1.sh "$RUN_DIR" 500 >> "$DIAG" 2>&1
  log "held-out 평가 완료 — $RUN_DIR/heldout_eval/step*.json"
else
  log "체크포인트 없음 — held-out 평가 생략"
fi
log "watch_run1 종료"
