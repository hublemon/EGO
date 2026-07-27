#!/usr/bin/env bash
# defense plan v2 순서 1·2·8 — Plan B 종료 후 이어서 무인 실행.
#   docs/paper/2026-07-27_aaai_reviewer_defense_plan_v2_handoff.md
#
# A. rho=0 대조군      (main.tex L205 약속)          train 17m + battery 10m + harden 10m
# B. K ablation        (main.tex L289 tab:kablation) 4 x ~9m
# C. 축소 Tier 1       (C3/C4/C6, 4 arm)             12 x 7~10m
#
# 시간 최적화 (v2 §4·§6 반영):
#   · noimage 계열은 decord 에 진입하지 않는다 — blank 프레임 합성이라 프레임 추출이 0이다.
#     10분짜리가 ~7분이 되고 OOM 원인(리더 상주)이 구조적으로 사라진다.
#   · othervideo 는 원 표본 프레임을 그대로 써 캐시 히트 100%. othervideo_image(C8)는
#     캐시 무력화로 40~60분이 되므로 v2 판정대로 제외한다.
#   · C5(shuffled)·C7(reversed)·C9~C12(dose-response) 제외 — 판정을 바꾸지 않는데 비용의 40%.
#   · K=3/5 는 후보가 짧아 K=10 보다 빠르다.
#
# OOM 방어 (v2 §6.1):
#   · 셀마다 별도 프로세스 (GPU 메모리 반납)
#   · perturb_eval.py 는 finally 에서 vlm.close_readers()
#   · PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
# 멱등: 마커/산출물이 있으면 건너뛴다. 죽으면 같은 명령을 다시 걸면 이어서 간다.
set -uo pipefail
cd /mnt/nvme/migration/jihun/EGO_jihun3
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH=/mnt/nvme/migration/jihun/EGO_jihun3/src
export HF_HOME=/mnt/nvme/cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"      # 시간 계약 — 불변
export FRAME_CACHE_DIR=/mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2/frame_cache
export CKPT_KEEP_STEP_ADAPTERS=1
CFG=configs/step2_retrospection/cesft_v2_fp.yaml
ADAPT=outputs/step2_retrospection/cesft_v2_fp
QD=runs/ablation_v2                       # 큐 진행 기록
mkdir -p $QD
TL=$QD/timeline.jsonl

say(){ echo "[AB $(date '+%F %T')] $*"; }
ev(){ # ev <item_id> <event>   — 실측 타임라인 (아티팩트 UI 의 잔여시간 근거)
  $PY -c "
import json,time,sys
open('$TL','a').write(json.dumps({'item':sys.argv[1],'event':sys.argv[2],'ts':time.time()})+'\n')" "$1" "$2"
  $PY tools/ablation_progress.py >/dev/null 2>&1 || true
}

# ── 0. Plan B 종료 대기 (최대 6시간) ─────────────────────────────────────────
ev queue start
for i in $(seq 1 720); do
  pgrep -f run_planB.sh >/dev/null || break
  [ "$i" = 1 ] && { say "Plan B 진행 중 — GPU 반납 대기"; ev wait_planB start; }
  sleep 30
done
pgrep -f run_planB.sh >/dev/null && { say "!! Plan B 가 6시간 넘게 안 끝남 — 중단"; ev queue abort; exit 5; }
ev wait_planB done
say "GPU 확보 (Plan B 종료 확인)"

# ══ A. rho=0 대조군 ═══════════════════════════════════════════════════════════
R0=runs/cesft_v2_fp_r00
if [ ! -f "$R0/eval/sft_r00_c.harden_s3.json" ]; then
  say "==== A. rho=0 대조군 ===="
  mkdir -p $R0/{data,logs,status,markers,eval,probe}
  cp -n runs/cesft_v2_fp_c/overrides.json $R0/ 2>/dev/null || true   # ← covered_only. 빠뜨리면 표본 오염
  cp -n runs/cesft_v2_fp_c/probe/probe_set.json $R0/probe/ 2>/dev/null || true
  B=/mnt/nvme/migration/jihun/EGO_jihun3
  ln -sf $B/runs/cesft_v2_fp_c/data/chosen_train.jsonl $R0/data/chosen_train.jsonl
  for f in context_train.jsonl context_val.jsonl train_subset.json; do
    ln -sf $B/runs/cesft_v2_fp/data/$f $R0/data/$f; done
  for a in base cand_free theta_ce sft_r15; do
    ln -sf $B/runs/cesft_v2_fp/eval/$a.records.jsonl $R0/eval/$a.records.jsonl; done
  ln -sf $B/runs/cesft_v2_fp_c/eval/sft_r15_c.records.jsonl $R0/eval/sft_r15_c.records.jsonl

  export RETRO3_RUNS=$R0
  ev A_train start
  $PY -m ego.step2_retrospection.train.sft_v2 --config $CFG --run_name sft_r00_c \
      --init_adapter $ADAPT/theta_ce/adapter --ce_replay_rho 0.0 --ce_tau 1.0 \
      --epochs 1 --seed 42 --ckpt_every 50 --resume auto --probe_every 100 >> $R0/logs/stage.log 2>&1
  say "A.train exit=$?"; ev A_train done
  if [ -f "$ADAPT/sft_r00_c/adapter/adapter_model.safetensors" ]; then
    ev A_battery start
    $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm sft_r00_c \
        --adapter $ADAPT/sft_r00_c/adapter --eval_n 1000 >> $R0/logs/stage.log 2>&1
    say "A.battery exit=$?"; ev A_battery done
    ev A_harden start
    $PY -m ego.step2_retrospection.eval.harden_s3 --config $CFG --arm sft_r00_c \
        --adapter $ADAPT/sft_r00_c/adapter --n 400 >> $R0/logs/stage.log 2>&1
    say "A.harden exit=$?"; ev A_harden done
    $PY tools/paired_boot.py --run $R0 --arm_a sft_r00_c --gate G-ACC1 \
        --out $R0/eval/paired_G-ACC1_sft_r00_c.json >> $R0/logs/stage.log 2>&1
    $PY tools/paired_boot.py --run $R0 --arm_a sft_r00_c --arm_b theta_ce --gate G-NH \
        --out $R0/eval/paired_G-NH_sft_r00_c_vs_theta_ce.json >> $R0/logs/stage.log 2>&1
  else
    say "!! A: 어댑터 없음 — 이후 단계 건너뜀"
  fi
else
  say "A. rho=0 이미 완료 — SKIP"
fi

# ══ B. K ablation ═════════════════════════════════════════════════════════════
RK=runs/cesft_v2_fp_k
mkdir -p $RK/{data,logs,status,markers,eval}
cp -n runs/cesft_v2_fp_c/overrides.json $RK/ 2>/dev/null || true
B=/mnt/nvme/migration/jihun/EGO_jihun3
for f in context_val.jsonl; do ln -sf $B/runs/cesft_v2_fp/data/$f $RK/data/$f; done
export RETRO3_RUNS=$RK
say "==== B. K ablation ===="
$PY tools/ablation_progress.py --coverage_only > $RK/eval/coverage_at_k.json 2>/dev/null || true
for K in 5 3; do
  for arm in theta_ce sft_r15_c; do
    if [ -f "$RK/eval/${arm}_k${K}.json" ]; then say "B.${arm}_k${K} 이미 완료 — SKIP"; continue; fi
    ev "B_${arm}_k${K}" start
    $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm ${arm}_k${K} \
        --adapter $ADAPT/$arm/adapter --eval_n 1000 --top_k $K >> $RK/logs/stage.log 2>&1
    say "B.${arm}_k${K} exit=$?"; ev "B_${arm}_k${K}" done
  done
done

# ══ C. 축소 Tier 1 (C3 noimage / C4 nohist_noimage / C6 othervideo) ══════════
RT=runs/cesft_v2_fp_c                     # 4 arm 의 battery records 가 모두 있는 유일한 루트
export RETRO3_RUNS=$RT
# nohist(C2) 산출물을 한 루트로 모아 집계 가능하게 (기존 실행분 재사용 — 재실행 없음)
for a in base cand_free theta_ce; do
  ln -sf $B/runs/cesft_v2_fp/eval/${a}_nohist.records.jsonl $RT/eval/${a}_nohist.records.jsonl 2>/dev/null || true
done
say "==== C. 축소 Tier 1 ===="
for mode in noimage nohist_noimage othervideo; do
  for arm in base cand_free theta_ce sft_r15_c; do
    if [ -f "$RT/eval/perturb_verdict_${arm}_${mode}.json" ]; then
      say "C.${arm}/${mode} 이미 완료 — SKIP"; continue; fi
    case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
    ev "C_${arm}_${mode}" start
    if [ -z "$AD" ]; then
      $PY tools/oom_opt/perturb_eval.py --config $CFG --arm "$arm" --mode "$mode" \
          --eval_n 1000 >> $RT/logs/perturb.log 2>&1
    else
      $PY tools/oom_opt/perturb_eval.py --config $CFG --arm "$arm" --adapter "$AD" \
          --mode "$mode" --eval_n 1000 >> $RT/logs/perturb.log 2>&1
    fi
    say "C.${arm}/${mode} exit=$?"; ev "C_${arm}_${mode}" done
  done
done

ev queue done
say "==== 전체 완료 ===="
$PY tools/ablation_progress.py 2>&1 | tail -20
