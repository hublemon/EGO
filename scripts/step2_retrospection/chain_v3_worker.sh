#!/usr/bin/env bash
# 체인 v3 워커 — **서버 A·B 어느 쪽에서든 같은 스크립트를 그대로 실행**한다.
#
# 조정 장치는 공유 파일시스템 위의 원자적 claim 하나뿐이다(mkdir 은 xfs 에서 원자적).
#   runs/chain_v3_claims/<cell>.claim/   ← 먼저 mkdir 에 성공한 서버가 그 셀을 갖는다
# 두 서버가 같은 셀을 동시에 집는 경쟁이 구조적으로 불가능하므로 서로 통신할 필요가 없다.
# 추가로 A 는 앞에서, B 는 뒤에서 훑어 경쟁 자체를 줄인다(ORDER=rev).
#
# 산출물이 이미 있으면 claim 없이 건너뛴다(멱등) — 단일 서버로 돌던 기존 chain_v3 의
# 결과를 그대로 이어받는다. 죽으면 다시 걸면 이어서 간다.
#
# 기동:
#   서버 A:  setsid nohup bash scripts/step2_retrospection/chain_v3_worker.sh          > runs/worker_A.log 2>&1 < /dev/null &
#   서버 B:  setsid nohup env ORDER=rev bash scripts/step2_retrospection/chain_v3_worker.sh > runs/worker_B.log 2>&1 < /dev/null &
#
# 불변 계약: 프레임 8 · RETRO_NEXT_GAP_TEXT · 1인칭 · 출력 형식. 원본 run dir 은 읽기만 한다.
set -uo pipefail
cd /mnt/nvme/migration/jihun/EGO_jihun3
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH=/mnt/nvme/migration/jihun/EGO_jihun3/src
export HF_HOME=/mnt/nvme/cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"      # 시간 계약 — 불변
export FRAME_CACHE_DIR=/mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2/frame_cache
CFG=configs/step2_retrospection/cesft_v2_fp.yaml
ADAPT=outputs/step2_retrospection/cesft_v2_fp
BASE=/mnt/nvme/migration/jihun/EGO_jihun3
RC=runs/cesft_v2_fp_curve
FG=runs/cesft_v2_fp_fg2
LN=runs/cesft_v2_fp_lenient
CL=runs/chain_v3_claims
ME="${WORKER_NAME:-$(hostname)}"
ORDER="${ORDER:-fwd}"
LOG=runs/chain_v3.log
say(){ echo "[W:$ME $(date '+%F %T')] $*" | tee -a $LOG; }

mkdir -p $CL $RC/{eval,logs,data} $FG/{eval,logs,data,status,markers} $LN/{eval,logs,data,status,markers}

# ── 하트비트 — UI 가 서버별 생존을 실측으로 보여주기 위한 것 ──────────────────
( while true; do date +%s > $CL/hb.$ME; sleep 30; done ) &
HB=$!
cleanup(){ kill $HB 2>/dev/null; rm -f $CL/hb.$ME; }
trap cleanup EXIT INT TERM

# ── 공통 준비 (양쪽에서 돌아도 무해) ─────────────────────────────────────────
cp -n runs/cesft_v2_fp_c/overrides.json $FG/ 2>/dev/null || true
cp -n runs/cesft_v2_fp_c/overrides.json $LN/ 2>/dev/null || true
ln -sfn $BASE/runs/cesft_v2_fp/data/context_val.jsonl $FG/data/context_val.jsonl
ln -sfn $BASE/runs/cesft_v2_fp/data/context_val.jsonl $LN/data/context_val.jsonl
ln -sfn $BASE/runs/cesft_v2_fp_c/eval/base.records.jsonl $RC/eval/base.records.jsonl
for f in context_val.jsonl context_train.jsonl train_subset.json; do
  [ -e $RC/data/$f ] || ln -s $BASE/runs/cesft_v2_fp/data/$f $RC/data/$f
done

CARMS=base,pro_s200,pro_final,retro_s100,retro_s200,retro_final,r00_final
CN=250

# ── 셀 정의: name|out|dep|runner ─────────────────────────────────────────────
# dep 이 빈 문자열이 아니면 그 파일이 생길 때까지 이 셀은 집지 않는다.
CELLS=(
  "A_fg_base|$FG/eval/freegen_base_cand_free.json||fg base"
  "A_fg_cand_free|$FG/eval/freegen_cand_free_cand_free.json||fg cand_free"
  "A_fg_theta_ce|$FG/eval/freegen_theta_ce_cand_free.json||fg theta_ce"
  "A_fg_sft_r15_c|$FG/eval/freegen_sft_r15_c_cand_free.json||fg sft_r15_c"
  "B_r00_s100|$RC/eval/r00_s100.json||batt_curve r00_s100 $ADAPT/sft_r00_c/adapter_step100"
  "B_r00_s200|$RC/eval/r00_s200.json||batt_curve r00_s200 $ADAPT/sft_r00_c/adapter_step200"
  "B_r00_final|$RC/eval/r00_final.json||batt_curve r00_final $ADAPT/sft_r00_c/adapter"
  "C_plan|$RC/eval/harden_paired_plan.json|$RC/eval/r00_final.json|cplan"
  "C_b_base|$RC/eval/base.harden_paired.json|$RC/eval/harden_paired_plan.json|belief base"
  "C_b_pro_final|$RC/eval/pro_final.harden_paired.json|$RC/eval/harden_paired_plan.json|belief pro_final $ADAPT/theta_ce/adapter"
  "C_b_retro_final|$RC/eval/retro_final.harden_paired.json|$RC/eval/harden_paired_plan.json|belief retro_final $ADAPT/sft_r15_c/adapter"
  "C_b_retro_s100|$RC/eval/retro_s100.harden_paired.json|$RC/eval/harden_paired_plan.json|belief retro_s100 $ADAPT/sft_r15_c/adapter_step100"
  "C_b_pro_s200|$RC/eval/pro_s200.harden_paired.json|$RC/eval/harden_paired_plan.json|belief pro_s200 $ADAPT/theta_ce/adapter_step200"
  "C_b_retro_s200|$RC/eval/retro_s200.harden_paired.json|$RC/eval/harden_paired_plan.json|belief retro_s200 $ADAPT/sft_r15_c/adapter_step200"
  "C_b_r00_final|$RC/eval/r00_final.harden_paired.json|$RC/eval/harden_paired_plan.json|belief r00_final $ADAPT/sft_r00_c/adapter"
  "D_batt_base|$LN/eval/base.json||batt_len base"
  "D_batt_cand_free|$LN/eval/cand_free.json||batt_len cand_free"
  "D_batt_theta_ce|$LN/eval/theta_ce.json||batt_len theta_ce"
  "D_batt_sft_r15|$LN/eval/sft_r15.json||batt_len sft_r15"
  "D_batt_sft_r15_c|$LN/eval/sft_r15_c.json||batt_len sft_r15_c"
)

# ── 러너 ─────────────────────────────────────────────────────────────────────
fg(){ local arm="$1"; local ad=""
  [ "$arm" != base ] && ad="--adapter $ADAPT/$arm/adapter"
  RETRO3_RUNS=$FG $PY -m ego.step2_retrospection.eval.freegen --config $CFG --arm "$arm" $ad \
      --eval_n 500 --max_new_tokens 512 --lenient --save_text >> $FG/logs/stage.log 2>&1
}
batt_curve(){ local tag="$1" ad="$2"
  [ -d "$ad" ] || { say "MISSING adapter $ad"; return 9; }
  RETRO3_RUNS=$RC $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm "$tag" \
      --adapter "$ad" --eval_n 1000 --covered_only >> $RC/logs/stage.log 2>&1
}
batt_len(){ local arm="$1"; local ad=""
  [ "$arm" != base ] && ad="--adapter $ADAPT/$arm/adapter"
  RETRO3_RUNS=$LN $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm "$arm" $ad \
      --eval_n 1000 --covered_only --lenient --save_text >> $LN/logs/stage.log 2>&1
}
cplan(){
  RETRO3_RUNS=$RC $PY tools/harden_paired.py --stage plan --arms "$CARMS" --n $CN --donor base \
      --config $CFG >> $RC/logs/curve_belief.log 2>&1
}
belief(){ local tag="$1" ad="${2:-}"
  local extra=""; [ -n "$ad" ] && extra="--adapter $ad"
  RETRO3_RUNS=$RC $PY tools/harden_paired.py --stage run --arm "$tag" $extra --curve \
      --config $CFG >> $RC/logs/curve_belief.log 2>&1
}

# ── 메인 루프 ────────────────────────────────────────────────────────────────
say "워커 시작 (order=$ORDER, 셀 ${#CELLS[@]}개)"
idle=0
while :; do
  progressed=0; pending=0
  IDX=($(seq 0 $((${#CELLS[@]}-1))))
  [ "$ORDER" = rev ] && IDX=($(seq $((${#CELLS[@]}-1)) -1 0))
  for i in "${IDX[@]}"; do
    IFS='|' read -r name out dep runner <<< "${CELLS[$i]}"
    [ -s "$out" ] && continue                            # 이미 완료 — claim 불필요
    pending=$((pending+1))
    [ -n "$dep" ] && [ ! -s "$dep" ] && continue          # 선행 미충족 — 다음 기회에
    mkdir "$CL/$name.claim" 2>/dev/null || continue       # 남이 잡았다
    echo "$ME" > "$CL/$name.claim/host"; date +%s > "$CL/$name.claim/started"
    say "▶ $name"
    $runner; rc=$?
    if [ -s "$out" ]; then
      say "✓ $name (rc=$rc)"; progressed=1
    else
      say "✗ $name 산출물 없음 (rc=$rc) — claim 해제, 다른 서버가 재시도 가능"
      rm -rf "$CL/$name.claim"
    fi
  done
  if [ "$pending" = 0 ]; then say "남은 셀 없음 — 종료"; touch runs/chain_v3.DONE; break; fi
  if [ "$progressed" = 0 ]; then
    idle=$((idle+1))
    # 전부 남의 claim 이거나 선행 대기 — 상대 서버가 도는 중이다. 30초 뒤 다시 본다.
    [ $((idle % 20)) = 1 ] && say "대기 (미완 $pending 개, 전부 claim 중이거나 선행 대기)"
    # 60분 넘게 아무것도 못 하면 죽은 claim 을 회수한다
    if [ "$idle" -gt 120 ]; then
      for d in $CL/*.claim; do
        [ -d "$d" ] || continue
        h=$(cat "$d/host" 2>/dev/null); s=$(cat "$d/started" 2>/dev/null || echo 0)
        hb=$(cat "$CL/hb.$h" 2>/dev/null || echo 0); now=$(date +%s)
        if [ $((now - hb)) -gt 1200 ] && [ $((now - s)) -gt 1200 ]; then
          say "죽은 claim 회수: $(basename $d) (host=$h, 하트비트 $((now-hb))초 전)"; rm -rf "$d"
        fi
      done
      idle=0
    fi
    sleep 30
  else
    idle=0
  fi
done
say "워커 종료"
