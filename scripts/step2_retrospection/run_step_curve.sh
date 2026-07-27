#!/usr/bin/env bash
# Figure 1 학습 스텝-정확도 곡선용 체크포인트 평가.
#   목적: 저장된 중간 adapter 를 **동일 평가셋·동일 분모**(Shared WM-Covered Set, n=1000,
#         covered_only)로 재평가해 스텝별 Within-Boundary Accuracy 를 얻는다.
#   probe_acc(n=32)는 분모가 다르고 CI 가 ±15pp 라 논문 곡선으로 쓸 수 없다 — 그래서 이 재평가가 필요하다.
#
#   x축은 누적 optimizer step. Stage1(Prospection/Answer-Only) 0~523,
#   Stage2(Retrospection) 는 523 + s 로 이어 붙인다(θ_CE final 에서 warm-start 하므로).
#
# 멱등: 산출 JSON 이 있으면 건너뛴다. 죽으면 같은 명령을 다시 걸면 이어서 간다.
set -uo pipefail
cd /mnt/nvme/migration/jihun/EGO_jihun3
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH=/mnt/nvme/migration/jihun/EGO_jihun3/src
export HF_HOME=/mnt/nvme/cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO_NEXT_GAP_TEXT="after the current action ends"
export FRAME_CACHE_DIR=/mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2/frame_cache
CFG=configs/step2_retrospection/cesft_v2_fp.yaml
ADAPT=outputs/step2_retrospection/cesft_v2_fp
RC=runs/cesft_v2_fp_curve
mkdir -p $RC/eval $RC/logs $RC/data
# context_val 은 fp 코호트와 동일해야 한다 (평가셋 고정)
for f in context_val.jsonl context_train.jsonl train_subset.json; do
  [ -e $RC/data/$f ] || ln -s /mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2_fp/data/$f $RC/data/$f
done
export RETRO3_RUNS=$RC
say(){ echo "[CURVE $(date '+%F %T')] $*" | tee -a $RC/logs/curve.log; }

cell(){ # cell <arm_tag> <adapter_path>
  local tag="$1" ad="$2"
  if [ -s "$RC/eval/${tag}.json" ]; then say "skip $tag (완료)"; return 0; fi
  if [ ! -d "$ad" ]; then say "MISSING adapter $ad — 건너뜀"; return 0; fi
  say "start $tag"
  $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm "$tag" \
      --adapter "$ad" --eval_n 1000 --covered_only >> $RC/logs/stage.log 2>&1
  say "done $tag exit=$?"
}

say "==== step curve 시작 (13 셀 예상 ~2h) ===="
for s in 100 200 300 400 500; do cell "pro_s${s}"  "$ADAPT/theta_ce/adapter_step${s}"; done
cell "pro_final" "$ADAPT/theta_ce/adapter"
for s in 100 200 300 400 500; do cell "ans_s${s}"  "$ADAPT/cand_free/adapter_step${s}"; done
cell "ans_final" "$ADAPT/cand_free/adapter"
for s in 100 200 300; do cell "retro_s${s}" "$ADAPT/sft_r15_c/adapter_step${s}"; done
cell "retro_final" "$ADAPT/sft_r15_c/adapter"
say "==== 완료 ===="
