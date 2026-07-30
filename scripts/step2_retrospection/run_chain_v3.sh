#!/usr/bin/env bash
# 무인 체인 v3 — 진행 중인 run_step_curve.sh 가 끝난 뒤 이어서 4단계를 순차 실행한다.
#
#  A) freegen 재실행 (파싱 규칙 수정본)
#     원인 확정: cand-free malformed 17.8%(sft_r15_c) vs 2.4%(theta_ce) 의 실체는
#     **</action> 누락**이다. 실패 16건 전수 확인 → reasoning·task_belief 는 열림/닫힘 16/16
#     정상, <action> 만 열림 16 / 닫힘 5. 토큰 예산 문제가 아니다: 320→512 로 올려도 네 조건
#     **전 지표가 소수점까지 동일**했고 실패 표본 집합도 80건 그대로 일치(교집합 80/80),
#     정상 생성분 reasoning 최대 95단어. 복원된 action 11건은 GT 적중 36%·경계 내 18%.
#     → vlm.parse_trace(lenient=True) 로 마지막 태그의 EOS 종료를 허용. 전 조건 동일 적용.
#     원문(--save_text)을 저장해 이후 규칙 변경은 재실행 없이 재채점한다.
#
#  B) rho=0 정확도 곡선 셀 — Figure 1 왼쪽 패널의 점선(리플레이 없음) 3점.
#
#  C) 스텝별 belief 인과 곡선 — Figure 1 오른쪽 패널.
#     harden_paired --curve 로 own/swap_b_shared/para 3변형만 채점(7변형 대비 arm 당 54→~23분).
#     **새 plan** 이므로 Plan B 절대값과 직접 비교하지 않는다(공통셋이 다르다). 곡선 내부 비교만 유효.
#
#  D) 파싱 규칙 robustness — battery 를 lenient 로 5 arm 재실행(별도 run dir).
#     헤드라인 표를 대체하지 않는다. "규칙을 바꾸면 결론이 뒤집히는가"에만 답한다.
#
# 불변 계약: 프레임 8 · RETRO_NEXT_GAP_TEXT · 1인칭 · 출력 형식. 원본 run dir 은 건드리지 않는다.
# 멱등: 각 셀 산출물이 있으면 건너뛴다. 죽으면 같은 스크립트를 다시 걸면 이어서 간다.
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
B=/mnt/nvme/migration/jihun/EGO_jihun3
RC=runs/cesft_v2_fp_curve            # 곡선(정확도 + belief) — step_curve 와 같은 dir
FG=runs/cesft_v2_fp_fg2              # freegen 수정본
LN=runs/cesft_v2_fp_lenient          # battery robustness
LOG=runs/chain_v3.log
say(){ echo "[V3 $(date '+%F %T')] $*" | tee -a $LOG; }

mkdir -p $RC/{eval,logs,data} $FG/{eval,logs,data,status,markers} $LN/{eval,logs,data,status,markers}

# ── 0) 앞선 step_curve 종료 대기 (최대 3시간) ────────────────────────────────
for i in $(seq 1 360); do
  pgrep -f run_step_curve.sh >/dev/null || break
  [ "$i" = 1 ] && say "step_curve 진행 중 — GPU 반납 대기"
  sleep 30
done
pgrep -f run_step_curve.sh >/dev/null && { say "!! step_curve 가 3시간 넘게 안 끝남 — 중단"; exit 5; }
say "GPU 확보 — 체인 시작"

# ══════════════════════════════════════════════════════════════════════════════
# A) freegen 재실행 (lenient, 전 조건 동일 예산 512)
# ══════════════════════════════════════════════════════════════════════════════
cp -n runs/cesft_v2_fp_c/overrides.json $FG/ 2>/dev/null || true    # covered_only — 필수
ln -sfn $B/runs/cesft_v2_fp/data/context_val.jsonl $FG/data/context_val.jsonl
say "==== A) freegen lenient (4 조건) ===="
export RETRO3_RUNS=$FG
for arm in base cand_free theta_ce sft_r15_c; do
  if [ -s "$FG/eval/freegen_${arm}_cand_free.json" ]; then say "A:$arm SKIP"; continue; fi
  case "$arm" in base) AD="";; *) AD="--adapter $ADAPT/$arm/adapter";; esac
  say "A: $arm 시작"
  $PY -m ego.step2_retrospection.eval.freegen --config $CFG --arm "$arm" $AD \
      --eval_n 500 --max_new_tokens 512 --lenient --save_text >> $FG/logs/stage.log 2>&1
  say "A: $arm exit=$?"
done
$PY - <<'EOF' 2>&1 | tee -a runs/cesft_v2_fp_fg2/logs/compare.log | tee -a runs/chain_v3.log
import json, os
old = {"base":"runs/cesft_v2_fp/eval/freegen_base_cand_free.json",
       "cand_free":"runs/cesft_v2_fp/eval/freegen_cand_free_cand_free.json",
       "theta_ce":"runs/cesft_v2_fp/eval/freegen_theta_ce_cand_free.json",
       "sft_r15_c":"runs/cesft_v2_fp_c/eval/freegen_sft_r15_c_cand_free.json"}
print(f"\n{'조건':11s} {'malformed 기존→수정':>24s} {'구제율':>7s} "
      f"{'gt_correct':>20s} {'in_support':>20s}")
for a, p in old.items():
    q = f"runs/cesft_v2_fp_fg2/eval/freegen_{a}_cand_free.json"
    if not os.path.exists(q):
        print(f"{a:11s}  (수정본 없음)"); continue
    o, n = json.load(open(p)), json.load(open(q))
    print(f"{a:11s} {100*o['malformed']:10.1f} → {100*n['malformed']:<10.1f} "
          f"{100*n.get('recovered',0):6.1f}% "
          f"{100*o['gt_correct']:8.1f} → {100*n['gt_correct']:<8.1f} "
          f"{100*o['in_support']:8.1f} → {100*n['in_support']:<8.1f}")
EOF

# ══════════════════════════════════════════════════════════════════════════════
# B) rho=0 정확도 곡선 셀
# ══════════════════════════════════════════════════════════════════════════════
export RETRO3_RUNS=$RC
cell(){ # cell <tag> <adapter>
  local tag="$1" ad="$2"
  [ -s "$RC/eval/${tag}.json" ] && { say "skip $tag"; return 0; }
  [ -d "$ad" ] || { say "MISSING adapter $ad — 건너뜀"; return 0; }
  say "cell $tag 시작"
  $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm "$tag" \
      --adapter "$ad" --eval_n 1000 --covered_only >> $RC/logs/stage.log 2>&1
  say "cell $tag exit=$?"
}
say "==== B) rho=0 정확도 3점 ===="
cell "r00_s100"  "$ADAPT/sft_r00_c/adapter_step100"
cell "r00_s200"  "$ADAPT/sft_r00_c/adapter_step200"
cell "r00_final" "$ADAPT/sft_r00_c/adapter"

# ══════════════════════════════════════════════════════════════════════════════
# C) 스텝별 belief 인과 곡선 (harden_paired --curve)
# ══════════════════════════════════════════════════════════════════════════════
say "==== C) belief 인과 곡선 ===="
# base 는 곡선의 0점 기준 — 기존 fp_c 의 battery 레코드를 그대로 쓴다(같은 covered set).
ln -sfn $B/runs/cesft_v2_fp_c/eval/base.records.jsonl $RC/eval/base.records.jsonl
CARMS=base,pro_s200,pro_final,retro_s100,retro_s200,retro_final,r00_final
CN=250
if [ -s "$RC/eval/harden_paired_plan.json" ]; then
  say "C: plan 존재 — SKIP"
else
  say "C: plan (arms=$CARMS, n=$CN, donor=base)"
  $PY tools/harden_paired.py --stage plan --arms "$CARMS" --n $CN --donor base \
      --config $CFG >> $RC/logs/curve_belief.log 2>&1
  say "C: plan exit=$?"
  [ -s "$RC/eval/harden_paired_plan.json" ] || { say "!! C plan 실패 — C 건너뜀"; CARMS=""; }
fi
bcell(){ # bcell <tag> <adapter|"">
  local tag="$1" ad="$2"
  [ -s "$RC/eval/$tag.harden_paired.json" ] && { say "skip belief:$tag"; return 0; }
  say "belief:$tag 시작"
  if [ -z "$ad" ]; then
    $PY tools/harden_paired.py --stage run --arm "$tag" --curve --config $CFG \
        >> $RC/logs/curve_belief.log 2>&1
  else
    $PY tools/harden_paired.py --stage run --arm "$tag" --adapter "$ad" --curve --config $CFG \
        >> $RC/logs/curve_belief.log 2>&1
  fi
  say "belief:$tag exit=$?"
}
if [ -n "$CARMS" ]; then
  # 중요도 순 — 중간에 끊겨도 핵심 주장이 남도록 base·양 끝점을 먼저 채운다.
  bcell base         ""
  bcell pro_final    "$ADAPT/theta_ce/adapter"
  bcell retro_final  "$ADAPT/sft_r15_c/adapter"
  bcell retro_s100   "$ADAPT/sft_r15_c/adapter_step100"
  bcell pro_s200     "$ADAPT/theta_ce/adapter_step200"
  bcell retro_s200   "$ADAPT/sft_r15_c/adapter_step200"
  bcell r00_final    "$ADAPT/sft_r00_c/adapter"
fi

# ══════════════════════════════════════════════════════════════════════════════
# D) 파싱 규칙 robustness — battery lenient (별도 dir, 헤드라인 대체 아님)
# ══════════════════════════════════════════════════════════════════════════════
say "==== D) battery lenient robustness ===="
cp -n runs/cesft_v2_fp_c/overrides.json $LN/ 2>/dev/null || true
ln -sfn $B/runs/cesft_v2_fp/data/context_val.jsonl $LN/data/context_val.jsonl
export RETRO3_RUNS=$LN
for arm in base cand_free theta_ce sft_r15 sft_r15_c; do
  [ -s "$LN/eval/${arm}.json" ] && { say "D:$arm SKIP"; continue; }
  case "$arm" in base) AD="";; *) AD="--adapter $ADAPT/$arm/adapter";; esac
  say "D: $arm 시작"
  $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm "$arm" $AD \
      --eval_n 1000 --covered_only --lenient --save_text >> $LN/logs/stage.log 2>&1
  say "D: $arm exit=$?"
done

say "==== 체인 v3 완료 ===="
touch runs/chain_v3.DONE
