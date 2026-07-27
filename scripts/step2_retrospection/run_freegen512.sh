#!/usr/bin/env bash
# freegen 재실행 — max_new_tokens 320 → 512.
#
# 왜: 후보 미제시 레짐에서 sft_r15_c 의 malformed 가 17.8%(89/500)로 theta_ce(2.4%)의 7배다.
#     malformed 레코드는 태그가 하나도 파싱되지 않았고(=출력이 통째로 잘림), 정상 생성분이
#     83.5 단어로 theta_ce(55.1)보다 52% 길다. 둘 다 기본 320 토큰을 썼다.
#     battery 에서는 오히려 sft_r15_c 의 malformed 가 더 낮다(2.8% vs 4.3%) — 후보가 제시되면
#     출력이 짧아지기 때문. 따라서 토큰 예산 초과가 유력하다.
#
# 공정성: **전 조건을 같은 예산으로 다시 돌린다.** 한 arm 만 512 로 올리면 비교가 깨진다.
# 원본 불가침: 별도 run dir(runs/cesft_v2_fp_fg512)에서 실행한다. 기존 320 산출물은 그대로 둔다.
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
R=runs/cesft_v2_fp_fg512
B=/mnt/nvme/migration/jihun/EGO_jihun3
MNT=512
say(){ echo "[FG $(date '+%F %T')] $*"; }

mkdir -p $R/{data,logs,status,markers,eval}
cp -n runs/cesft_v2_fp_c/overrides.json $R/ 2>/dev/null || true      # covered_only — 필수
ln -sf $B/runs/cesft_v2_fp/data/context_val.jsonl $R/data/context_val.jsonl

say "==== freegen @ max_new_tokens=$MNT (전 조건 동일 예산) ===="
export RETRO3_RUNS=$R
for arm in base cand_free theta_ce sft_r15_c; do
  if [ -f "$R/eval/freegen_${arm}_cand_free.json" ]; then say "$arm 이미 완료 — SKIP"; continue; fi
  case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
  say "-- $arm"
  if [ -z "$AD" ]; then
    $PY -m ego.step2_retrospection.eval.freegen --config $CFG --arm "$arm" \
        --eval_n 500 --max_new_tokens $MNT >> $R/logs/stage.log 2>&1
  else
    $PY -m ego.step2_retrospection.eval.freegen --config $CFG --arm "$arm" --adapter "$AD" \
        --eval_n 500 --max_new_tokens $MNT >> $R/logs/stage.log 2>&1
  fi
  say "$arm exit=$?"
done

say "==== 320 vs 512 대비 ===="
$PY - <<'EOF' | tee -a runs/cesft_v2_fp_fg512/logs/compare.log
import json, os
old = {"base":"runs/cesft_v2_fp/eval/freegen_base_cand_free.json",
       "cand_free":"runs/cesft_v2_fp/eval/freegen_cand_free_cand_free.json",
       "theta_ce":"runs/cesft_v2_fp/eval/freegen_theta_ce_cand_free.json",
       "sft_r15_c":"runs/cesft_v2_fp_c/eval/freegen_sft_r15_c_cand_free.json"}
print(f"{'arm':11s} {'malformed 320→512':>22s} {'gt_correct 320→512':>23s} {'in_support 320→512':>23s}")
for a, p in old.items():
    q = f"runs/cesft_v2_fp_fg512/eval/freegen_{a}_cand_free.json"
    if not os.path.exists(q):
        print(f"{a:11s}  (512 산출물 없음)"); continue
    o, n = json.load(open(p)), json.load(open(q))
    print(f"{a:11s} {o['malformed']:9.3f} → {n['malformed']:<9.3f} "
          f"{o['gt_correct']:10.3f} → {n['gt_correct']:<10.3f} "
          f"{o['in_support']:10.3f} → {n['in_support']:<10.3f}")
print("\n판정: malformed 가 크게 줄면 토큰 예산 문제였다는 뜻 — 능력 저하가 아니다.")
EOF
say "FG 완료"
