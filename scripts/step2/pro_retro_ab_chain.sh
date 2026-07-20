#!/usr/bin/env bash
# pro_retro_ab_chain.sh — "credit 을 action 에 국소화하면 달라지는가" 를 두 트랙에서 동시에 검증.
#
# 진단 배경: R1 span 분해가 belief +0.917 vs action +0.007 (131:1) 이었고, F0 의 ③ 인과도
#   0.008 에 머물렀다. 두 현상의 원인이 같다 — **행동으로 정해진 credit 이 trace 전체
#   토큰에 균등 분배된다**. B0-DPO 는 쌍 전체가 다르고, F0-REINFORCE 는
#   loss = −adv · mean_logp(완성부 전체) 다. 어느 쪽도 credit 이 action 에 걸린 적이 없다.
#
#   A (offline/선호): P2-only DPO — 쌍이 action 만 다르다 (P1 제거로 span 확산 차단)
#   B (online/보상):  F0 W-EMA + --credit action — advantage 를 <action> 이후 토큰에만
#
# 두 arm 은 GPU 를 하나씩 잡고 병렬로 돈다. 마커: F0B0_AB_DONE / F0B0_AB_FAILED
set -euo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"; PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
BAT=$EGO_ROOT/runs/f0_battery
PD=$BAT/b0_p12
R1D=$BAT/b0_r1
AD=$BAT/ab_a; mkdir -p "$AD"
FAA=$REPO/outputs/step2/f0_final_v2_val_1f/checkpoint-500
TR1F=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
AOUT=$REPO/outputs/step2/b0_p2only_1f
BOUT=$REPO/outputs/step2/f0_wema_actioncredit_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/F0B0_AB_FAILED"; exit 1; }

# ── P12 종료 대기 (GPU 를 넘겨받는다) ────────────────────────────────────────
if [ ! -f "$BAT/B0_P12_DONE" ] && [ ! -f "$BAT/B0_P12_FAILED" ]; then
  say "=== B0-P12 종료 대기 ==="
  while [ ! -f "$BAT/B0_P12_DONE" ] && [ ! -f "$BAT/B0_P12_FAILED" ]; do sleep 120; done
fi
[ -f "$BAT/B0_P12_FAILED" ] && die "P12 가 실패로 끝났다 — 원인 확인 후 수동 재개"
say "=== A/B 병렬 시작 (A=cuda:0 P2-only DPO · B=cuda:1 F0 span-credit) ==="

# ── ARM A : P2-only DPO (cuda:0) ────────────────────────────────────────────
arm_a(){
  set -e
  if [ ! -s "$AD/pairs_p2only.jsonl" ]; then
    say "[A] P2-only 쌍 생성 (max_p1=0, max_p2=4)"
    $PY -m ego.step2_vlm_alignment.retro.build_pairs_contrastive \
      --samples "$PD/b0_samples_8gen.jsonl" --train_jsonl "$TR1F" \
      --max_p1 0 --max_p2 4 \
      --out_pairs "$AD/pairs_p2only.jsonl" --out_hard "$AD/hard_ids.json" \
      --out_stats "$AD/stats_p2only.json" > "$AD/build.log" 2>&1 || die "[A] 쌍 생성 실패"
    cat "$AD/build.log"
  fi
  N=$(wc -l < "$AD/pairs_p2only.jsonl"); say "[A] $N 쌍"
  [ "$N" -ge 1200 ] || die "[A] 쌍 수 부족($N)"
  if [ ! -f "$AOUT/TRAINING_DONE" ]; then
    say "[A] DPO 학습 (cuda:0)"
    rm -rf "$AOUT"
    CUDA_VISIBLE_DEVICES=0 $PY src/ego/step2_vlm_alignment/retro/train_retro_dpo.py \
      --dpo_jsonl "$AD/pairs_p2only.jsonl" --faa_adapter "$FAA" --output_dir "$AOUT" \
      --max_length 4096 --max_prompt_length 1024 > "$AD/dpo.log" 2>&1 || die "[A] DPO 실패"
    $PY -c "
from safetensors.torch import load_file
a=load_file('$FAA/adapter_model.safetensors'); b=load_file('$AOUT/adapter_model.safetensors')
d=max((a[k]-b[k]).abs().max().item() for k in a if k in b)
print('[guard] max adapter weight diff vs FAA:', d)
raise SystemExit(0 if d>1e-7 else 1)" || die "[A] 무학습"
    touch "$AOUT/TRAINING_DONE"
  fi
  [ -s "$BAT/abA_gen_1f.json" ] || { say "[A] 생성 acc";
    $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:0 --adapter "$AOUT" \
      --out "$BAT/abA_gen_1f.json" > "$BAT/abA_gen_1f.log" 2>&1 || die "[A] eval 실패"; }
  # span margin — P12·R1 과 동일 heldout 쌍으로 비교 가능하게
  [ -s "$BAT/remeasure_abA.json" ] || { say "[A] span margin";
    $PY scripts/step2/remeasure_retro_margin.py --dpo_jsonl "$R1D/b0_r1_dpo_heldout.jsonl" \
      --policies "faa:$FAA,p2only:$AOUT" --device cuda:0 \
      --out "$BAT/remeasure_abA.json" > "$BAT/remeasure_abA.log" 2>&1 || die "[A] margin 실패"; }
  say "[A] 완료"
}

# ── ARM B : F0 W-EMA + action-span credit (cuda:1) ──────────────────────────
arm_b(){
  set -e
  if [ ! -f "$BOUT/TRAINING_DONE" ]; then
    say "[B] smoke (12샘플, credit=action)"
    rm -rf "$BAT/f0smoke_ac"
    $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$BAT/f0smoke_ac" \
      --full_trace --reward wm --credit action --max_new_tokens 384 --batch_gen 2 \
      --max_samples 12 --accum 4 --log_every 6 --save_every 100000 --device cuda:1 \
      > "$BAT/f0smoke_ac.log" 2>&1 || die "[B] smoke 실패 — $BAT/f0smoke_ac.log"
    # action 태그 미검출로 전량 스킵되면 학습이 0 → smoke 로그에 기록이 남는지 확인
    grep -q "\[gr\]" "$BAT/f0smoke_ac.log" || die "[B] smoke 무학습 (action span 미검출 의심)"
    say "[B] 학습 (cuda:1, 5000 샘플)"
    $PY scripts/step2/pro_gr_train.py --train_jsonl "$TR1F" --output_dir "$BOUT" \
      --full_trace --reward wm --credit action --max_new_tokens 384 --batch_gen 4 \
      --max_samples 5000 --accum 16 --save_every 1250 --device cuda:1 \
      > "$BAT/train_ac.log" 2>&1 || die "[B] 학습 실패 — $BAT/train_ac.log"
    ls "$BOUT"/checkpoint-final >/dev/null 2>&1 || die "[B] 최종 체크포인트 없음"
    touch "$BOUT/TRAINING_DONE"
  fi
  [ -s "$BAT/abB_gen_1f.json" ] || { say "[B] 생성 acc";
    $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:1 \
      --adapter "$BOUT/checkpoint-final" --out "$BAT/abB_gen_1f.json" \
      > "$BAT/abB_gen_1f.log" 2>&1 || die "[B] eval 실패"; }
  [ -s "$BAT/swap_abB.json" ] || { say "[B] ③ belief swap (이 arm 의 목적 지표)";
    $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
      --records "$BAT/abB_gen_1f.records.jsonl" --adapter "$BOUT/checkpoint-final" \
      --device cuda:1 --out "$BAT/swap_abB.json" > "$BAT/swap_abB.log" 2>&1 || die "[B] swap 실패"; }
  say "[B] 완료"
}

arm_a > "$BAT/ab_arm_a.log" 2>&1 & PA=$!
arm_b > "$BAT/ab_arm_b.log" 2>&1 & PB=$!
FAIL=0
wait $PA || FAIL=1
wait $PB || FAIL=1
[ "$FAIL" = 0 ] || die "arm 실패 — $BAT/ab_arm_{a,b}.log"

# ── 판정 ────────────────────────────────────────────────────────────────────
$PY - "$BAT" "$AD" > "$BAT/F0B0_AB_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat, ad = Path(sys.argv[1]), Path(sys.argv[2])
def J(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None
a_gen, a_rem = J(bat/"abA_gen_1f.json"), J(bat/"remeasure_abA.json")
b_gen, b_sw  = J(bat/"abB_gen_1f.json"), J(bat/"swap_abB.json")
p12_gen, p12_rem = J(bat/"b0p12_gen_1f.json"), J(bat/"remeasure_b0p12.json")
def span(rem, name):
    s = ((rem or {}).get("policies", {}).get(name, {}).get("improvement_vs_ref", {}) or {}).get("span", {})
    return s.get("task_belief"), s.get("action")
print("# A/B — credit 국소화 검증 결과 (자동 생성)\n")
print("## ARM A · P2-only DPO (offline 선호)\n")
print("| 지표 | R1 | P12 혼합 | **A (P2-only)** |")
print("|---|---|---|---|")
print(f"| 생성 acc | 0.238 | {(p12_gen or {}).get('acc')} | {(a_gen or {}).get('acc')} |")
b1, a1 = span(p12_rem, "b0p12"); b2, a2 = span(a_rem, "p2only")
print(f"| belief-span 개선 | +0.9174 | {b1} | {b2} |")
print(f"| action-span 개선 | +0.0069 | {a1} | {a2} |")
def ratio(b, a):
    return f"{b/a:.1f} : 1" if (b and a and a > 0) else "—"
print(f"| **belief : action** | 131 : 1 | {ratio(b1, a1)} | **{ratio(b2, a2)}** |")
st = J(ad/"stats_p2only.json")
if st: print(f"\n쌍: P2 {st['p2_pairs']} · 최종 {st['final_pairs']} (P1 0)")
print("\n## ARM B · F0 W-EMA + action-span credit (online 보상)\n")
print("| 지표 | base | W-EMA (credit=all) | **B (credit=action)** |")
print("|---|---|---|---|")
print(f"| acc | 0.242 | 0.280 | {(b_gen or {}).get('acc')} |")
print(f"| G2 | 0.3089 | 0.3821 | {(b_gen or {}).get('g2_acc')} |")
print(f"| wm_follow | 0.328 | 0.350 | {(b_gen or {}).get('wm_follow')} |")
print(f"| ③ 인과 | 0.016 | 0.0081 | {(b_sw or {}).get('causal_sensitivity')} |")
print("""
## 읽는 법
- **A** 는 'DPO 쌍에서 action 만 다르게 하면 credit 이 action 으로 가는가'. belief:action 비율이
  131:1 에서 한 자릿수로 떨어지면 가설 입증 — acc 절대값보다 이 비율이 본 arm 의 목적 지표다.
- **B** 는 'REINFORCE 의 advantage 를 action 토큰에만 걸면 인과(③)가 생기는가'.
  ③ > 0.03 이면 F0 트랙에서 인과가 처음으로 만들어진 것. acc 는 유지(≥0.26)만 확인한다.
- 둘 다 무반응이면 credit 국소화 가설이 기각되고, 남는 후보는 belief 를 직접 조작하는
  P3(반사실 쌍) 뿐이다.
""")
PYEOF
cat "$BAT/F0B0_AB_RESULTS.md"
touch "$BAT/F0B0_AB_DONE"; say "=== A/B 체인 DONE ==="
