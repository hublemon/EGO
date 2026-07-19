#!/usr/bin/env bash
# b0_ablation_chain.sh — B0 MVP 완료 후 유효 ablation 무인 체인 (데드라인 하드 게이트).
#   A0 ③ belief-swap on B0 ckpt (+④ judge 시도, 비치명) — B0 핵심 성공 지표
#   A1 action-patch DPO: chosen = FAA 자기 trace + GT action 패치 (teacher 투영 제거)
#       → full-trace projected hindsight 기여 분리. 각 arm: train → 생성 acc → ③ swap
#   A2 데이터 ½ (750쌍) 스케일링 ablation (시간 남을 때만)
# 전 단계 멱등 · B0_FAILED 시 ABL_SKIPPED · 완료 ABL_DONE / 실패 ABL_FAILED
set -uo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
[ -f /root/.config/ego/letsur.env ] && source /root/.config/ego/letsur.env
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"; PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
BAT=$EGO_ROOT/runs/f0_battery
MVP=$BAT/b0_mvp
ABL=$BAT/b0_abl; mkdir -p "$ABL"
FAA=$REPO/outputs/step2/f0_final_v2_val_1f/checkpoint-500
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
DL=$(date -d '2026-07-19 05:30' +%s)   # S9 TRL 호환 실패(~1.5h 손실)로 연장 — 원 데드라인 02:45
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/ABL_FAILED"; exit 1; }
left(){ echo $(( (DL - $(date +%s)) / 60 )); }   # 남은 분

say "=== B0 ablation 체인 대기 시작 (deadline 02:45, 남은 $(left)분) ==="

# ── B0 체인 완료 대기 ────────────────────────────────────────────────
while [ ! -f "$BAT/B0_DONE" ] && [ ! -f "$BAT/B0_FAILED" ]; do
  [ "$(left)" -le 60 ] && { say "데드라인 임박 — B0 미완, ablation 스킵"; touch "$BAT/ABL_SKIPPED"; exit 0; }
  sleep 120
done
[ -f "$BAT/B0_FAILED" ] && { say "B0 실패 — ablation 스킵"; touch "$BAT/ABL_SKIPPED"; exit 0; }
B0OUT=$REPO/outputs/step2/b0_mvp
B0CKPT=$(ls -d "$B0OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$B0CKPT" ] || die "B0 체크포인트 없음"
say "B0_DONE 확인 (ckpt=$B0CKPT, 남은 $(left)분)"

# arm 공통: 생성 acc(records 생산) → ③ swap. gen cuda:1 / 부가평가 cuda:0
eval_arm(){ local name=$1 ckpt=$2
  if [ ! -s "$BAT/${name}_gen_1f.json" ]; then
    say "[$name] 생성 acc (500)"
    $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:1 \
      --adapter "$ckpt" --out "$BAT/${name}_gen_1f.json" > "$BAT/${name}_gen_1f.log" 2>&1 \
      || die "[$name] 생성 acc 실패 — $BAT/${name}_gen_1f.log"
  fi
  if [ ! -s "$BAT/swap_${name}.json" ]; then
    say "[$name] ③ belief-swap"
    $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
      --records "$BAT/${name}_gen_1f.records.jsonl" --adapter "$ckpt" \
      --device cuda:0 --out "$BAT/swap_${name}.json" > "$BAT/swap_${name}.log" 2>&1 \
      || die "[$name] swap 실패 — $BAT/swap_${name}.log"
  fi
}

# ── A0: B0 ckpt 사후 검증 (③ 필수, ④ margin 비치명) ─────────────────
if [ ! -s "$BAT/swap_b0_1f.json" ]; then
  say "A0 ③ belief-swap on B0 (핵심 성공 지표: FAA 0.0081 대비)"
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
    --records "$BAT/b0_gen_1f.records.jsonl" --adapter "$B0CKPT" \
    --device cuda:0 --out "$BAT/swap_b0_1f.json" > "$BAT/swap_b0_1f.log" 2>&1 \
    || die "A0 swap 실패 — $BAT/swap_b0_1f.log"
fi
# ④ belief_action_link judge (인터페이스 불일치 가능 — 비치명, API 병렬)
( $PY -m ego.step2_vlm_alignment.judge_reasoning --run_dir "$B0OUT" \
    --judge_model gemini-2.5-pro --per_step 3 > "$ABL/judge_b0.log" 2>&1 || true ) &

# ── A1: action-patch DPO (teacher 투영 기여 분리) ─────────────────────
if [ "$(left)" -lt 100 ]; then
  say "잔여 $(left)분 < 100 — A1/A2 스킵, 요약으로"; SKIP_ARMS=1
else
  SKIP_ARMS=0
  if [ ! -s "$ABL/dpo_actpatch.jsonl" ]; then
    say "A1 데이터: chosen = rejected trace + GT action 패치"
    $PY - "$MVP/b0_dpo.jsonl" "$ABL/dpo_actpatch.jsonl" <<'PYEOF'
import json, re, sys
src, dst = sys.argv[1], sys.argv[2]
ACT = re.compile(r"<action>(.*?)</action>", re.DOTALL)
kept = drop_parse = drop_same = 0
with open(dst, "w", encoding="utf-8") as f:
    for l in open(src, encoding="utf-8"):
        r = json.loads(l)
        mg, mr = ACT.search(r["chosen"]), ACT.search(r["rejected"])
        if not (mg and mr):
            drop_parse += 1; continue
        gt = mg.group(1)
        # chosen = FAA(rejected) 자기 trace 에 action 만 GT 로 교체 — teacher 투영 제거
        patched = r["rejected"][: mr.start(1)] + gt + r["rejected"][mr.end(1):]
        if patched == r["rejected"]:
            drop_same += 1; continue          # GT == rejected action → 신호 없음
        r2 = dict(r); r2["chosen"] = patched
        f.write(json.dumps(r2, ensure_ascii=False) + "\n"); kept += 1
print(f"[actpatch] kept={kept} drop_parse={drop_parse} drop_same_action={drop_same}")
PYEOF
    [ -s "$ABL/dpo_actpatch.jsonl" ] || die "A1 데이터 구성 실패"
  fi
  A1OUT=$REPO/outputs/step2/b0_abl_actpatch
  if ! ls -d "$A1OUT"/checkpoint-* >/dev/null 2>&1; then
    say "A1 DPO 학습 (동일 하이퍼파라미터)"
    python -m accelerate.commands.launch --multi_gpu --num_processes 2 \
      src/ego/step2_vlm_alignment/b0/train_b0_dpo.py \
      --dpo_jsonl "$ABL/dpo_actpatch.jsonl" --faa_adapter "$FAA" --output_dir "$A1OUT" \
      --beta 0.1 --learning_rate 5e-6 --num_train_epochs 1.0 \
      --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
      --save_steps 50 > "$ABL/train_actpatch.log" 2>&1 || die "A1 학습 실패 — $ABL/train_actpatch.log"
  fi
  A1CKPT=$(ls -d "$A1OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$A1CKPT" ] || die "A1 체크포인트 없음"
  # margin 평가(비치명, cuda:0) — gen(cuda:1) 과 병렬
  [ -s "$BAT/abl_actpatch_eval.json" ] || CUDA_VISIBLE_DEVICES=0 $PY -m ego.step2_vlm_alignment.b0.evaluate_b0 \
    --dpo_jsonl "$MVP/b0_dpo_heldout.jsonl" --faa_adapter "$FAA" --b0_adapter "$A1CKPT" \
    --out "$BAT/abl_actpatch_eval.json" > "$BAT/abl_actpatch_eval.log" 2>&1 || true &
  MARGIN_PID=$!
  if [ ! -s "$BAT/abl_actpatch_gen_1f.json" ]; then
    say "A1 생성 acc (500)"
    $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:1 \
      --adapter "$A1CKPT" --out "$BAT/abl_actpatch_gen_1f.json" > "$BAT/abl_actpatch_gen_1f.log" 2>&1 \
      || die "A1 생성 acc 실패"
  fi
  wait $MARGIN_PID 2>/dev/null || true
  if [ ! -s "$BAT/swap_abl_actpatch.json" ]; then
    say "A1 ③ belief-swap"
    $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
      --records "$BAT/abl_actpatch_gen_1f.records.jsonl" --adapter "$A1CKPT" \
      --device cuda:0 --out "$BAT/swap_abl_actpatch.json" > "$BAT/swap_abl_actpatch.log" 2>&1 \
      || die "A1 swap 실패"
  fi
fi

# ── A2: 데이터 ½ 스케일링 (시간 남을 때만) ───────────────────────────
if [ "${SKIP_ARMS:-0}" = 0 ] && [ "$(left)" -ge 100 ]; then
  if [ ! -s "$ABL/dpo_half.jsonl" ]; then
    say "A2 데이터: b0_dpo 앞 절반 (동일 순서)"
    $PY -c "
import json
rows=[json.loads(l) for l in open('$MVP/b0_dpo.jsonl',encoding='utf-8')]
open('$ABL/dpo_half.jsonl','w',encoding='utf-8').write(
  '\n'.join(json.dumps(r,ensure_ascii=False) for r in rows[:len(rows)//2])+'\n')
print('half pairs:',len(rows)//2)"
  fi
  A2OUT=$REPO/outputs/step2/b0_abl_half
  if ! ls -d "$A2OUT"/checkpoint-* >/dev/null 2>&1; then
    say "A2 DPO 학습 (½ 데이터)"
    python -m accelerate.commands.launch --multi_gpu --num_processes 2 \
      src/ego/step2_vlm_alignment/b0/train_b0_dpo.py \
      --dpo_jsonl "$ABL/dpo_half.jsonl" --faa_adapter "$FAA" --output_dir "$A2OUT" \
      --beta 0.1 --learning_rate 5e-6 --num_train_epochs 1.0 \
      --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
      --save_steps 50 > "$ABL/train_half.log" 2>&1 || die "A2 학습 실패 — $ABL/train_half.log"
  fi
  A2CKPT=$(ls -d "$A2OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$A2CKPT" ] || die "A2 체크포인트 없음"
  eval_arm abl_half "$A2CKPT"
else
  say "A2 스킵 (잔여 $(left)분)"
fi
wait 2>/dev/null || true

# ── 요약 ─────────────────────────────────────────────────────────────
$PY - "$BAT" > "$BAT/ABLATION_SUMMARY.txt" <<'PYEOF'
import json, sys
from pathlib import Path
bat = Path(sys.argv[1])
def J(n):
    p = bat / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
print("=== B0 ablation 요약 ===")
rows = [
    ("FAA step500 (baseline)", "step500_1f.json", "swap_step500_1f.json"),
    ("B0 (teacher full-trace)", "b0_gen_1f.json", "swap_b0_1f.json"),
    ("A1 action-patch (no teacher)", "abl_actpatch_gen_1f.json", "swap_abl_actpatch.json"),
    ("A2 half-data", "abl_half_gen_1f.json", "swap_abl_half.json"),
]
for name, gen, swap in rows:
    g = J(gen) if gen else None
    s = J(swap)
    acc = f"acc={g['acc']} G2={g.get('g2_acc')}" if g else "acc=?"
    cs = f"causal_sens={s['causal_sensitivity']} (swap {s['swap_action_change']}/ctrl {s['control_action_change']})" if s else "swap=?"
    print(f"{name:32s} {acc:24s} {cs}")
for n in ["b0_eval.json", "abl_actpatch_eval.json"]:
    d = J(n)
    if d: print(f"[margin] {n}: " + json.dumps({k: d[k] for k in list(d)[:8]}, ensure_ascii=False))
PYEOF
cat "$BAT/ABLATION_SUMMARY.txt"
touch "$BAT/ABL_DONE"; say "=== ablation 체인 DONE (잔여 $(left)분) ==="
