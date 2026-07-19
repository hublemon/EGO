#!/usr/bin/env bash
# b0_auto_chain.sh — 배터리 ③④ → B0 MVP(1,500 프롬프트) 전 과정 무인 체인.
#   S1 ③ belief-swap (base + step500)          [GPU 2장 병렬, ~40분]
#   S2 ④ belief_action_link judge 재채점        [API, S1 과 병렬]
#   S3 MVP 서브샘플 1,500 (seed 0)
#   S4 FAA 롤아웃 train (2-way GPU 샤딩) → S5 병합
#   S6 teacher 빌드 (2-way 샤딩) → S7 재검증 게이트
#   S8 heldout 500 pairs (margin 평가용)
#   S9 DPO 학습 (2×H200) → S10 evaluate_b0 → S11 B0 생성 acc → S12 요약
# 전 단계 멱등 · 실패 시 B0_FAILED 마커 후 정지 · smoke-first (대량 GPU 전 2샘플 예행)
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
MVP=$BAT/b0_mvp; mkdir -p "$MVP"
FAA=$REPO/outputs/step2/f0_final_v2_val_1f/checkpoint-500     # frozen FAA (회의 기본값)
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
TR1F=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/B0_FAILED"; exit 1; }

say "=== B0 체인 시작 (FAA=$FAA) ==="

# ── S1 ③ belief-swap: base(cuda:0) + step500(cuda:1) ─────────────────
run_swap(){ local name=$1 dev=$2 rec=$3 adapter=$4
  [ -s "$BAT/$name.json" ] && { say "skip $name"; return; }
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" --records "$BAT/$rec.records.jsonl" \
    --device "$dev" ${adapter:+--adapter "$adapter"} --out "$BAT/$name.json" \
    > "$BAT/$name.log" 2>&1; }
say "S1 ③ belief-swap"
run_swap swap_base_1f    cuda:0 base_1f_strict "" &
run_swap swap_step500_1f cuda:1 step500_1f "$FAA" &
# ── S2 ④ judge 재채점 (API — GPU 와 병렬) ────────────────────────────
say "S2 ④ belief_action_link judge (gemini-2.5-pro)"
( for RD in outputs/step2/f0_final_v2_val_1f outputs/step2/f0_rerun_ga8_1f "$EGO_ROOT/runs/f0_final"; do
    [ -s "$RD/judge_curve_bal.done" ] && continue
    $PY -m ego.step2_vlm_alignment.judge_reasoning --run_dir "$RD" \
        --judge_model gemini-2.5-pro --per_step 3 >> "$BAT/judge_bal.log" 2>&1 \
      && touch "$RD/judge_curve_bal.done"
  done ) &
JUDGE_PID=$!
wait %1 %2 2>/dev/null
for n in swap_base_1f swap_step500_1f; do [ -s "$BAT/$n.json" ] || die "③ $n 실패 — $BAT/$n.log"; done
say "③ 완료: $($PY -c "
import json
for n in ['swap_base_1f','swap_step500_1f']:
    d=json.load(open('$BAT/'+n+'.json'))
    print(n,'sensitivity',d['causal_sensitivity'],end='; ')")"

# ── S3 MVP 서브샘플 ───────────────────────────────────────────────────
if [ ! -s "$MVP/grpo_mvp.jsonl" ]; then
  say "S3 MVP 서브샘플 1,500 (seed 0)"
  $PY - "$TR1F" "$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f_b0meta.jsonl" "$MVP" <<'PYEOF'
import json, random, sys
tr, meta, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(tr)]
metas = {m["sample_id"]: m for m in map(json.loads, open(meta))}
random.Random(0).shuffle(rows)
picked = rows[:1500]
with open(f"{out}/grpo_mvp.jsonl", "w") as f:
    for r in picked: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{out}/b0meta_mvp.jsonl", "w") as f:
    n = 0
    for r in picked:
        m = metas.get(r["frame_id"])
        if m: f.write(json.dumps(m, ensure_ascii=False) + "\n"); n += 1
print(f"picked 1500, meta matched {n}")
PYEOF
fi

# ── smoke: 대량 GPU 전 2샘플 예행 (FAA 롤아웃 + teacher 빌드) ─────────
if [ ! -f "$MVP/.smoke_ok" ]; then
  say "smoke: FAA 롤아웃/teacher 2샘플 예행"
  head -2 "$MVP/grpo_mvp.jsonl" > "$MVP/smoke_in.jsonl"
  CUDA_VISIBLE_DEVICES=0 $PY -m ego.step2_vlm_alignment.b0.generate_faa_traces \
    --faa_adapter "$FAA" --train_jsonl "$MVP/smoke_in.jsonl" --out "$MVP/smoke_faa.jsonl" \
    --num_generations 2 > "$MVP/smoke.log" 2>&1 || die "smoke FAA 롤아웃 실패 — $MVP/smoke.log"
  head -2 "$MVP/b0meta_mvp.jsonl" > "$MVP/smoke_meta.jsonl"
  $PY -m ego.step2_vlm_alignment.b0.merge_b0_samples --faa_traces "$MVP/smoke_faa.jsonl" \
    --b0meta "$MVP/smoke_meta.jsonl" --out "$MVP/smoke_samples.jsonl" >> "$MVP/smoke.log" 2>&1 \
    || die "smoke merge 실패"
  CUDA_VISIBLE_DEVICES=0 $PY -m ego.step2_vlm_alignment.b0.build_dpo_dataset \
    --samples "$MVP/smoke_samples.jsonl" --out_train "$MVP/smoke_dpo.jsonl" \
    --out_audit "$MVP/smoke_audit.jsonl" >> "$MVP/smoke.log" 2>&1 || die "smoke teacher 빌드 실패"
  # DPO 트레이너 예행 — TRL 버전/API 비호환을 대량 GPU 전에 잡는다 (S9 실패 재발 방지)
  if [ -s "$MVP/smoke_dpo.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY src/ego/step2_vlm_alignment/b0/train_b0_dpo.py \
      --dpo_jsonl "$MVP/smoke_dpo.jsonl" --faa_adapter "$FAA" \
      --output_dir "$MVP/smoke_dpo_out" --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 --save_steps 999 >> "$MVP/smoke.log" 2>&1 \
      || die "smoke DPO 트레이너 실패 — $MVP/smoke.log"
  else
    say "⚠ smoke DPO 쌍 0개(전부 SAME/SAME 드랍) — 트레이너 예행 생략"
  fi
  touch "$MVP/.smoke_ok"; say "smoke OK (rollout+teacher+trainer)"
fi

# ── S4 FAA 롤아웃 (2-way 샤딩) ────────────────────────────────────────
if [ ! -s "$MVP/faa_traces.jsonl" ]; then
  say "S4 FAA 롤아웃 1,500×4 (2-way)"
  $PY -c "
import json
rows=[json.loads(l) for l in open('$MVP/grpo_mvp.jsonl')]
for i in range(2):
    with open(f'$MVP/in_{i}.jsonl','w') as f:
        for r in rows[i::2]: f.write(json.dumps(r,ensure_ascii=False)+'\n')"
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i $PY -m ego.step2_vlm_alignment.b0.generate_faa_traces \
      --faa_adapter "$FAA" --train_jsonl "$MVP/in_$i.jsonl" --out "$MVP/faa_$i.jsonl" \
      --num_generations 4 --temperature 1.0 > "$MVP/faa_$i.log" 2>&1 &
  done; wait
  [ -s "$MVP/faa_0.jsonl" ] && [ -s "$MVP/faa_1.jsonl" ] || die "S4 롤아웃 실패"
  cat "$MVP/faa_0.jsonl" "$MVP/faa_1.jsonl" > "$MVP/faa_traces.jsonl"
fi

# ── S5 병합 ───────────────────────────────────────────────────────────
[ -s "$MVP/b0_samples.jsonl" ] || { say "S5 병합";
  $PY -m ego.step2_vlm_alignment.b0.merge_b0_samples --faa_traces "$MVP/faa_traces.jsonl" \
    --b0meta "$MVP/b0meta_mvp.jsonl" --out "$MVP/b0_samples.jsonl" || die "S5 병합 실패"; }

# ── S6 teacher 빌드 (2-way 샤딩) ──────────────────────────────────────
if [ ! -s "$MVP/b0_dpo.jsonl" ]; then
  say "S6 teacher 빌드 (raw+projection+equivalence)"
  $PY -c "
import json
rows=[json.loads(l) for l in open('$MVP/b0_samples.jsonl')]
for i in range(2):
    with open(f'$MVP/bs_{i}.jsonl','w') as f:
        for r in rows[i::2]: f.write(json.dumps(r,ensure_ascii=False)+'\n')"
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i $PY -m ego.step2_vlm_alignment.b0.build_dpo_dataset \
      --samples "$MVP/bs_$i.jsonl" --out_train "$MVP/dpo_$i.jsonl" \
      --out_audit "$MVP/audit_$i.jsonl" > "$MVP/build_$i.log" 2>&1 &
  done; wait
  [ -s "$MVP/dpo_0.jsonl" ] && [ -s "$MVP/dpo_1.jsonl" ] || die "S6 teacher 빌드 실패"
  cat "$MVP/dpo_0.jsonl" "$MVP/dpo_1.jsonl" > "$MVP/b0_dpo.jsonl"
  cat "$MVP/audit_0.jsonl" "$MVP/audit_1.jsonl" > "$MVP/b0_audit.jsonl" 2>/dev/null || true
fi

# ── S7 재검증 게이트 ─────────────────────────────────────────────────
say "S7 DPO 데이터 재검증"
$PY -m ego.step2_vlm_alignment.b0.validate_cli --dpo "$MVP/b0_dpo.jsonl" || die "S7 검증 FAIL"

# ── S8 heldout pairs (margin 평가용, 500) ─────────────────────────────
if [ ! -s "$MVP/b0_dpo_heldout.jsonl" ]; then
  say "S8 heldout pairs 500"
  head -500 "$J1F" > "$MVP/heldout500.jsonl"
  $PY -c "
import json
rows=[json.loads(l) for l in open('$MVP/heldout500.jsonl')]
for i in range(2):
    with open(f'$MVP/ho_in_{i}.jsonl','w') as f:
        for r in rows[i::2]: f.write(json.dumps(r,ensure_ascii=False)+'\n')"
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i $PY -m ego.step2_vlm_alignment.b0.generate_faa_traces \
      --faa_adapter "$FAA" --train_jsonl "$MVP/ho_in_$i.jsonl" --out "$MVP/faa_ho_$i.jsonl" \
      --num_generations 4 > "$MVP/faa_heldout_$i.log" 2>&1 &
  done; wait
  [ -s "$MVP/faa_ho_0.jsonl" ] && [ -s "$MVP/faa_ho_1.jsonl" ] || die "S8 heldout 롤아웃 실패"
  cat "$MVP/faa_ho_0.jsonl" "$MVP/faa_ho_1.jsonl" > "$MVP/faa_heldout.jsonl"
  $PY -m ego.step2_vlm_alignment.b0.merge_b0_samples --faa_traces "$MVP/faa_heldout.jsonl" \
    --b0meta "$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f_b0meta.jsonl" \
    --out "$MVP/b0_samples_heldout.jsonl" || die "S8 병합 실패"
  CUDA_VISIBLE_DEVICES=1 $PY -m ego.step2_vlm_alignment.b0.build_dpo_dataset \
    --samples "$MVP/b0_samples_heldout.jsonl" --out_train "$MVP/b0_dpo_heldout.jsonl" \
    --out_audit "$MVP/b0_audit_heldout.jsonl" > "$MVP/build_heldout.log" 2>&1 \
    || die "S8 heldout 빌드 실패"
fi

# ── S9 DPO 학습 ───────────────────────────────────────────────────────
B0OUT=$REPO/outputs/step2/b0_mvp
if ! ls -d "$B0OUT"/checkpoint-* >/dev/null 2>&1; then
  say "S9 B0 DPO 학습"
  python -m accelerate.commands.launch --multi_gpu --num_processes 2 \
    src/ego/step2_vlm_alignment/b0/train_b0_dpo.py \
    --dpo_jsonl "$MVP/b0_dpo.jsonl" --faa_adapter "$FAA" --output_dir "$B0OUT" \
    --beta 0.1 --learning_rate 5e-6 --num_train_epochs 1.0 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --save_steps 50 || die "S9 DPO 학습 실패"
fi
B0CKPT=$(ls -d "$B0OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$B0CKPT" ] || die "B0 체크포인트 없음"

# ── S10·S11 평가 ─────────────────────────────────────────────────────
say "S10 evaluate_b0 (margin·coherence) + S11 생성 acc"
[ -s "$BAT/b0_eval.json" ] || CUDA_VISIBLE_DEVICES=0 $PY -m ego.step2_vlm_alignment.b0.evaluate_b0 \
  --dpo_jsonl "$MVP/b0_dpo_heldout.jsonl" --faa_adapter "$FAA" --b0_adapter "$B0CKPT" \
  --out "$BAT/b0_eval.json" > "$BAT/b0_eval.log" 2>&1 &
[ -s "$BAT/b0_gen_1f.json" ] || $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:1 \
  --adapter "$B0CKPT" --out "$BAT/b0_gen_1f.json" > "$BAT/b0_gen_1f.log" 2>&1 &
wait
[ -s "$BAT/b0_eval.json" ] || die "S10 evaluate_b0 실패 — $BAT/b0_eval.log"
[ -s "$BAT/b0_gen_1f.json" ] || die "S11 생성 acc 실패 — $BAT/b0_gen_1f.log"

# ── S12 요약 ─────────────────────────────────────────────────────────
$PY - "$BAT" <<'PYEOF'
import json
from pathlib import Path
bat = Path(__import__("sys").argv[1])
print("=== B0 MVP 결과 요약 ===")
for n in ["swap_base_1f", "swap_step500_1f"]:
    d = json.loads((bat / f"{n}.json").read_text())
    print(f"③ {n}: causal_sensitivity={d['causal_sensitivity']} "
          f"(swap {d['swap_action_change']} / control {d['control_action_change']})")
d = json.loads((bat / "b0_eval.json").read_text())
print("B0 margin/coherence:", {k: d[k] for k in list(d)[:8]})
g = json.loads((bat / "b0_gen_1f.json").read_text())
print(f"B0 생성 acc={g['acc']} G2={g['g2_acc']} (FAA step500 대비: 0.230 / G2 0.325)")
PYEOF
touch "$BAT/B0_DONE"; say "=== B0 체인 DONE ==="
