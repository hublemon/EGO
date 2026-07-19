#!/usr/bin/env bash
# b0_full_chain.sh — B0 풀 스케일: B0_VALIDATED 일 때만 R1 레시피를 전체 train set 으로 확장.
#   원칙: 검증된 R1 레시피 그대로, FAA=F0-L 고정 — 확정 변수는 스케일 하나뿐.
#   (새 F0 로의 FAA 마이그레이션은 사람 리뷰 후 별도 단계)
# 게이트: B0_R1_DONE 대기 → B0_VALIDATED 없으면 NEEDS 마커 + 종료(풀 스케일 금지 유지)
#        → F0_WE_{DONE|SKIPPED|FAILED} 대기 (GPU 확보)
# 단계: S1 신규 프롬프트 FAA 롤아웃(2-way) → S2 R1 gated pair 빌드(2-way) + 기존 R1 pair 병합
#      → S3 전체 DPO → S4 평가(acc·③·span-margin) → B0_FULL_DONE / B0_FULL_FAILED
set -euo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"; PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
BAT=$EGO_ROOT/runs/f0_battery
MVP=$BAT/b0_mvp
R1D=$BAT/b0_r1
FD=$BAT/b0_full; mkdir -p "$FD"
FAA=$REPO/outputs/step2/f0_final_v2_val_1f/checkpoint-500
TR1F=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
B0META=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f_b0meta.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
B0OUT=$REPO/outputs/step2/b0_full_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/B0_FULL_FAILED"; exit 1; }

say "=== B0 풀 스케일 체인: B0_R1 마커 대기 ==="
while [ ! -f "$BAT/B0_R1_DONE" ] && [ ! -f "$BAT/B0_R1_FAILED" ]; do sleep 300; done
[ -f "$BAT/B0_R1_FAILED" ] && { say "R1 실패 — 풀 스케일 보류"; touch "$BAT/B0_FULL_SKIPPED"; exit 0; }
if [ ! -f "$BAT/B0_VALIDATED" ]; then
  say "B0_VALIDATED 미충족 — 풀 스케일 금지 유지 (사전 등록 게이트)"
  touch "$BAT/NEEDS_DECISION_B0_FULL"; touch "$BAT/B0_FULL_SKIPPED"; exit 0
fi
say "B0_VALIDATED 확인 — F0-WE 종료 대기 (GPU 확보)"
while [ ! -f "$BAT/F0_WE_DONE" ] && [ ! -f "$BAT/F0_WE_SKIPPED" ] && [ ! -f "$BAT/F0_WE_FAILED" ]; do sleep 300; done
while [ ! -f "$BAT/F0_WEMA_DONE" ] && [ ! -f "$BAT/F0_WEMA_FAILED" ]; do sleep 300; done
say "GPU 확보 — 풀 스케일 시작"

# ── S1 신규 프롬프트 FAA 롤아웃 (MVP 1500 제외 전체, 2-way) ─────────────────
if [ ! -s "$FD/faa_new.jsonl" ]; then
  say "S1 신규 프롬프트 추출 + FAA 롤아웃"
  $PY - "$TR1F" "$MVP/b0_samples.jsonl" "$FD" <<'PYEOF' || die "신규 목록 추출 실패"
import json, sys
from pathlib import Path
tr, mvp, fd = sys.argv[1], sys.argv[2], Path(sys.argv[3])
done = {json.loads(l)["sample_id"] for l in open(mvp)}
rows = [json.loads(l) for l in open(tr) if l.strip()]
new = [r for r in rows if str(r.get("frame_id","")) not in done]
h = len(new)//2
for i, part in enumerate((new[:h], new[h:])):
    with open(fd/f"in_new_{i}.jsonl", "w") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"[S1] 신규 {len(new)} (기존 {len(done)}) → 2-way")
PYEOF
  for i in 0 1; do
    [ -s "$FD/faa_new_$i.jsonl" ] && continue
    CUDA_VISIBLE_DEVICES=$i $PY -m ego.step2_vlm_alignment.b0.generate_faa_traces \
      --faa_adapter "$FAA" --train_jsonl "$FD/in_new_$i.jsonl" --out "$FD/faa_new_$i.jsonl" \
      --num_generations 4 > "$FD/faa_new_$i.log" 2>&1 & eval "R$i=\$!"
  done
  wait ${R0:-} || die "FAA 롤아웃 shard0 실패"; wait ${R1:-} || die "FAA 롤아웃 shard1 실패"
  cat "$FD/faa_new_0.jsonl" "$FD/faa_new_1.jsonl" > "$FD/faa_new.jsonl"
  say "S1 OK: $(wc -l < "$FD/faa_new.jsonl") 롤아웃"
fi

# ── S2 병합 + R1 gated pair 빌드 (신규만, 2-way) → 기존 R1 pair 와 합침 ─────
if [ ! -s "$FD/b0_full_dpo.jsonl" ]; then
  [ -s "$FD/samples_new.jsonl" ] || { say "S2 병합";
    $PY -m ego.step2_vlm_alignment.b0.merge_b0_samples --faa_traces "$FD/faa_new.jsonl" \
      --b0meta "$B0META" --out "$FD/samples_new.jsonl" || die "병합 실패"; }
  for sh in 0 1; do
    [ -s "$FD/train_new_$sh.jsonl" ] && continue
    CUDA_VISIBLE_DEVICES=$sh $PY -m ego.step2_vlm_alignment.b0.build_dpo_dataset_r1 \
      --samples "$FD/samples_new.jsonl" --train_jsonl "$TR1F" \
      --shard $sh --num_shards 2 \
      --out_train "$FD/train_new_$sh.jsonl" --out_audit "$FD/audit_new_$sh.jsonl" \
      --out_stats "$FD/stats_new_$sh.json" > "$FD/build_new_$sh.log" 2>&1 & eval "B$sh=\$!"
  done
  wait ${B0:-} 2>/dev/null || true; wait ${B1:-} 2>/dev/null || true
  [ -s "$FD/train_new_0.jsonl" ] || die "빌드 shard0 실패 — $FD/build_new_0.log"
  [ -s "$FD/train_new_1.jsonl" ] || die "빌드 shard1 실패 — $FD/build_new_1.log"
  cat "$R1D/train_0.jsonl" "$R1D/train_1.jsonl" \
      "$FD/train_new_0.jsonl" "$FD/train_new_1.jsonl" > "$FD/b0_full_dpo.jsonl"
  say "S2 OK: 전체 $(wc -l < "$FD/b0_full_dpo.jsonl") pairs (R1 재사용 + 신규)"
fi

# ── S3 전체 DPO + no-train guard ────────────────────────────────────────────
if [ ! -f "$B0OUT/TRAINING_DONE" ]; then
  rm -rf "$B0OUT"
  say "S3 DPO 학습 (cuda:0, 전체 pairs)"
  CUDA_VISIBLE_DEVICES=0 $PY src/ego/step2_vlm_alignment/b0/train_b0_dpo.py \
    --dpo_jsonl "$FD/b0_full_dpo.jsonl" --faa_adapter "$FAA" --output_dir "$B0OUT" \
    --max_length 4096 --max_prompt_length 1024 \
    > "$FD/dpo.log" 2>&1 || die "DPO 실패 — $FD/dpo.log"
  $PY -c "
from safetensors.torch import load_file
a=load_file('$FAA/adapter_model.safetensors'); b=load_file('$B0OUT/adapter_model.safetensors')
d=max((a[k]-b[k]).abs().max().item() for k in a if k in b)
print('[guard] max adapter weight diff vs FAA:', d)
raise SystemExit(0 if d>1e-7 else 1)" || die "무학습: 체크포인트가 FAA 와 동일"
  touch "$B0OUT/TRAINING_DONE"
fi

# ── S4 평가 ─────────────────────────────────────────────────────────────────
[ -s "$BAT/b0full_gen_1f.json" ] || { say "S4a 생성 acc";
  $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:0 \
    --adapter "$B0OUT" --out "$BAT/b0full_gen_1f.json" > "$BAT/b0full_gen_1f.log" 2>&1 \
    || die "생성 acc 실패"; }
[ -s "$BAT/swap_b0full.json" ] || { say "S4b ③ swap";
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
    --records "$BAT/b0full_gen_1f.records.jsonl" --adapter "$B0OUT" \
    --device cuda:1 --out "$BAT/swap_b0full.json" > "$BAT/swap_b0full.log" 2>&1 \
    || die "③ swap 실패"; }
[ -s "$BAT/remeasure_b0full.json" ] || { say "S4c span-margin";
  $PY scripts/step2/remeasure_b0_margin.py --dpo_jsonl "$R1D/b0_r1_dpo_heldout.jsonl" \
    --policies "faa:$FAA,b0full:$B0OUT" --device cuda:1 \
    --out "$BAT/remeasure_b0full.json" > "$BAT/remeasure_b0full.log" 2>&1 \
    || die "margin 실패"; }

# ── 요약 ────────────────────────────────────────────────────────────────────
$PY - "$BAT" "$FD" > "$BAT/B0_FULL_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat, fd = Path(sys.argv[1]), Path(sys.argv[2])
def J(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None
gen, sw, rem = J(bat/"b0full_gen_1f.json"), J(bat/"swap_b0full.json"), J(bat/"remeasure_b0full.json")
r1g, r1s = J(bat/"b0r1_gen_1f.json"), J(bat/"swap_b0r1.json")
print("# B0 풀 스케일 (R1 레시피 × 전체 train set) 결과 — 자동 생성\n")
print("| 지표 | B0-MVP | B0-R1 | B0-FULL |")
print("|---|---|---|---|")
print(f"| acc | 0.248 | {r1g and r1g.get('acc')} | {gen and gen.get('acc')} |")
print(f"| G2 | 0.342 | {r1g and r1g.get('g2_acc')} | {gen and gen.get('g2_acc')} |")
print(f"| ③ 인과 | 0.006 | {r1s and r1s.get('causal_sensitivity')} | {sw and sw.get('causal_sensitivity')} |")
if rem:
    blk = rem.get("policies", {}).get("b0full", {})
    print(f"| action-span margin | +0.014 | (R1 결과 참조) | {blk.get('improvement_vs_ref',{}).get('span',{}).get('action')} |")
n = sum(1 for _ in open(fd/"b0_full_dpo.jsonl")) if (fd/"b0_full_dpo.jsonl").exists() else None
print(f"\n학습 pairs: {n} (R1 재사용 + 신규 빌드). 스케일링 판정: FULL 이 R1 대비 개선되면 데이터 스케일 유효.")
PYEOF
cat "$BAT/B0_FULL_RESULTS.md"
touch "$BAT/B0_FULL_DONE"; say "=== B0 풀 스케일 체인 DONE ==="
