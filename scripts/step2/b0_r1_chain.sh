#!/usr/bin/env bash
# b0_r1_chain.sh — B0-R1: GT-hidden gated teacher 리팩터 무인 체인.
#   리팩터 핸드오프 v2 정정 반영: hard action gate · goal suffix 추출 · G2 retention 게이트.
#   FAA 롤아웃(rejected)은 MVP 산출 전량 재사용 — 변수는 teacher 구성뿐.
# smoke(3샘플 전 경로) → pair 재구축(2-way, cuda:1 즉시 + cuda:0 은 F0_GA 종료 대기)
# → heldout 재구축 → DPO(max_length 4096 + no-train guard) → 평가(acc·③·span-margin)
# → B0_VALIDATED 판정 → B0_R1_DONE / B0_R1_FAILED.
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
R1D=$BAT/b0_r1; mkdir -p "$R1D"
FAA=$REPO/outputs/step2/f0_final_v2_val_1f/checkpoint-500
TR1F=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
B0OUT=$REPO/outputs/step2/b0_r1_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/B0_R1_FAILED"; exit 1; }

# ── S1 smoke: 3샘플 전 경로 (goal 누출 assert 는 빌더 내장 — 산출 통계로 검사) ──
if [ ! -f "$R1D/.smoke_ok" ]; then
  say "S1 smoke: 3샘플 gated 빌드 (cuda:1)"
  CUDA_VISIBLE_DEVICES=1 $PY -m ego.step2_vlm_alignment.b0.build_dpo_dataset_r1 \
    --samples "$MVP/b0_samples.jsonl" --train_jsonl "$TR1F" --limit 3 \
    --out_train "$R1D/smoke_train.jsonl" --out_audit "$R1D/smoke_audit.jsonl" \
    --out_stats "$R1D/smoke_stats.json" > "$R1D/smoke.log" 2>&1 || die "smoke 실패 — $R1D/smoke.log"
  $PY - "$R1D/smoke_stats.json" <<'PYEOF' || die "smoke 통계 이상"
import json, sys
s = json.load(open(sys.argv[1]))["r1"]
total = s["gate_pass"] + s["gate_fail"] + s["goal_leak_dropped"] + s["no_future_suffix"]
assert total >= 1, "어떤 샘플도 gate 단계에 도달하지 못함"
print("[smoke]", json.dumps(s, ensure_ascii=False))
PYEOF
  grep -c 'DROPPED_GATE_FAIL' "$R1D/smoke_audit.jsonl" >/dev/null 2>&1 || true
  touch "$R1D/.smoke_ok"; say "S1 OK"
else say "S1 skip"; fi

# ── S2 pair 재구축 (train 1500, 2-way): shard0=cuda:1 즉시, shard1=cuda:0 대기 ──
build_shard(){ local sh=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu $PY -m ego.step2_vlm_alignment.b0.build_dpo_dataset_r1 \
    --samples "$MVP/b0_samples.jsonl" --train_jsonl "$TR1F" \
    --shard $sh --num_shards 2 \
    --out_train "$R1D/train_$sh.jsonl" --out_audit "$R1D/audit_$sh.jsonl" \
    --out_stats "$R1D/stats_$sh.json" > "$R1D/build_$sh.log" 2>&1
}
if [ ! -s "$R1D/b0_r1_dpo.jsonl" ]; then
  say "S2 재구축: shard0(cuda:1) 시작 — shard1 은 F0_GA 마커 대기"
  [ -s "$R1D/train_0.jsonl" ] || { build_shard 0 1 & P0=$!; }
  if [ ! -s "$R1D/train_1.jsonl" ]; then
    while [ ! -f "$BAT/F0_GA_DONE" ] && [ ! -f "$BAT/F0_GA_FAILED" ]; do sleep 120; done
    say "cuda:0 확보 — shard1 시작"
    build_shard 1 0 || die "shard1 실패 — $R1D/build_1.log"
  fi
  wait ${P0:-} 2>/dev/null || { [ -s "$R1D/train_0.jsonl" ] || die "shard0 실패 — $R1D/build_0.log"; }
  cat "$R1D/train_0.jsonl" "$R1D/train_1.jsonl" > "$R1D/b0_r1_dpo.jsonl"
  cat "$R1D/audit_0.jsonl" "$R1D/audit_1.jsonl" > "$R1D/b0_r1_audit.jsonl"
  say "S2 OK: $(wc -l < "$R1D/b0_r1_dpo.jsonl") pairs"
fi

# ── S3 retention 게이트 (G2/G1 ≥ 0.5 보고 — 미달 시 마커만, 중단 안 함) ─────
$PY - "$R1D" > "$R1D/retention.json" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
agg = {"gate_pass":0,"gate_fail":0,"goal_leak_dropped":0,"no_future_suffix":0,
       "by_group":{g:{"seen":0,"pass":0} for g in ("G1","G2","OUT")}}
for i in (0,1):
    s = json.load(open(d/f"stats_{i}.json"))["r1"]
    for k in ("gate_pass","gate_fail","goal_leak_dropped","no_future_suffix"):
        agg[k]+=s[k]
    for g in agg["by_group"]:
        for kk in ("seen","pass"): agg["by_group"][g][kk]+=s["by_group"][g][kk]
def rate(g):
    x=agg["by_group"][g]; return x["pass"]/x["seen"] if x["seen"] else None
r1,r2 = rate("G1"), rate("G2")
agg["retention_g1"], agg["retention_g2"] = r1, r2
agg["g2_over_g1"] = (r2/r1) if (r1 and r2 is not None) else None
print(json.dumps(agg, ensure_ascii=False, indent=1))
PYEOF
cat "$R1D/retention.json"
$PY -c "
import json; a=json.load(open('$R1D/retention.json'))
ok = (a['g2_over_g1'] or 0) >= 0.5
open('$BAT/NEEDS_DECISION_B0R1_G2','w').close() if not ok else None
print('[gate] G2/G1 retention', a['g2_over_g1'], 'OK' if ok else '미달(마커 생성, 계속 진행)')"

# ── S4 heldout 재구축 (③·margin 용, 2-way) ─────────────────────────────────
if [ ! -s "$R1D/b0_r1_dpo_heldout.jsonl" ]; then
  say "S4 heldout 재구축 (2-way)"
  for sh in 0 1; do
    [ -s "$R1D/ho_train_$sh.jsonl" ] && continue
    CUDA_VISIBLE_DEVICES=$sh $PY -m ego.step2_vlm_alignment.b0.build_dpo_dataset_r1 \
      --samples "$MVP/b0_samples_heldout.jsonl" --train_jsonl "$J1F" \
      --shard $sh --num_shards 2 \
      --out_train "$R1D/ho_train_$sh.jsonl" --out_audit "$R1D/ho_audit_$sh.jsonl" \
      --out_stats "$R1D/ho_stats_$sh.json" > "$R1D/ho_build_$sh.log" 2>&1 & eval "H$sh=\$!"
  done
  wait ${H0:-} 2>/dev/null || true; wait ${H1:-} 2>/dev/null || true
  [ -s "$R1D/ho_train_0.jsonl" ] || die "heldout shard0 실패"
  [ -s "$R1D/ho_train_1.jsonl" ] || die "heldout shard1 실패"
  cat "$R1D/ho_train_0.jsonl" "$R1D/ho_train_1.jsonl" > "$R1D/b0_r1_dpo_heldout.jsonl"
  say "S4 OK: $(wc -l < "$R1D/b0_r1_dpo_heldout.jsonl") heldout pairs"
fi

# ── S5 DPO (FAA init + frozen FAA ref, max_length 4096) + no-train guard ───
if [ ! -f "$B0OUT/TRAINING_DONE" ]; then
  rm -rf "$B0OUT"
  say "S5 DPO 학습 (cuda:0)"
  CUDA_VISIBLE_DEVICES=0 $PY src/ego/step2_vlm_alignment/b0/train_b0_dpo.py \
    --dpo_jsonl "$R1D/b0_r1_dpo.jsonl" --faa_adapter "$FAA" --output_dir "$B0OUT" \
    --max_length 4096 --max_prompt_length 1024 \
    > "$R1D/dpo.log" 2>&1 || die "DPO 실패 — $R1D/dpo.log"
  $PY -c "
from safetensors.torch import load_file
a=load_file('$FAA/adapter_model.safetensors'); b=load_file('$B0OUT/adapter_model.safetensors')
d=max((a[k]-b[k]).abs().max().item() for k in a if k in b)
print('[guard] max adapter weight diff vs FAA:', d)
raise SystemExit(0 if d>1e-7 else 1)" || die "무학습: 체크포인트가 FAA 와 동일"
  touch "$B0OUT/TRAINING_DONE"
fi

# ── S6 평가: 생성 acc(cuda:0) ∥ ③ swap·margin(cuda:1) ──────────────────────
[ -s "$BAT/b0r1_gen_1f.json" ] || { say "S6a 생성 acc";
  $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:0 \
    --adapter "$B0OUT" --out "$BAT/b0r1_gen_1f.json" > "$BAT/b0r1_gen_1f.log" 2>&1 \
    || die "생성 acc 실패"; }
[ -s "$BAT/swap_b0r1.json" ] || { say "S6b ③ belief-swap";
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
    --records "$BAT/b0r1_gen_1f.records.jsonl" --adapter "$B0OUT" \
    --device cuda:1 --out "$BAT/swap_b0r1.json" > "$BAT/swap_b0r1.log" 2>&1 \
    || die "③ swap 실패"; }
[ -s "$BAT/remeasure_b0r1.json" ] || { say "S6c span-margin 재측정";
  $PY scripts/step2/remeasure_b0_margin.py --dpo_jsonl "$R1D/b0_r1_dpo_heldout.jsonl" \
    --policies "faa:$FAA,b0r1:$B0OUT" --device cuda:1 \
    --out "$BAT/remeasure_b0r1.json" > "$BAT/remeasure_b0r1.log" 2>&1 \
    || die "margin 재측정 실패"; }

# ── S7 요약 + B0_VALIDATED 판정 ─────────────────────────────────────────────
$PY - "$BAT" "$R1D" > "$BAT/B0_R1_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat, r1d = Path(sys.argv[1]), Path(sys.argv[2])
def J(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None
gen, swap = J(bat/"b0r1_gen_1f.json"), J(bat/"swap_b0r1.json")
rem, ret = J(bat/"remeasure_b0r1.json"), J(r1d/"retention.json")
print("# B0-R1 (GT-hidden gated teacher) 결과 — 자동 생성\n")
print("| 지표 | MVP(B0) | 목표 | R1 |")
print("|---|---|---|---|")
span = None
if rem:
    blk = rem.get("policies", {}).get("b0r1", {})
    span = blk.get("improvement_vs_ref", {}).get("span", {}).get("action")
cs = swap.get("causal_sensitivity") if swap else None
acc = gen.get("acc") if gen else None
g2g1 = ret.get("g2_over_g1") if ret else None
print(f"| ③ 인과 민감도 | 0.006 | > 0.03 | {cs} |")
print(f"| action-span margin | +0.014 | ≥ +0.023 | {span} |")
print(f"| 생성 acc | 0.248 | ≥ 0.248 | {acc} |")
print(f"| G2/G1 retention | — | ≥ 0.5 | {g2g1} |")
if gen: print(f"\n생성 상세: G2 {gen.get('g2_acc')} · wm_follow {gen.get('wm_follow')} · parse {gen.get('parse_rate')}")
if ret: print(f"gate: pass {ret['gate_pass']} / fail {ret['gate_fail']} · goal 누출 드랍 {ret['goal_leak_dropped']}")
ok = all(x is not None for x in (cs, acc, g2g1)) and cs > 0.03 and acc >= 0.248 and g2g1 >= 0.5 \
     and (span is None or span >= 0.023)
print("\n판정:", "**B0_VALIDATED 충족**" if ok else "미충족 — 사전 등록 기준 대비 부족 항목 확인")
Path(bat/"B0_VALIDATED").touch() if ok else None
PYEOF
cat "$BAT/B0_R1_RESULTS.md"
touch "$BAT/B0_R1_DONE"; say "=== B0-R1 체인 DONE ==="
