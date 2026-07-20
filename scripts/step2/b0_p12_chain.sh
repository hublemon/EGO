#!/usr/bin/env bash
# b0_p12_chain.sh — B0 재설계 1단계: P1(자기대조) + P2(최소대조) 최소대조 쌍 DPO.
#   진단: DPO 는 chosen/rejected 의 '가장 쉬운 차이'를 배운다 → MVP/R1 은 문체(다른 모델)를
#   배웠다(span: belief +0.802 / action +0.014). 처방: 쌍이 가르칠 것만 다르게.
#     P1 = FAA 자기 롤아웃 중 GT 맞춘 것 ≻ 틀린 것 (문체 상쇄, 정확도 직격)
#     P2 = 같은 reasoning·belief + GT action ≻ 같은 것 + 다른 후보 (action-span 직격)
#     teacher = FAA 가 8번 다 틀린 어려운 샘플만 gated 로 보충 (역할 재정의)
# 사전 등록 기준(이 단계): acc ≥ 0.26  AND  action-span margin ≥ +0.023  (인과는 다음 단계 P3)
# 마커: B0_P12_DONE / B0_P12_FAILED
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
PD=$BAT/b0_p12; mkdir -p "$PD"
FAA=$REPO/outputs/step2/f0_final_v2_val_1f/checkpoint-500
TR1F=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
B0OUT=$REPO/outputs/step2/b0_p12_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/B0_P12_FAILED"; exit 1; }

# ── S1 추가 롤아웃 4개 (기존 4개 + 신규 4개 = 8 롤아웃), 2-way 샤딩 ──────────
if [ ! -s "$PD/faa_extra.jsonl" ]; then
  say "S1 추가 롤아웃 (1500 프롬프트 × 4 gen, 2-way)"
  $PY - "$MVP/b0_samples.jsonl" "$TR1F" "$PD" <<'PYEOF' || die "S1 입력 분할 실패"
import json, sys
from pathlib import Path
mvp, tr, pd = sys.argv[1], sys.argv[2], Path(sys.argv[3])
ids = {json.loads(l)["sample_id"] for l in open(mvp)}
rows = [r for r in (json.loads(l) for l in open(tr) if l.strip())
        if str(r.get("frame_id","")) in ids]
h = len(rows)//2
for i, part in enumerate((rows[:h], rows[h:])):
    with open(pd/f"in_{i}.jsonl","w") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"[S1] {len(rows)} 프롬프트 → 2-way")
PYEOF
  for i in 0 1; do
    [ -s "$PD/faa_extra_$i.jsonl" ] && continue
    CUDA_VISIBLE_DEVICES=$i $PY -m ego.step2_vlm_alignment.b0.generate_faa_traces \
      --faa_adapter "$FAA" --train_jsonl "$PD/in_$i.jsonl" --out "$PD/faa_extra_$i.jsonl" \
      --num_generations 4 > "$PD/faa_extra_$i.log" 2>&1 & eval "R$i=\$!"
  done
  wait ${R0:-} || die "롤아웃 shard0 실패 — $PD/faa_extra_0.log"
  wait ${R1:-} || die "롤아웃 shard1 실패 — $PD/faa_extra_1.log"
  cat "$PD/faa_extra_0.jsonl" "$PD/faa_extra_1.jsonl" > "$PD/faa_extra.jsonl"
  say "S1 OK: $(wc -l < "$PD/faa_extra.jsonl") 샘플"
fi

# ── S2 기존 4 + 신규 4 병합 → 8 롤아웃 샘플 파일 ────────────────────────────
if [ ! -s "$PD/b0_samples_8gen.jsonl" ]; then
  say "S2 롤아웃 병합 (4+4=8)"
  $PY - "$MVP/b0_samples.jsonl" "$PD/faa_extra.jsonl" "$PD/b0_samples_8gen.jsonl" <<'PYEOF' \
    || die "S2 병합 실패"
import json, sys
base, extra, out = sys.argv[1], sys.argv[2], sys.argv[3]
ex = {}
for l in open(extra):
    if l.strip():
        r = json.loads(l); ex[r["sample_id"]] = r.get("faa_traces", [])
n_merged = 0
with open(out, "w") as f:
    for l in open(base):
        if not l.strip(): continue
        s = json.loads(l)
        add = ex.get(s["sample_id"], [])
        if add: n_merged += 1
        s["faa_traces"] = (s.get("faa_traces") or []) + add
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"[S2] {n_merged} 샘플에 롤아웃 추가")
PYEOF
  say "S2 OK"
fi

# ── S3 P1+P2 쌍 생성 ────────────────────────────────────────────────────────
if [ ! -s "$PD/pairs_p12.jsonl" ]; then
  say "S3 P1+P2 최소대조 쌍 생성"
  $PY -m ego.step2_vlm_alignment.b0.build_pairs_contrastive \
    --samples "$PD/b0_samples_8gen.jsonl" --train_jsonl "$TR1F" \
    --out_pairs "$PD/pairs_p12.jsonl" --out_hard "$PD/hard_ids.json" \
    --out_stats "$PD/stats_p12.json" > "$PD/build_p12.log" 2>&1 || die "S3 실패 — $PD/build_p12.log"
  cat "$PD/build_p12.log"
  say "S3 OK: $(wc -l < "$PD/pairs_p12.jsonl") 쌍"
fi

# ── S4 teacher 보충 (FAA 가 8번 다 틀린 어려운 샘플만, 2-way) ───────────────
if [ ! -s "$PD/pairs_teacher.jsonl" ]; then
  say "S4 teacher 보충 (전부-오답 구간)"
  $PY - "$PD/b0_samples_8gen.jsonl" "$PD/hard_ids.json" "$PD" <<'PYEOF' || die "S4 분할 실패"
import json, sys
from pathlib import Path
src, hard, pd = sys.argv[1], sys.argv[2], Path(sys.argv[3])
ids = set(json.load(open(hard)))
rows = [s for s in (json.loads(l) for l in open(src) if l.strip()) if s["sample_id"] in ids]
h = len(rows)//2
for i, part in enumerate((rows[:h], rows[h:])):
    with open(pd/f"hard_{i}.jsonl","w") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"[S4] 어려운 샘플 {len(rows)} → 2-way")
PYEOF
  for sh in 0 1; do
    [ -s "$PD/t_train_$sh.jsonl" ] && continue
    CUDA_VISIBLE_DEVICES=$sh $PY -m ego.step2_vlm_alignment.b0.build_dpo_dataset_r1 \
      --samples "$PD/hard_$sh.jsonl" --train_jsonl "$TR1F" --shard 0 --num_shards 1 \
      --out_train "$PD/t_train_$sh.jsonl" --out_audit "$PD/t_audit_$sh.jsonl" \
      --out_stats "$PD/t_stats_$sh.json" > "$PD/t_build_$sh.log" 2>&1 & eval "T$sh=\$!"
  done
  wait ${T0:-} 2>/dev/null || true; wait ${T1:-} 2>/dev/null || true
  cat "$PD/t_train_0.jsonl" "$PD/t_train_1.jsonl" > "$PD/pairs_teacher.jsonl" 2>/dev/null || true
  say "S4 OK: $(wc -l < "$PD/pairs_teacher.jsonl" 2>/dev/null || echo 0) teacher 쌍"
fi

# ── S5 통합 + DPO ───────────────────────────────────────────────────────────
if [ ! -f "$B0OUT/TRAINING_DONE" ]; then
  cat "$PD/pairs_p12.jsonl" "$PD/pairs_teacher.jsonl" > "$PD/pairs_all.jsonl" 2>/dev/null \
    || cp "$PD/pairs_p12.jsonl" "$PD/pairs_all.jsonl"
  N=$(wc -l < "$PD/pairs_all.jsonl"); say "S5 DPO 학습 (cuda:0, $N 쌍)"
  [ "$N" -ge 2000 ] || die "쌍 수 부족($N) — 롤아웃/파싱 점검 필요"
  rm -rf "$B0OUT"
  CUDA_VISIBLE_DEVICES=0 $PY src/ego/step2_vlm_alignment/b0/train_b0_dpo.py \
    --dpo_jsonl "$PD/pairs_all.jsonl" --faa_adapter "$FAA" --output_dir "$B0OUT" \
    --max_length 4096 --max_prompt_length 1024 \
    > "$PD/dpo.log" 2>&1 || die "DPO 실패 — $PD/dpo.log"
  $PY -c "
from safetensors.torch import load_file
a=load_file('$FAA/adapter_model.safetensors'); b=load_file('$B0OUT/adapter_model.safetensors')
d=max((a[k]-b[k]).abs().max().item() for k in a if k in b)
print('[guard] max adapter weight diff vs FAA:', d)
raise SystemExit(0 if d>1e-7 else 1)" || die "무학습: 체크포인트가 FAA 와 동일"
  touch "$B0OUT/TRAINING_DONE"
fi

# ── S6 평가 (R1 과 동일 측정자 — heldout pair 재사용으로 비교 가능) ─────────
[ -s "$BAT/b0p12_gen_1f.json" ] || { say "S6a 생성 acc";
  $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:0 \
    --adapter "$B0OUT" --out "$BAT/b0p12_gen_1f.json" > "$BAT/b0p12_gen_1f.log" 2>&1 \
    || die "생성 acc 실패"; }
[ -s "$BAT/remeasure_b0p12.json" ] || { say "S6b span-margin";
  $PY scripts/step2/remeasure_b0_margin.py --dpo_jsonl "$R1D/b0_r1_dpo_heldout.jsonl" \
    --policies "faa:$FAA,b0p12:$B0OUT" --device cuda:1 \
    --out "$BAT/remeasure_b0p12.json" > "$BAT/remeasure_b0p12.log" 2>&1 || die "margin 실패"; }
[ -s "$BAT/swap_b0p12.json" ] || { say "S6c ③ swap (참고 — 이 단계 기준 아님)";
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
    --records "$BAT/b0p12_gen_1f.records.jsonl" --adapter "$B0OUT" \
    --device cuda:1 --out "$BAT/swap_b0p12.json" > "$BAT/swap_b0p12.log" 2>&1 || true; }

# ── S7 판정 ─────────────────────────────────────────────────────────────────
$PY - "$BAT" "$PD" > "$BAT/B0_P12_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat, pd = Path(sys.argv[1]), Path(sys.argv[2])
def J(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None
gen, rem, sw = J(bat/"b0p12_gen_1f.json"), J(bat/"remeasure_b0p12.json"), J(bat/"swap_b0p12.json")
st = J(pd/"stats_p12.json")
span = (rem or {}).get("policies", {}).get("b0p12", {}).get(
    "improvement_vs_ref", {}).get("span", {}).get("action")
acc = (gen or {}).get("acc")
print("# B0-P12 (최소대조 쌍) 결과 — 자동 생성\n")
print("| 지표 | MVP | R1 | 목표 | P12 |")
print("|---|---|---|---|---|")
print(f"| 생성 acc | 0.248 | 0.238 | ≥ 0.26 | {acc} |")
print(f"| action-span margin | +0.014 | +0.007 | ≥ +0.023 | {span} |")
print(f"| G2 | 0.342 | 0.2764 | (참고) | {(gen or {}).get('g2_acc')} |")
print(f"| ③ 인과 | 0.006 | 0.008 | (다음 단계) | {(sw or {}).get('causal_sensitivity')} |")
if st:
    print(f"\n쌍 구성: P1 {st['p1_pairs']} · P2 {st['p2_pairs']} · 최종 {st['final_pairs']}")
    print(f"롤아웃 분포: 혼재 {st['mixed']} / 전부정답 {st['all_correct']} / 전부오답 {st['all_wrong']}")
ok = (acc is not None and acc >= 0.26) and (span is not None and span >= 0.023)
print("\n판정:", "**통과 — P3(belief 반사실) 단계로 진행 가치**" if ok
      else "미충족 — 항목별 원인 분석 필요")
Path(bat/"B0_P12_PASSED").touch() if ok else None
PYEOF
cat "$BAT/B0_P12_RESULTS.md"
touch "$BAT/B0_P12_DONE"; say "=== B0-P12 체인 DONE ==="
