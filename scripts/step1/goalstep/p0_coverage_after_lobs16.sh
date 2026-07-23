#!/usr/bin/env bash
# p0_coverage_after_lobs16.sh — lobs16 학습 종료 대기 → full-val coverage@K 측정 (P0, 재추출 0)
#
# 배경: Pro(RL+EMA)+Retro(CE) anticipation 계획의 Gate 0.
#   lobs8 dev subset 실측: action top-5 26.9 / top-10 39.6 / top-15 47.5 (%) — 50% 근접.
#   본 체인은 lobs16 학습이 끝나면 lobs16·lobs8 두 체크포인트를 full-val(7,214)로 평가해
#   coverage@{5,10,15}를 비교하고 Gate 0(≥50%) 판정을 기록한다.
set -u
REPO=/mnt/nvme/migration/jihun/EGO_jihun2
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
RUN16=$REPO/outputs/goalstep/runs/z1_start_m1_lobs16_vna
RUN8=$REPO/outputs/goalstep/runs/z1_start_m1_lobs8_vna
OUT=$REPO/outputs/goalstep/coverage_p0
LOG=$OUT/chain.log
mkdir -p "$OUT"
cd "$REPO" || exit 1
say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

say "===== P0 coverage 체인 시작: lobs16 학습 종료 대기 ====="

# ── 1) lobs16 학습 종료 대기 (5분 폴링, 최대 8h) ────────────────────────────
deadline=$(( $(date +%s) + 8*3600 ))
while [ ! -s "$RUN16/final_metrics.json" ]; do
  if [ $(date +%s) -gt $deadline ]; then
    say "치명적: 8h 내 lobs16 final_metrics.json 미생성"; touch "$OUT/CHAIN_FAILED"; exit 1
  fi
  sleep 300
done
say "lobs16 학습 완료 감지 → full-val 평가 시작"
sleep 60   # 체크포인트 flush 여유

# ── 2) full-val 평가 (lobs16 → lobs8 순, best_action_top5 체크포인트) ────────
for name in lobs16 lobs8; do
  if [ "$name" = "lobs16" ]; then RUN=$RUN16; CFG=configs/step1/goalstep/z1_start_m1_lobs16_vna.yaml
  else RUN=$RUN8; CFG=configs/step1/goalstep/z1_start_m1_lobs8_vna.yaml; fi
  CKPT=$RUN/best_action_top5.pt
  [ -s "$CKPT" ] || CKPT=$RUN/best.pt
  if [ ! -s "$OUT/${name}_full.json" ]; then
    say "--- $name full-val 평가 ($CKPT) ---"
    CUDA_VISIBLE_DEVICES=0 $PY scripts/step1/goalstep/evaluate_checkpoint_full.py \
      --config "$CFG" --checkpoint "$CKPT" \
      --output "$OUT/${name}_full.json" --predictions-output "$OUT/${name}_preds.jsonl" \
      > "$OUT/${name}_eval.log" 2>&1
    say "  rc=$?"
  fi
done

# ── 3) Gate 0 판정 + 리포트 ─────────────────────────────────────────────────
$PY - "$OUT" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def acc(p):
    m = json.load(open(p))
    # evaluate_checkpoint_full 출력 구조 대응 (metrics 중첩 유무 모두 탐색)
    def find(d):
        if isinstance(d, dict):
            if "accuracy_top5" in d: return d
            for v in d.values():
                r = find(v)
                if r: return r
        return None
    d = find(m)
    return {k: round(float(d[k]["action"]), 2) for k in
            ("accuracy_top1","accuracy_top5","accuracy_top10","accuracy_top15")}
rows = {n: acc(out/f"{n}_full.json") for n in ("lobs8","lobs16")}
lines = ["# P0 coverage 결과 (full-val 7,214 · start−1s strict)","",
         "| config | top-1 | top-5 | top-10 | top-15 |","|---|---:|---:|---:|---:|"]
best15 = 0.0
for n, a in rows.items():
    lines.append(f"| {n} | {a['accuracy_top1']} | {a['accuracy_top5']} | {a['accuracy_top10']} | {a['accuracy_top15']} |")
    best15 = max(best15, a["accuracy_top15"])
    print(n, a)
gate = "통과" if best15 >= 50.0 else ("근접 — K=20 확장 검토" if best15 >= 45.0 else "미달 — 옵션 8 probe/τ_a 재검토")
lines += ["", f"**Gate 0 (cov@K ≥ 50%)**: top-15 최고 {best15}% → **{gate}**"]
(out/"COVERAGE_P0.md").write_text("\n".join(lines))
print("Gate 0:", gate, f"(top-15 최고 {best15}%)")
PYEOF

say "===== P0 coverage 체인 종료 ====="
touch "$OUT/CHAIN_DONE"
