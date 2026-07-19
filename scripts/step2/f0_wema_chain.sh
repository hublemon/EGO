#!/usr/bin/env bash
# f0_wema_chain.sh — F0-W-EMA: full-trace + WM likelihood 보상(GT-free) + EMA 기준선.
#   목적: 같은 최적화(EMA)에서 보상만 WM vs GT 를 갈라 비교 (F0-WE 와 쌍).
#   예상 분기: W-EMA 는 wm_follow↑·acc≤0.374·G2↓(복사기 수렴) vs WE 는 acc·G2 동반↑ 여부.
#   GT-free 이므로 성공 시 '방법' 자격 있음. 전체 5000 샘플 (GT 필터 미적용).
# GPU: cuda:1 — B0_R1 종료 후 슬롯 사용. 마커: F0_WEMA_DONE/FAILED.
set -euo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"; PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
BAT=$EGO_ROOT/runs/f0_battery
TRAIN_JSONL=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
OUT=$REPO/outputs/step2/f0_wema_fulltrace_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/F0_WEMA_FAILED"; exit 1; }

say "=== F0-W-EMA 체인: B0_R1 종료 대기 (cuda:1 확보) ==="
while [ ! -f "$BAT/B0_R1_DONE" ] && [ ! -f "$BAT/B0_R1_FAILED" ]; do sleep 300; done
say "cuda:1 확보 — 시작"

if [ ! -f "$BAT/.f0smoke_wema" ]; then
  say "smoke: W-EMA 12샘플 (cuda:1)"
  SDIR=$BAT/f0smoke_wema; rm -rf "$SDIR"
  $PY scripts/step2/f0_gr_train.py --train_jsonl "$TRAIN_JSONL" --output_dir "$SDIR" \
    --full_trace --reward wm --max_new_tokens 384 --batch_gen 2 \
    --max_samples 12 --accum 4 --log_every 6 --save_every 100000 --device cuda:1 \
    > "$SDIR.log" 2>&1 || die "W-EMA smoke 실행 실패 — $SDIR.log"
  $PY - "$SDIR/gr_log.jsonl" <<'PYEOF' || die "W-EMA smoke 로그 이상"
import json, math, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
assert rows and all(math.isfinite(r["loss"]) for r in rows)
assert any(r["reward_ma"] > 0 for r in rows), "WM reward 전부 0 — likelihood 매칭 이상"
print("[smoke]", rows)
PYEOF
  touch "$BAT/.f0smoke_wema"; say "smoke OK"
else say "smoke skip"; fi

if [ ! -f "$OUT/TRAINING_DONE" ]; then
  rm -rf "$OUT"
  say "F0-W-EMA 학습 (cuda:1, 5000 샘플, batch_gen 4)"
  $PY scripts/step2/f0_gr_train.py --train_jsonl "$TRAIN_JSONL" --output_dir "$OUT" \
    --full_trace --reward wm --max_new_tokens 384 --batch_gen 4 \
    --max_samples 5000 --accum 16 --save_every 1250 --device cuda:1 \
    > "$BAT/train_wema.log" 2>&1 || die "W-EMA 학습 실패 — $BAT/train_wema.log"
  ls "$OUT"/checkpoint-final >/dev/null 2>&1 || die "최종 체크포인트 없음"
  touch "$OUT/TRAINING_DONE"
fi

if [ ! -s "$BAT/f0wema_final.json" ]; then
  $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:1 \
    --adapter "$OUT/checkpoint-final" --out "$BAT/f0wema_final.json" \
    > "$BAT/f0wema_final.log" 2>&1 || die "W-EMA eval 실패"
fi
if [ ! -s "$BAT/swap_f0wema.json" ]; then
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
    --records "$BAT/f0wema_final.records.jsonl" --adapter "$OUT/checkpoint-final" \
    --device cuda:1 --out "$BAT/swap_f0wema.json" > "$BAT/swap_f0wema.log" 2>&1 \
    || die "W-EMA ③ swap 실패"
fi

$PY - "$BAT" > "$BAT/F0_WEMA_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat = Path(sys.argv[1])
def J(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None
print("# F0-W-EMA (full-trace + WM likelihood + EMA, GT-free) 결과 — 자동 생성\n")
print("| 모델 | acc | G2 | wm_follow | parse |")
print("|---|---|---|---|---|")
for name, p in [("base", bat/"base_1f_strict.json"),
                ("F0-W (WM, 그룹 adv)", bat/"f0w_step500.json"),
                ("F0-W-EMA (WM, EMA)", bat/"f0wema_final.json"),
                ("F0-WE (GT, EMA)", bat/"f0we_final.json")]:
    d = J(p)
    if d: print(f"| {name} | {d['acc']} | {d.get('g2_acc')} | {d.get('wm_follow')} | {d.get('parse_rate')} |")
sw = J(bat/"swap_f0wema.json")
if sw: print(f"\n③ W-EMA causal_sensitivity: {sw['causal_sensitivity']}")
we, wm = J(bat/"f0we_final.json"), J(bat/"f0wema_final.json")
print("\n판독 가이드: W-EMA 는 wm_follow↑(복사 수렴)일수록 acc→0.374·G2→0 예상.")
print("WE 와의 비교 = '보상 정의가 목표점을 결정한다' 가설의 직접 검증.")
PYEOF
cat "$BAT/F0_WEMA_RESULTS.md"
touch "$BAT/F0_WEMA_DONE"; say "=== F0-W-EMA 체인 DONE ==="
