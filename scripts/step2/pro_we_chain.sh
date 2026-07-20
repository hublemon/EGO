#!/usr/bin/env bash
# pro_we_chain.sh — F0-WE 확정 후보 run: full-trace(reasoning/belief 유지) + GT/outcome
#   바이너리 + EMA 기준선 REINFORCE (생성 기반, GRPO 그룹-상대 advantage 대체).
# 자동 분기(사전 등록): F0_GA_DONE 후 f0gr_final.acc − f0g_step500.acc ≥ +0.02 일 때만 실행.
#   미달 → F0_WE_SKIPPED + F0_FINAL_W (extro = F0-W 최종 확정, run 불필요).
# 완료 마커: F0_WE_DONE / F0_WE_FAILED / F0_WE_SKIPPED  (b0_full_chain 이 이 마커로 GPU 대기)
set -euo pipefail
export EGO_ROOT="${EGO_ROOT:-/mnt/nvme/migration/jihun/EGO}"
export HF_HOME=/mnt/nvme/cache TRANSFORMERS_CACHE=/mnt/nvme/cache/transformers
export PYTHONIOENCODING=utf-8
[ -f /root/.config/ego/letsur.env ] && source /root/.config/ego/letsur.env
ENVBIN=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin
export PATH="$ENVBIN:$PATH"; PY=$ENVBIN/python
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
BAT=$EGO_ROOT/runs/f0_battery
TRAIN_JSONL=$BAT/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
J1F=$BAT/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
OUT=$REPO/outputs/step2/f0_we_fulltrace_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/F0_WE_FAILED"; exit 1; }

say "=== F0-WE 체인: F0_GA(GR 진단) 마커 대기 ==="
while [ ! -f "$BAT/F0_GA_DONE" ] && [ ! -f "$BAT/F0_GA_FAILED" ]; do sleep 120; done
if [ -f "$BAT/F0_GA_FAILED" ]; then
  say "GR 진단 실패 — 판정 불가, F0-WE 보류"; touch "$BAT/F0_WE_SKIPPED"; exit 0
fi

# ── 사전 등록 판정: GR − F0-G ≥ +0.02 ───────────────────────────────────────
GO=$($PY -c "
import json
gr = json.load(open('$BAT/f0gr_final.json'))['acc']
fg = json.load(open('$BAT/f0g_step500.json'))['acc']
d = gr - fg
print(f'{d:+.4f} {\"GO\" if d >= 0.02 else \"NOGO\"}')" ) || die "판정 입력 없음"
say "판정: GR−F0G = $GO"
if [[ "$GO" == *NOGO* ]]; then
  say "기준 미달 → extro = F0-W 최종 확정 (run 생략)"
  echo "extro 최종 = F0-W (wm_clean). GR 진단 결과 EMA 최적화로도 GT acc 개선 미달($GO) — 용량 상한 서사." \
    > "$BAT/F0_FINAL_W.txt"
  touch "$BAT/F0_WE_SKIPPED"; exit 0
fi

# ── smoke: full-trace 12샘플 ────────────────────────────────────────────────
if [ ! -f "$BAT/.f0smoke_we" ]; then
  say "smoke: F0-WE 12샘플 full-trace (cuda:0)"
  SDIR=$BAT/f0smoke_we; rm -rf "$SDIR"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TRAIN_JSONL" --output_dir "$SDIR" \
    --full_trace --max_new_tokens 384 --batch_gen 2 \
    --max_samples 12 --accum 4 --log_every 6 --save_every 100000 --device cuda:0 \
    > "$SDIR.log" 2>&1 || die "WE smoke 실행 실패 — $SDIR.log"
  $PY - "$SDIR/gr_log.jsonl" <<'PYEOF' || die "WE smoke 로그 이상"
import json, math, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
assert rows and all(math.isfinite(r["loss"]) for r in rows)
print("[smoke]", rows)
PYEOF
  touch "$BAT/.f0smoke_we"; say "smoke OK"
else say "smoke skip"; fi

# ── 학습: 5000 샘플 full-trace REINFORCE-EMA ────────────────────────────────
if [ ! -f "$OUT/TRAINING_DONE" ]; then
  rm -rf "$OUT"
  say "F0-WE 학습 (cuda:0, 5000 샘플, batch_gen 4)"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TRAIN_JSONL" --output_dir "$OUT" \
    --full_trace --max_new_tokens 384 --batch_gen 4 \
    --max_samples 5000 --accum 16 --save_every 1250 --device cuda:0 \
    > "$BAT/train_we.log" 2>&1 || die "F0-WE 학습 실패 — $BAT/train_we.log"
  ls "$OUT"/checkpoint-final >/dev/null 2>&1 || die "최종 체크포인트 없음"
  touch "$OUT/TRAINING_DONE"
fi

# ── 평가: eval_battery(3태그) + ③ swap ──────────────────────────────────────
if [ ! -s "$BAT/f0we_final.json" ]; then
  $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:0 \
    --adapter "$OUT/checkpoint-final" --out "$BAT/f0we_final.json" \
    > "$BAT/f0we_final.log" 2>&1 || die "WE eval 실패"
fi
if [ ! -s "$BAT/swap_f0we.json" ]; then
  $PY scripts/step2/eval_belief_swap.py --jsonl "$J1F" \
    --records "$BAT/f0we_final.records.jsonl" --adapter "$OUT/checkpoint-final" \
    --device cuda:0 --out "$BAT/swap_f0we.json" > "$BAT/swap_f0we.log" 2>&1 \
    || die "WE ③ swap 실패"
fi

# ── 요약 ────────────────────────────────────────────────────────────────────
$PY - "$BAT" > "$BAT/F0_WE_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat = Path(sys.argv[1])
def J(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None
print("# F0-WE (full-trace + GT/outcome + EMA) 확정 후보 결과 — 자동 생성\n")
print("| 모델 | acc | G2 | cond.acc | wm_follow | parse |")
print("|---|---|---|---|---|---|")
for name, p in [("base", bat/"base_1f_strict.json"),
                ("F0-W clean (GT-free)", bat/"f0w_step500.json"),
                ("F0-G (GT, 그룹 adv)", bat/"f0g_step500.json"),
                ("F0-WE (GT, EMA)", bat/"f0we_final.json")]:
    d = J(p)
    if d: print(f"| {name} | {d['acc']} | {d.get('g2_acc')} | {d.get('acc_given_gt_in_top5')} "
                f"| {d.get('wm_follow')} | {d.get('parse_rate')} |")
sw = J(bat/"swap_f0we.json")
if sw: print(f"\n③ F0-WE causal_sensitivity: {sw['causal_sensitivity']} "
             f"(swap {sw['swap_action_change']} / ctrl {sw['control_action_change']})")
we, fw = J(bat/"f0we_final.json"), J(bat/"f0w_step500.json")
if we and fw:
    print(f"\n판정 힌트: WE−W acc {we['acc']-fw['acc']:+.3f}, WE G2 {we.get('g2_acc')} "
          f"(G2 가 상승해야 'WM top-1 을 이기는 학습' 성립)")
print("\n주의: GT 보상 run — 채택 여부·서사(outcome reward 해석)는 사람 리뷰 필요.")
PYEOF
cat "$BAT/F0_WE_RESULTS.md"
touch "$BAT/F0_WE_DONE"; say "=== F0-WE 체인 DONE ==="
