#!/usr/bin/env bash
# pro_ga_chain.sh — F0-GR 진단: action-only 생성 기반 REINFORCE + EMA 기준선.
#   경위: (1) action-only GRPO 는 8롤아웃 전원 동일(그룹 분산 0)로 무학습 ×2회.
#         (2) exact-CE(후보 스코어링)는 "생성으로 행동 선택" 전제와 불일치로 설계 기각.
#   → 생성은 유지하고 기준선만 그룹→EMA 로 교체한 REINFORCE 로 진단 수행.
#   마커는 F0_GA_DONE/FAILED 유지 (retro_r1_chain 이 이 마커로 cuda:0 대기).
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
OUT=$REPO/outputs/step2/f0_gr_actiononly_1f
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "✗ $*"; touch "$BAT/F0_GA_FAILED"; exit 1; }

# ── smoke: 32샘플 — loss 유한 + advantage 실제 발생 확인 ────────────────────
if [ ! -f "$BAT/.f0smoke_gr" ]; then
  say "smoke: F0-GR 32샘플 (cuda:0)"
  SDIR=$BAT/f0smoke_gr; rm -rf "$SDIR"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TRAIN_JSONL" --output_dir "$SDIR" \
    --max_samples 32 --accum 8 --log_every 16 --save_every 100000 --device cuda:0 \
    > "$SDIR.log" 2>&1 || die "GR smoke 실행 실패 — $SDIR.log"
  $PY - "$SDIR/gr_log.jsonl" <<'PYEOF' || die "GR smoke 로그 이상"
import json, math, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
assert rows, "로그 없음"
assert all(math.isfinite(r["loss"]) for r in rows), "loss 비정상"
assert any(r["mean_abs_adv"] > 0.05 for r in rows), "advantage 소멸 — 기준선 이상"
print("[smoke]", rows)
PYEOF
  touch "$BAT/.f0smoke_gr"; say "smoke OK"
else say "smoke skip"; fi

# ── 학습: 7000 샘플 REINFORCE-EMA ───────────────────────────────────────────
if [ ! -f "$OUT/TRAINING_DONE" ]; then
  rm -rf "$OUT"
  say "F0-GR 학습 (cuda:0, 7000 샘플)"
  $PY scripts/step2/pro_gr_train.py --train_jsonl "$TRAIN_JSONL" --output_dir "$OUT" \
    --max_samples 7000 --accum 16 --device cuda:0 \
    > "$BAT/train_gr.log" 2>&1 || die "F0-GR 학습 실패 — $BAT/train_gr.log"
  ls "$OUT"/checkpoint-final >/dev/null 2>&1 || die "최종 체크포인트 없음"
  touch "$OUT/TRAINING_DONE"
fi

# ── 평가: base(action-only 프롬프트) + GR final ─────────────────────────────
if [ ! -s "$BAT/base_actiononly.json" ]; then
  $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:0 --action_only \
    --max_new_tokens 64 --out "$BAT/base_actiononly.json" \
    > "$BAT/base_actiononly.log" 2>&1 || die "base action-only eval 실패"
fi
if [ ! -s "$BAT/f0gr_final.json" ]; then
  $PY scripts/step2/eval_battery.py --jsonl "$J1F" --device cuda:0 --action_only \
    --max_new_tokens 64 --adapter "$OUT/checkpoint-final" \
    --out "$BAT/f0gr_final.json" > "$BAT/f0gr_final.log" 2>&1 || die "GR eval 실패"
fi

# ── 요약 ────────────────────────────────────────────────────────────────────
$PY - "$BAT" "$OUT" > "$BAT/F0_GA_RESULTS.md" <<'PYEOF'
import json, sys
from pathlib import Path
bat, out = Path(sys.argv[1]), Path(sys.argv[2])
def J(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None
print("# F0-GR (action-only REINFORCE-EMA 진단) 결과 — 자동 생성\n")
print("> 경위: action-only GRPO 는 롤아웃 전원 동일(그룹 분산 0)로 무학습 — '탐색은")
print("> reasoning 텍스트에 있었다'는 실측. 후보 스코어링(exact-CE)은 생성-행동 전제와")
print("> 불일치로 설계 기각. 본 진단은 생성 유지 + 기준선만 그룹→보상 EMA 로 교체.\n")
print("| 모델 | acc | cond.acc(GT∈top5) | wm_follow | parse |")
print("|---|---|---|---|---|")
for name, p in [("base (3태그 프롬프트)", bat/"base_1f_strict.json"),
                ("base (action-only 프롬프트)", bat/"base_actiononly.json"),
                ("F0-G step500 (full-trace GRPO+GT)", bat/"f0g_step500.json"),
                ("F0-GR final (action-only REINFORCE+GT)", bat/"f0gr_final.json")]:
    d = J(p)
    if d: print(f"| {name} | {d['acc']} | {d.get('acc_given_gt_in_top5')} "
                f"| {d.get('wm_follow')} | {d.get('parse_rate')} |")
gl = out/"gr_log.jsonl"
if gl.exists():
    rows = [json.loads(l) for l in open(gl)]
    if rows:
        print(f"\ntrain 궤적: reward_ma {rows[0]['reward_ma']}→{rows[-1]['reward_ma']} · "
              f"baseline {rows[-1]['baseline']} ({rows[-1]['seen']}샘플)")
gr, fg = J(bat/"f0gr_final.json"), J(bat/"f0g_step500.json")
if gr and fg:
    d = gr["acc"] - fg["acc"]
    print(f"\n판정: F0-GR − F0-G = {d:+.3f}")
    print("- ≥ +0.02 → 병목 = 그룹-상대 advantage/trace 경로 → F0-W 를 EMA 기준선으로 재편할 가치")
    print("- < +0.02 → 병목 = 모델/데이터 상한 → extro 는 F0-W 로 최종 확정 (서사 완결)")
PYEOF
cat "$BAT/F0_GA_RESULTS.md"
touch "$BAT/F0_GA_DONE"; say "=== F0-GR 체인 DONE ==="
