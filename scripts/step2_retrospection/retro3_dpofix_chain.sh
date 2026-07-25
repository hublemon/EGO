#!/usr/bin/env bash
# retro3 수정판 DPO 우선 체인 (2026-07-24 사용자 확정 — GPU 순번 재편):
#   개입③(base) 완료 대기 → 개입③(r1_sft) → AO pair 재생성 →
#   DPO(dpo_d1_fix: AO+fieldwise) → 배터리·개입 → RETRO3_CHAIN_DONE → retro4 인계.
# dpo가 G3로 또 abort해도 marker note만 남기고 retro4 인계는 진행한다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH="$REPO/src" HF_HOME=/mnt/nvme/cache TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RETRO3_RUNS=runs/retro3
LOG=runs/retro3/logs/chain.log
MK=runs/retro3/markers
ADAPT=outputs/step2_retrospection

run() { local mk="$1"; shift; [[ -f "$MK/$mk" ]] && { echo "[dpofix] SKIP $mk"; return 0; }
  echo "[dpofix] ==== $* ====" >> "$LOG"; "$@" >> "$LOG" 2>&1 || true
  [[ -f "$MK/$mk" ]] && echo "[dpofix] OK $mk" || echo "[dpofix] FAIL $mk (계속)"; }

# 돌고 있는 개입③(base) 완료 대기 (최대 1h)
for i in $(seq 1 60); do
  [[ -f "$MK/S7_INTERVENTION_BASE_DONE" ]] && break
  pgrep -f "eval.intervention" > /dev/null || break
  sleep 60
done

run S7_INTERVENTION_BASE_DONE   "$PY" -m ego.step2_retrospection.eval.intervention --arm base --n 300
run S7_INTERVENTION_R1_SFT_DONE "$PY" -m ego.step2_retrospection.eval.intervention --arm r1_sft --adapter "$ADAPT/r1_sft/adapter" --n 300

# G3-abort 잔재 보존 후, AO 증강 pair로 재구성
if [[ -d "$ADAPT/dpo_d1" && ! -d "$ADAPT/dpo_d1_g3abort" ]]; then
  mv "$ADAPT/dpo_d1" "$ADAPT/dpo_d1_g3abort"
  mv runs/retro3/probe/dpo_d1.jsonl runs/retro3/probe/archive/dpo_d1_g3abort.jsonl 2>/dev/null || true
fi
rm -f "$MK/S5_PAIRS_DONE"
run S5_PAIRS_DONE "$PY" -m ego.step2_retrospection.pairs.build_pairs --action_only_aug

run S6_DPO_D1_FIX_DONE "$PY" -m ego.step2_retrospection.train.dpo_fb \
  --run_name dpo_d1_fix --loss_mode fieldwise

if [[ -f "$MK/S6_DPO_D1_FIX_DONE" ]]; then
  run S7_EVAL_DPO_D1_FIX_DONE "$PY" -m ego.step2_retrospection.eval.battery \
    --arm dpo_d1_fix --adapter "$ADAPT/dpo_d1_fix/adapter" --eval_n 1000
  run S7_INTERVENTION_DPO_D1_FIX_DONE "$PY" -m ego.step2_retrospection.eval.intervention \
    --arm dpo_d1_fix --adapter "$ADAPT/dpo_d1_fix/adapter" --n 300
  NOTE="dpofix 완주"
else
  NOTE="dpo_d1_fix 미완(G3 재발 가능성) — 로그 확인"
fi

echo "{\"ts\": $(date +%s), \"partial\": \"$NOTE — base/r1_sft/dpo_d1_fix 평가 후 retro4 인계\"}" \
  > "$MK/RETRO3_CHAIN_DONE"
echo "[dpofix] 완료 — retro4로 GPU 인계 ($NOTE)"
