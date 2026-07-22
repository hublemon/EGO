#!/usr/bin/env bash
# step2_expc_g1anchor_chain.sh — Exp-C: candidate CE + G1 selective-trust 앵커 (H200 1장)
#
#   setsid nohup bash scripts/step2/step2_expc_g1anchor_chain.sh \
#     > /mnt/nvme/migration/jihun/EGO/runs/expc/launch.log 2>&1 < /dev/null &
#   현황: http://<host>:7866/
#
# ── 목표 ─────────────────────────────────────────────────────────────────────
# Exp-A(순수 candidate CE) 결과: 전체 acc 0.3860, L0(0.3994)를 0.013 못 넘음.
#   G1 보존 0.4806→0.6714 · G2 교정 0.3988→0.5202 (둘 다 프로젝트 최고)
# 남은 갭의 정체는 G1 퇴행이다 — WM 이 맞힌 것 중 아직 33% 를 놓친다.
# G1 에서만 모델 후보분포를 WM prior 로 당겨(selective trust) 그 33% 를 줄인다.
#   참조는 F0 가 아니라 WM 자신 — F0 의 G1 보존이 0.497 이라 F0 앵커는 역효과.
#
# ── 판정 (사전 등록) ─────────────────────────────────────────────────────────
#   성공 = 전체 acc > L0(0.3994)  AND  G2 교정이 Exp-A(0.5202) 대비 −1.5pp 이내 유지
#   부분 = L0 미달이나 G1 보존이 Exp-A(0.6714) 대비 유의 상승
#   실패 = G1 만 오르고 G2 가 무너져 acc 정체 (selective trust 실패)
set -u
EGO_ROOT=/mnt/nvme/migration/jihun/EGO
REPO=/mnt/nvme/migration/jihun/EGO_jihun
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
OUT=$EGO_ROOT/runs/expc
TR=$EGO_ROOT/runs/f0_battery/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl
HO=$EGO_ROOT/runs/f0_battery/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl
BASE_EVAL=$EGO_ROOT/runs/cce/eval_scored_base.json   # Exp-A 에서 만든 대조군 재사용
DEV=cuda:1
LOG=$OUT/chain.log
mkdir -p "$OUT"; cd "$REPO" || exit 1
say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
die() { say "치명적: $*"; touch "$OUT/CHAIN_FAILED"; exit 1; }

KW_LIST=${KW_LIST:-"0.3 1.0"}
MAXS=${MAXS:-5000}
say "===== Exp-C G1 앵커 체인 시작 (GPU $DEV · keep_weight 후보: $KW_LIST) ====="

for KW in $KW_LIST; do
  ADAPT=$REPO/outputs/step2/expc_kw${KW}_1f
  EVJ=$OUT/eval_kw${KW}.json
  if [ ! -d "$ADAPT/checkpoint-final" ]; then
    say "--- 학습 kw=$KW ($MAXS 샘플) ---"
    $PY scripts/step2/pro_gx_train.py --train_jsonl "$TR" --output_dir "$ADAPT" \
      --max_samples "$MAXS" --accum 16 --lr 1e-5 --keep_weight "$KW" --device $DEV \
      --log_every 100 --save_every 100000 > "$OUT/train_kw${KW}.log" 2>&1
    say "  학습 kw=$KW rc=$?"
  fi
  [ -d "$ADAPT/checkpoint-final" ] || die "kw=$KW 체크포인트 없음"
  if [ ! -s "$EVJ" ]; then
    say "--- 평가 kw=$KW (heldout 전량) ---"
    $PY scripts/step2/eval_candidate_scored.py --jsonl "$HO" --adapter "$ADAPT/checkpoint-final" \
      --limit 0 --action_only --device $DEV --out "$EVJ" > "$OUT/eval_kw${KW}.log" 2>&1
    say "  평가 kw=$KW rc=$?"
  fi
done

# ── 판정: L0 를 넘은 것 중 acc 최고, 없으면 G1 보존 최고 ──────────────────────
say "--- 판정 ---"
$PY - "$OUT" "$BASE_EVAL" $KW_LIST <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
out=Path(sys.argv[1]); base=json.load(open(sys.argv[2])); kws=sys.argv[3:]
L0=base["L0_wm_top1"]
print(f"  L0 {L0}  ·  Exp-A: acc 0.3860 G1 0.6714 G2 0.5202  ·  base: acc {base['acc']} G1 {base['g1_retention']} G2 {base['g2_correction']}")
best=None
for kw in kws:
    p=out/f"eval_kw{kw}.json"
    if not p.is_file(): continue
    d=json.load(open(p))
    beat="★L0 초과" if d["acc"]>L0 else ""
    print(f"  kw={kw}: acc {d['acc']} {beat} · G1 {d['g1_retention']} · G2 {d['g2_correction']} · 조건부 {d['conditional_acc']}")
    key=(d["acc"]>L0, d["acc"])
    if best is None or key>best[0]: best=(key, kw, d)
if best:
    _,kw,d=best
    ok = d["acc"]>L0 and d["g2_correction"]>=0.5202-0.015
    print(f"  선택: kw={kw}  →  {'성공 (L0 초과 + G2 보존)' if ok else ('부분 성공' if d['g1_retention']>0.6714 else '실패')}")
    (out/"chosen_kw.txt").write_text(kw)
PY
say "===== Exp-C 종료 ====="; touch "$OUT/CHAIN_DONE"
