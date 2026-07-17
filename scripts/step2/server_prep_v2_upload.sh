#!/usr/bin/env bash
# server_prep_v2_upload.sh — 학습 서버에서 F0 v2 데이터 빌드 + Colab 업로드 패키징 (원커맨드).
#
# 실행 위치: H200 학습 서버 (v1 산출물 ~/work/jihun/EGO/data/grpo_dataset 이 있는 곳).
# GPU 불필요 — V-JEPA2 predictions 는 v1 것을 재사용, 프레임 재추출은 CPU(decord).
#
#   bash scripts/step2/server_prep_v2_upload.sh
#   # 또는 레포 클론 없이:
#   curl -sL https://raw.githubusercontent.com/hublemon/EGO/main/scripts/step2/server_prep_v2_upload.sh | bash
#
# 하는 일:
#   1. hublemon/EGO 최신 클론/풀 (~/EGO_hub) — v2 빌드 스크립트 확보
#   2. 선행물 검사 (selected/predictions train+heldout, EK100 어노테이션, 영상 접근)
#   3. build_f0_v2_data.sh 실행 — 4f grid 프레임 + strict cutoff memory + assemble/convert
#      + 자동 leakage 검사 (PASS 실패 시 여기서 중단 = freeze 게이트)
#   4. 산출물 검증 (라인 수 · frame_meta.n_frames==4 · cutoff_rule strict)
#   5. 업로드 패키징: f0_v2_upload.tgz (grpo_train/heldout/b0meta jsonl + 참조 프레임 전체 + md5)
#   6. rclone 이 있으면 Drive 업로드까지, 없으면 수동 업로드 안내 출력
#
# ⚠ 주의: extract_frame_train 은 frames/{sample_id}.jpg 를 4f grid 로 **덮어쓴다** (v1 1f 프레임 소실).
#   1f-base 비교(3중 비교)를 위해 기존 frames/ 를 frames_1f_backup/ 으로 먼저 백업한다.
set -euo pipefail

SRV_EGO="${SRV_EGO:-$HOME/work/jihun/EGO}"          # 데이터 스크립트들이 하드코딩한 루트
HUB_DIR="${HUB_DIR:-$HOME/EGO_hub}"                  # hublemon/EGO 클론 위치
GD="$SRV_EGO/data/grpo_dataset"
LOG="$SRV_EGO/f0_v2_build.log"
DRIVE_REMOTE="${DRIVE_REMOTE:-}"                     # 예: gdrive:step2_vlm_grpo/v2 (rclone remote)

say() { echo -e "\n=== $* ===" | tee -a "$LOG"; }

say "[0] 환경"
echo "SRV_EGO=$SRV_EGO  HUB_DIR=$HUB_DIR" | tee -a "$LOG"
[ -d "$SRV_EGO" ] || { echo "✗ $SRV_EGO 없음 — SRV_EGO=<경로> 로 재실행"; exit 2; }

say "[1] hublemon/EGO 확보"
if [ -d "$HUB_DIR/.git" ]; then git -C "$HUB_DIR" pull --ff-only; else
  git clone https://github.com/hublemon/EGO.git "$HUB_DIR"; fi
git -C "$HUB_DIR" log --oneline -1 | tee -a "$LOG"

say "[2] 선행물 검사"
MISS=0
for f in selected_train.jsonl selected_heldout.jsonl predictions_train.jsonl predictions_heldout.jsonl; do
  if [ -s "$GD/$f" ]; then echo "  OK   $f ($(wc -l < "$GD/$f") lines)"; else echo "  MISS $f"; MISS=1; fi
done | tee -a "$LOG"
[ -d "$SRV_EGO/src/epic-kitchens-100-annotations" ] && echo "  OK   annotations" || { echo "  MISS annotations"; MISS=1; }
[ "$MISS" = 0 ] || { echo "✗ 선행물 누락 — v1 파이프라인 산출물 위치 확인 필요"; exit 2; }

say "[3] v1 1f 프레임 백업 (3중 비교용)"
if [ -d "$GD/frames" ] && [ ! -d "$GD/frames_1f_backup" ]; then
  cp -a "$GD/frames" "$GD/frames_1f_backup"
  echo "  backup: frames_1f_backup/ ($(ls "$GD/frames_1f_backup" | wc -l) files)" | tee -a "$LOG"
else
  echo "  (backup 존재 또는 frames 없음 — 건너뜀)" | tee -a "$LOG"
fi

say "[4] v2 빌드 (strict cutoff + 4f + leakage 게이트)"
cd "$HUB_DIR"
bash scripts/step2/build_f0_v2_data.sh 2>&1 | tee -a "$LOG"
# build 스크립트가 leakage FAIL 시 비-0 종료 (set -e 로 여기서 중단됨)

say "[5] 산출물 검증"
python3 - "$GD" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
gd = Path(sys.argv[1])
ok = True
for name, lo in [("grpo_train.jsonl", 3000), ("grpo_heldout.jsonl", 300)]:
    p = gd / name
    if not p.exists():
        print(f"  ✗ {name} 없음"); ok = False; continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    fm = rows[0].get("frame_meta") or {}
    nf_ok = all((r.get("frame_meta") or {}).get("n_frames") == 4 for r in rows[:50])
    print(f"  {name}: {len(rows)} lines · n_frames==4 (첫50): {nf_ok}")
    ok = ok and len(rows) >= lo and nf_ok
mem = gd / "memory_train.jsonl"
if mem.exists():
    r0 = json.loads(mem.open().readline())
    rule = (r0.get("memory_context") or r0).get("cutoff_rule", r0.get("cutoff_rule"))
    print(f"  memory cutoff_rule: {rule}")
sys.exit(0 if ok else 3)
PY

say "[6] 업로드 패키징"
cd "$GD"
TRAIN_FRAMES=$(python3 -c "
import json
seen=set()
for f in ['grpo_train.jsonl','grpo_heldout.jsonl']:
    for l in open(f):
        p=json.loads(l)['image_path']
        seen.add(p.split('/frames/')[-1])
print('\n'.join(sorted(seen)))")
echo "$TRAIN_FRAMES" | sed 's|^|frames/|' > upload_frames.txt
tar czf f0_v2_upload.tgz grpo_train.jsonl grpo_heldout.jsonl grpo_train_b0meta.jsonl \
    frames_manifest_train.jsonl -T upload_frames.txt 2>/dev/null || \
tar czf f0_v2_upload.tgz grpo_train.jsonl grpo_heldout.jsonl -T upload_frames.txt
md5sum f0_v2_upload.tgz | tee -a "$LOG"
du -h f0_v2_upload.tgz | tee -a "$LOG"

say "[7] 업로드"
if [ -n "$DRIVE_REMOTE" ] && command -v rclone >/dev/null; then
  rclone copy f0_v2_upload.tgz "$DRIVE_REMOTE" --progress
  echo "  rclone 완료 → $DRIVE_REMOTE/f0_v2_upload.tgz" | tee -a "$LOG"
else
  cat <<EOF | tee -a "$LOG"
  rclone 미설정 — 수동 업로드:
    파일: $GD/f0_v2_upload.tgz
    방법 A) 로컬로 가져와 Drive 웹 업로드:  scp <서버>:$GD/f0_v2_upload.tgz .
    방법 B) 서버에 rclone 설정 후:  DRIVE_REMOTE=gdrive:step2_vlm_grpo/v2 bash $0
  Colab 에서 풀기:
    !cd /content/drive/MyDrive/step2_vlm_grpo && mkdir -p v2 && tar xzf f0_v2_upload.tgz -C v2
  Colab path_map:
    --path_map "$SRV_EGO/data/grpo_dataset/frames=/content/drive/MyDrive/step2_vlm_grpo/v2/frames"
EOF
fi

say "[DONE] 로그: $LOG"
