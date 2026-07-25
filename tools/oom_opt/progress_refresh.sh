#!/usr/bin/env bash
# 진행 HTML 을 주기 재생성 (관측 전용, stdlib python 만 — GPU/모델 로드 없음).
# optim_progress.html 은 30초 meta-refresh 라, 이 루프가 그 파일을 갱신하면 브라우저가 자동 반영.
set -uo pipefail
REPO="${REPO:-/mnt/nvme/migration/jihun/EGO_jihun3}"
cd "$REPO"
PY="${PY:-/opt/conda/bin/python3}"
INTERVAL="${INTERVAL:-30}"
while true; do
  "$PY" tools/oom_opt/render_progress.py "$REPO/runs/cesft_v2" >/dev/null 2>>"$REPO/runs/cesft_v2/logs/render.log" || true
  # 체인 완료되면 한 번 더 렌더 후 종료
  if [ -f "$REPO/runs/cesft_v2/markers/CESFT_V2_CHAIN_DONE" ]; then
    "$PY" tools/oom_opt/render_progress.py "$REPO/runs/cesft_v2" >/dev/null 2>&1 || true
    break
  fi
  sleep "$INTERVAL"
done
