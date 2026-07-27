#!/usr/bin/env bash
# 체인 v3 진행 아티팩트 5분 주기 갱신. 체인이 끝나면 한 번 더 쓰고 스스로 종료한다.
set -uo pipefail
cd /mnt/nvme/migration/jihun/EGO_jihun3
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
for i in $(seq 1 400); do   # 최대 ~33시간
  $PY tools/chain_progress.py >> runs/chain_progress.log 2>&1
  [ -f runs/chain_v3.DONE ] && { $PY tools/chain_progress.py >> runs/chain_progress.log 2>&1; break; }
  sleep 300
done
echo "[refresher $(date '+%F %T')] 종료" >> runs/chain_progress.log
