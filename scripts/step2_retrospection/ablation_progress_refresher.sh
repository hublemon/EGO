#!/usr/bin/env bash
# progress.json / progress.html 을 3분마다 재생성한다 (세션 독립).
# 러너는 단계 전환 시에만 갱신하는데 단계가 7~17분이라, 중간 경과가 반영되도록 별도로 돈다.
# 큐가 끝나면 자동 종료. 최대 12시간.
cd /mnt/nvme/migration/jihun/EGO_jihun3
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
for i in $(seq 1 240); do
  $PY tools/ablation_progress.py >/dev/null 2>&1
  grep -q '"item": "queue", "event": "done"' runs/ablation_v2/timeline.jsonl 2>/dev/null && break
  pgrep -f run_ablations_v2.sh >/dev/null || pgrep -f run_planB.sh >/dev/null || break
  sleep 180
done
$PY tools/ablation_progress.py >/dev/null 2>&1
