#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ego step3 run \
  --config configs/step3/planning_eval.yaml
