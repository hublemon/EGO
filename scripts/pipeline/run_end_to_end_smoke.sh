#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ego pipeline smoke-test \
  --config configs/pipeline/ego_end_to_end.yaml
