#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ego step2 build-data \
  --config configs/step2/sft_qwen3vl.yaml
