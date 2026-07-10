#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ego step2 grpo-noun \
  --config configs/step2/grpo_stage1_noun.yaml
