#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ego step2 grpo-action \
  --config configs/step2/grpo_stage2_action.yaml
