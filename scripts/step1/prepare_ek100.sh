#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ego step1 prepare \
  --config configs/step1/ek100_vjepa2.yaml
