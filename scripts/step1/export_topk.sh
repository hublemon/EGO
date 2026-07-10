#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ego step1 infer \
  --config configs/step1/inference.yaml
