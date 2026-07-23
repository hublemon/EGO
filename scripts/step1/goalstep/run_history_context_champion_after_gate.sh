#!/usr/bin/env bash
set -Eeuo pipefail

# OBSOLETE: P0-b is diagnostic-only.  Keeping the former filename as a
# fail-closed tombstone prevents an old tmux command or handoff note from
# silently reintroducing the retired 27.7 hard gate.
printf '%s\n' \
  'ERROR: this launcher is obsolete because the P0-b hard gate was retired.' \
  'Use scripts/step1/goalstep/run_history_phase1_after_store.sh instead.' >&2
exit 64
