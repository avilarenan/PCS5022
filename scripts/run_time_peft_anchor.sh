#!/usr/bin/env bash
set -euo pipefail

anchor_overrides=(
  -o 'experiment.datasets=[ECGCA515]'
  -o 'experiment.horizons=[96]'
  -o 'claim.test_role=development-parity'
)

# ECGCA515-h96 is an uncapped development/parity anchor, not confirmatory
# evidence. Completed method/LR/seed trials are cached atomically and reused.

utility-peft run-time-peft-reproduction \
  --config time_peft_reproduction \
  --stage tune \
  --test-role development-parity \
  "${anchor_overrides[@]}"

utility-peft run-time-peft-reproduction \
  --config time_peft_reproduction \
  --stage test \
  --test-role development-parity \
  "${anchor_overrides[@]}"

utility-peft run-time-peft-reproduction \
  --config time_peft_reproduction \
  --stage report \
  --test-role development-parity \
  "${anchor_overrides[@]}"
