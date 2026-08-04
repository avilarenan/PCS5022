#!/usr/bin/env bash
set -euo pipefail

# Explicitly capped plumbing check; its output is never accuracy evidence.
utility-peft run-time-peft-reproduction \
  --config time_peft_reproduction_smoke \
  --stage all \
  --test-role plumbing-smoke
