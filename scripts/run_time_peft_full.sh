#!/usr/bin/env bash
set -euo pipefail

# Do not launch this script until the primary configuration, required
# sensitivities, implementation hash, splits, and decision rules are archived.
# The test stage is a one-time read of the locked seven-dataset matrix.
utility-peft run-time-peft-reproduction \
  --config time_peft_reproduction \
  --stage tune \
  --test-role confirmatory \
  --download

utility-peft run-time-peft-reproduction \
  --config time_peft_reproduction \
  --stage test \
  --test-role confirmatory \
  --download

utility-peft run-time-peft-reproduction \
  --config time_peft_reproduction \
  --stage report \
  --test-role confirmatory
