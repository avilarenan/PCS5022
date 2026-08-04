#!/usr/bin/env bash
set -euo pipefail

# Preliminary/non-confirmatory two-phase screen. Every phase retains all data
# windows and uses h96, seeds 0/1, L/LFC, and LRs 1e-3/1e-4. Give it exclusive
# access to one A40; identical reruns resume atomically cached trials.
common_overrides=(
  -o 'experiment.horizons=[96]'
  -o 'experiment.seeds=[0,1]'
  -o 'experiment.actions=[L,LFC]'
  -o 'experiment.training.learning_rates=[0.001,0.0001]'
)

# Phase 1: the long ECG cell, limited to four complete epochs per trial.
ecg_overrides=(
  "${common_overrides[@]}"
  -o 'paths.artifacts=artifacts/time-peft-budget24/ecg'
  -o 'experiment.datasets=[ECGCA515]'
  -o 'experiment.training.max_epochs=4'
)

for stage in tune test report; do
  utility-peft run-time-peft-reproduction \
    --config time_peft_budget24 \
    --stage "$stage" \
    --test-role development-parity \
    --download \
    "${ecg_overrides[@]}"
done

# Phase 2: the five shorter synthetic cells, with eight complete epochs each.
synthetic_overrides=(
  "${common_overrides[@]}"
  -o 'paths.artifacts=artifacts/time-peft-budget24/synthetic'
  -o 'experiment.datasets=[Lorenz,CellCycle,DoublePendulum,Hopfield,LorenzCoupled]'
  -o 'experiment.training.max_epochs=8'
)

for stage in tune test report; do
  utility-peft run-time-peft-reproduction \
    --config time_peft_budget24 \
    --stage "$stage" \
    --test-role development-parity \
    "${synthetic_overrides[@]}"
done
