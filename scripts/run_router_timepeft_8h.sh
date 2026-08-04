#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

runner="$repo_root/.venv/bin/utility-peft"
log_dir="$repo_root/artifacts/router-timepeft-8h/logs"
mkdir -p "$log_dir"
exec >>"$log_dir/router-timepeft-8h.log" 2>&1

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') preparing leakage-safe episodes"
"$runner" prepare-data --config router_timepeft_8h --download

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') training provenance-bound Electricity source head"
"$runner" train-source-head --config router_timepeft_8h --download

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') running matched L/LF/LC/LFC LODO benchmark"
"$runner" run-correlation-benchmark --config router_timepeft_8h

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') primary router-Time-PEFT experiment complete"

# A longer-adaptation sensitivity follows the primary result. It reuses only the
# provenance-bound source head, regenerates its own immutable episodes/utilities,
# and writes to a separate artifact root. If it is interrupted, the completed
# 100-update primary report remains available and this stage resumes by config hash.
sensitivity_overrides=(
  -o 'paths.artifacts=artifacts/router-timepeft-8h-u300'
  -o 'model.source_head_checkpoint=artifacts/router-timepeft-8h/checkpoints/source-head-h96.pt'
  -o 'experiment.seeds=[0,1]'
  -o 'experiment.update_steps=300'
)

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') preparing 300-update sensitivity episodes"
"$runner" prepare-data --config router_timepeft_8h --download "${sensitivity_overrides[@]}"

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') running 300-update sensitivity benchmark"
"$runner" run-correlation-benchmark --config router_timepeft_8h \
  "${sensitivity_overrides[@]}"

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') all router-Time-PEFT experiments complete"
