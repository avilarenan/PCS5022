#!/usr/bin/env bash
set -euo pipefail

# Operational handoff for an already-running ECG phase. This watcher owns a
# single-instance lock, waits for that exact worker to finish, and then starts
# only the synthetic phase. It never signals or restarts the ECG worker.
if [[ $# -ne 1 || ! $1 =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 ECG_WORKER_PID" >&2
  exit 2
fi

ecg_pid=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

mkdir -p artifacts/time-peft-budget24/logs
exec 9>artifacts/time-peft-budget24/.synthetic-followon.lock
if ! flock -n 9; then
  echo "A Time-PEFT synthetic follow-on watcher or worker already owns the lock."
  exit 0
fi

exec >>artifacts/time-peft-budget24/logs/budget24.log 2>&1
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') follow-on watcher waiting for ECG PID $ecg_pid"

is_expected_ecg_worker() {
  [[ -r /proc/$ecg_pid/cmdline ]] || return 1
  local command_line
  command_line=$(tr '\0' ' ' </proc/$ecg_pid/cmdline)
  [[ $command_line == *"utility-peft run-time-peft-reproduction"* ]]
  [[ $command_line == *"paths.artifacts=artifacts/time-peft-budget24/ecg"* ]]
  [[ $command_line == *"experiment.datasets=[ECGCA515]"* ]]
}

while is_expected_ecg_worker; do
  sleep 30
done

if pgrep -f \
  'utility-peft run-time-peft-reproduction.*paths.artifacts=artifacts/time-peft-budget24/synthetic' \
  >/dev/null; then
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') synthetic phase is already running; exiting"
  exit 0
fi

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') ECG worker ended; starting synthetic phase"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "$repo_root/.venv/bin/utility-peft" run-time-peft-reproduction \
  --config time_peft_budget24 \
  --stage all \
  --test-role development-parity \
  -o 'paths.artifacts=artifacts/time-peft-budget24/synthetic' \
  -o 'experiment.datasets=[Lorenz,CellCycle,DoublePendulum,Hopfield,LorenzCoupled]' \
  -o 'experiment.training.max_epochs=8'
