# PCS5022 evolved bundle

Base repository: `avilarenan/PCS5022`, commit
`3f30fa9284e04268fb527689252b717b276b15fe` (`main`). This evolved snapshot is
designed to run both from the downloadable bundle and from the repository.

## Added in this bundle

- repaired the missing `utility_peft.data` package and anchored `/data/` ignore rule;
- pinned manifest/loaders for all 13 Time-PEFT datasets plus source-only Electricity;
- an Algorithm 1-faithful `LFC` baseline, explicit `LF`/`LC` routing ablations,
  and Q/K/V LoRA settings isolated from the historical A2-A5 implementation;
- one-forward, support-only residual correlation/autocorrelation evidence;
- source-only frequency/channel gates and leave-one-dataset-out comparison with `LFC`;
- complete accuracy, time, parameter, memory, FLOP, route, and analytical-cost reports;
- CPU smoke, five-dataset GPU pilot, and full-suite configs/scripts;
- `AGENTS.md`, Codex/VS Code handoff, experimental protocol, compute derivation, CI,
  and new regression tests.

## Verification performed while building the bundle

- Ruff: `ruff check src tests scripts` passed.
- Pytest: 79 non-GPU tests passed.
- CPU end-to-end smoke: 24 matched action records and LODO report generated.
- CPU evidence microbenchmarks: current lag-1 Gaussian TE proxy was 4.42x slower
  than the complete lag-8 correlation suite at `[B,C,T]=[64,21,96]`, and 13.44x
  slower at `[32,64,96]` on this host.
- Dataset implementation was checked against real downloads for all six standard
  datasets and both PhysioNet EDF records, including pinned hashes and shapes.

After the Algorithm 1 architecture correction, the tiny smoke's mean paired/unit
routed MSE was 0.02% higher than `LFC` and used 5.09% fewer active trainable
parameters. Its end-to-end time was 91.66% higher because evidence and noisy
two-update arm timings dominate this CPU-scale run. This is a plumbing result, not
evidence for or against the 100-update GPU hypothesis.

## Start here

1. Read `AGENTS.md` and `docs/CODEX_HANDOFF.md`.
2. Run `./scripts/setup_env.sh` with Python 3.11.
3. Run `./scripts/run_cpu_smoke.sh`.
4. On the GPU machine, run `./scripts/run_correlation_pilot.sh` before considering
   `./scripts/run_correlation_full.sh`.

No official Time-PEFT code was available when this bundle was prepared. Retain the
label **paper-specified Time-PEFT reimplementation** and the limitations in
`BASELINE_DISCREPANCIES.md`.
