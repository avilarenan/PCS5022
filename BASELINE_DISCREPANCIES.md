# Baseline Discrepancies

## MOMENT long-horizon forecasting head

Pinned checkpoint: `AutonLab/MOMENT-1-base@5e44b0ea26376a176360f87831124e018f876d96`.

The official foundation checkpoint pretrains the reconstruction head only. MOMENT's
long-horizon forecasting path constructs a new linear head for each lookback/horizon
shape, so the checkpoint does not provide a meaningful frozen long-horizon A0 model.
This matters because an untrained A0 head artificially inflates the apparent utility
of every action that trains the head.

The implementation therefore supports two explicit modes:

1. Production utility generation supplies a versioned source forecasting-head
   checkpoint through `model.source_head_checkpoint`.
2. Pilot and interface smoke tests may set `model.allow_random_head=true`. Evidence
   and model metadata then include `source_random_forecasting_head=1`, and such
   results must not be reported as reproduced forecasting baselines.

The A40 pilot trains shared horizon-specific heads on ETTm2 train timestamps and
selects checkpoints on ETTm2 validation timestamps. ETTm2 is excluded from every
evaluation fold. A per-channel standardizer is fitted only on the ETTm2 train split
and applied to source train/validation windows. Checkpoint metadata binds the file
hash, source-data hash, scaler-statistics hash, preprocessing specification, split
bounds, horizon, and complete evaluation-dataset exclusion set; loading fails if
the active provenance checks disagree with the pilot configuration.

## Time-PEFT adapters

The cited Time-PEFT work exposes no public implementation at the time of this build.
Frequency and channel adapters are isolated project-owned modules. Both use residual
zero-impact initialization and the proposal defaults (top-frequency fraction 1/4 and
bottleneck width `d_model/8`). They require numerical parity checks against any
subsequently released official implementation before reproduction claims are made.

The complexity-only controller uses normalized spectral entropy and a deterministic
linear-Gaussian lagged transfer-entropy estimator computed from support windows only.
This is an auditable proxy for the paper's inter-channel information-flow statistic,
not evidence of numerical parity with an unavailable author implementation.

`reproduce-time-peft --protocol matched` writes an explicit parity manifest. While
`verified=false`, generated reports use the label `Time-PEFT-style` and include a
hard claim guard. `--protocol paper` is blocked until an official runner and exact
accepted-paper configuration are available and pinned.
