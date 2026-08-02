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

## Paper-specified correlation benchmark

The newer `L/LF/LC/LFC` workflow is separate from the historical A2-A5 MVP. It
implements the accepted-paper defaults that can be recovered from the text: rank-8,
alpha-32 LoRA on Q/K/V; top-3 FFT filtering with `h2=h1`; and a shared-down,
channel-specific-up adapter with `r=h1/2`. `LFC` is the matched always-on baseline.

Algorithm 1 makes the full path explicit: `Efilt=Freq(Eback)`,
`Ech=Channel(Eback, Efilt)`, then `ForecastHead(LayerNorm(Ech))`. The `LFC` arm
implements that dataflow without a residual bypass. The smaller routing arms are
factorial extensions motivated by Table 4: `LF` uses the paper's F-both streams with
an explicitly chosen normalized sum, while `LC` supplies zeros in the absent
frequency slot and retains the matched channel-adapter capacity.

Other details remain under-specified without author code: frequency-amplitude
aggregation, adapter initialization, the head-side fusion operator for channel-free
F-both, LayerNorm affine parameters, per-dataset learning rates and batches, patience,
exact splits/preprocessing, random seeds, and chaotic-trajectory generation. This
repository uses per-sample/channel frequency scores averaged over hidden dimensions,
Xavier adapter initialization, and non-affine output LayerNorm. These explicit choices
and the two routing-only ablations prevent an exact reproduction claim.

The paper's spectral-entropy and transfer-entropy quantities are dataset diagnostics;
Algorithm 1 does not use them to select adapters. Residual-correlation activation is a
new outer router. Its deployment cost includes its frozen support forecast and evidence
pass, while the always-on `LFC` baseline is not charged for transfer-entropy extraction.

The five local chaotic generators are deterministic compatible substitutes, not exact
paper trajectories. GPU results must retain the label **paper-specified Time-PEFT
reimplementation** until an authoritative release enables numerical parity checks.
