# Utility-PEFT

Utility-PEFT is a research proof of concept for selecting how a time-series
foundation model should be adapted to a new forecasting episode. Instead of
assuming that every target needs the same adapter bank, it predicts the
counterfactual gain and cost of each candidate adaptation action from support
data only.

The current implementation uses MOMENT-base and evaluates frozen inference,
head tuning, LoRA, frequency and channel adapters, FourierFT, and full fine-tuning.
The original LaTeX proposal is preserved under
[`utility_peft_latex_project/`](utility_peft_latex_project/).

> **Research status: POC / forecasting MVP.** The end-to-end pipeline and first
> A40 pilot are complete. The oracle heterogeneity gate passed, but the learned
> selector did not beat the best fixed source action. These results do not show
> that Utility-PEFT surpasses Time-PEFT. The next research direction is a cheaper,
> model-aware measure of residual cross-channel correlation nonstationarity.

## Research Question

Time-series complexity describes the data. Utility-PEFT asks a different
question:

> Given a support episode, a frozen model, and a candidate adaptation action,
> how much query performance should that action gain, and what will it cost?

The action-conditioned controller uses three classes of support-only evidence:

1. Time-series structure, including spectral and multichannel descriptors.
2. Frozen-model mismatch on labeled support windows.
3. One-backward-pass gradient responsiveness.

It predicts normalized performance gain for each feasible action. Measured
parameter, FLOP, memory, and runtime costs are applied at selection time, so the
same utility labels can be reused under different deployment budgets.

The central hypothesis is that model-aware evidence should route adaptation more
effectively than a dataset-complexity score or one globally fixed action. The
first pilot supports action heterogeneity but does not yet support successful
cross-dataset routing.

## Current State

| Component | State |
| --- | --- |
| Python package, configs, CLI, and tests | Implemented |
| MOMENT-base integration and source forecasting heads | Implemented |
| Actions A0-A7 and exact state restoration | Implemented |
| Resume-safe utility generation and oracle gate | Implemented |
| Action-conditioned controller and LODO evaluation | Implemented |
| Four-dataset A40 pilot | Completed |
| H1: heterogeneous action utility | Passed |
| H2: model-aware evidence beats complexity-only routing | Not supported |
| H3: selective adaptation surpasses the matched baseline | Not supported |
| H4: selector transfers across held-out datasets | Not supported |
| Official Time-PEFT parity | Blocked: no public official code located |
| Residual correlation nonstationarity | Proposed next experiment |

The implementation enforces support/query separation with immutable episode
manifests. Descriptor extraction, normalization, controller inputs, and action
selection accept support data only. Query labels are reserved for offline utility
generation and final held-out evaluation.

## Pilot Results

The completed A40 pilot used Lorenz, ETTh1, Weather, and Exchange; horizons 96
and 336; three chronological episodes per dataset/horizon; support size 64;
query size 128; and seeds 0, 1, and 2. All trainable actions received exactly 100
updates. ETTm2 was used only for source-head training and was excluded from every
evaluation fold.

The run produced 512 successful immutable utility records:

- 504 records for A0-A6 across 24 episode groups and three seeds.
- 8 stratified A7 full-tuning references.
- 24 additional matched-baseline records were generated in the separate parity
  store for Lorenz and Weather.

| Result | Value |
| --- | ---: |
| Source-head validation MSE, horizon 96 | 0.132700 |
| Source-head validation MSE, horizon 336 | 0.229462 |
| Specialized winning families | LoRA, channel, frequency+channel, FourierFT |
| Best fixed action | A0, frozen/no-op |
| Mean paired oracle regret of best fixed action | 0.097327 |
| Oracle-regret 95% bootstrap interval | [0.062488, 0.142059] |
| Full controller mean LODO NDCG | 0.872973 |
| Complexity-only mean LODO NDCG | 0.808269 |
| Random-routing mean LODO NDCG | 0.726115 |
| Full controller mean oracle regret | 0.282610 |
| Best source-fixed mean oracle regret | 0.097327 |
| Time-PEFT-style A5 mean oracle regret | 0.568695 |
| Full-minus-complexity NDCG | 0.064704 |
| NDCG difference 95% interval | [0.000086, 0.128929] |
| Relative MSE versus Time-PEFT-style A5 | -2.67% |
| Relative-MSE 95% interval | [-16.33%, 14.05%] |
| End-to-end time reduction versus A5 | 77.24% |

The oracle result confirms that no trainable action is uniformly best. However,
the full controller's higher ranking quality did not translate into lower oracle
regret than the complexity-only or source-fixed baselines. The MSE interval
against A5 crosses zero, so neither accuracy superiority nor the preregistered
Pareto claim is established.

See [`reports/mvp_report.md`](reports/mvp_report.md) for the generated report and
[`reports/utility_summary.csv`](reports/utility_summary.csv) for the complete
dataset/horizon/action table.

## Next Direction: Residual Correlation Nonstationarity

The current Time-PEFT-inspired complexity baseline combines normalized spectral
entropy with a support-only linear-Gaussian transfer-entropy proxy. That proxy is
auditable but computationally expensive because it fits directed regressions for
every channel pair. It also describes the dataset rather than what the frozen
model failed to learn.

Plain mean Pearson correlation is cheap, but it is not sufficient by itself.
High correlation can indicate exploitable channel structure or simple redundancy,
and a single average discards lag, topology, direction, and regime changes.

The proposed next signal is **residual correlation nonstationarity (RCN)**. For
support prediction residuals

```text
E = frozen_prediction(X_support) - Y_support
```

compute a channel-correlation matrix for each support window, then summarize how
its off-diagonal entries vary across windows:

```text
RCN = mean_{i != j} std_window(abs(corr(E_window,i, E_window,j)))
```

Companion features should include mean absolute residual correlation, signed
correlation nonstationarity, residual-correlation effective rank, and maximum
lagged residual correlation. These are vectorizable matrix operations with no
pairwise regressions and no additional model backward pass.

RCN is deliberately model-aware. Stable correlated residuals indicate persistent
cross-channel structure that a channel adapter may exploit. Highly unstable
residual relationships may instead indicate that a static channel adapter should
be avoided. Utility-PEFT learns this direction from action outcomes rather than
assuming that a larger complexity value always implies more adaptation.

### Exploratory Screen

A post-hoc screen reused the completed utility table and defined the marginal
channel contribution as:

```text
delta_channel = 0.5 * ((gain(A4) - gain(A2)) + (gain(A5) - gain(A3)))
```

Across the 24 independent episode groups, after averaging action seeds:

| Support descriptor | Spearman association with `delta_channel` |
| --- | ---: |
| Mean absolute input correlation | approximately 0.30 |
| Maximum lagged input correlation | 0.38 |
| Input-correlation nonstationarity | -0.53 |
| Current transfer-entropy mean | 0.52 |

This is exploratory evidence, not a confirmatory result. It is small, affected by
dataset-level confounding, and uses input rather than residual nonstationarity.
It does show that correlation dynamics retain information lost by mean
correlation. A local implementation diagnostic also reduced the warmed Weather
correlation suite to milliseconds, while the current pairwise transfer-entropy
path remained orders of magnitude slower. A formal benchmark must be added before
making an efficiency claim.

### RCN Experiment Plan

1. Add an immutable episode-level evidence cache keyed by episode ID,
   preprocessing hash, model revision, source-head hash, and evidence version.
2. Implement raw, lagged, residual, effective-rank, and nonstationarity correlation
   features using support tensors only.
3. Add equal-capacity `te`, `correlation_raw`, `correlation_dynamic`,
   `correlation_residual`, `te_plus_correlation`, and `full` controller ablations.
4. Reuse the existing 512 action records. Recompute evidence only; do not mutate
   or regenerate utility labels.
5. Evaluate `delta_channel` prediction, LODO NDCG, oracle regret, top-k action
   accuracy, negative-adaptation rate, and evidence extraction time.
6. Use nested source-dataset checkpoint selection and paired stratified bootstrap
   intervals. Target-query data must remain inaccessible to every evidence path.
7. If the reuse-only screen is promising, add new chronological episodes and run
   the targeted pairs A2/A4 and A3/A5 before repeating the full action sweep.

The confirmatory criterion should be preregistered before the new run. The primary
claim requires lower held-out regret than transfer entropy with a paired interval
excluding zero. A weaker efficiency result can be reported if correlation is
NDCG-noninferior within 0.01 while extracting evidence at least 10 times faster.

## Foundation Model and Baseline Boundary

The MVP pins the official
[`AutonLab/MOMENT-1-base`](https://huggingface.co/AutonLab/MOMENT-1-base)
checkpoint at revision `5e44b0ea26376a176360f87831124e018f876d96`.
`momentfm` is installed from official commit
`38f7310ad594100747ca2a8357e9c7ca7d323e0e` (package version 0.1.5),
Transformers is pinned to 4.54.1, and PEFT is pinned to 0.17.1.

MOMENT's foundation checkpoint contains a reconstruction head, not a pretrained
long-horizon forecasting head. The pilot therefore trains horizon-specific heads
on ETTm2 and binds each checkpoint to source-data, preprocessing, split, and
evaluation-exclusion hashes. Random forecasting heads are allowed only in the
synthetic smoke path.

As of July 19, 2026, no public official Time-PEFT implementation was found on the
paper pages, author repositories, KAIST DMLab organization, or Hugging Face. The
project-owned frequency/channel implementation must therefore be called
**Time-PEFT-style**. Reports contain a machine-readable claim guard and cannot
claim to reproduce or surpass the published method. See
[`BASELINE_DISCREPANCIES.md`](BASELINE_DISCREPANCIES.md), the
[`ICML paper page`](https://icml.cc/virtual/2026/poster/61767), and the
[`OpenReview entry`](https://openreview.net/forum?id=n8seTOinYs).

## Actions

| ID | Trainable operation |
| --- | --- |
| A0 | Frozen/no-op |
| A1 | Forecast head |
| A2 | Head + LoRA |
| A3 | Head + LoRA + frequency adapter |
| A4 | Head + LoRA + channel adapter |
| A5 | Head + LoRA + frequency + channel adapters |
| A6 | Head + FourierFT |
| A7 | Full backbone fine-tuning, reference only |

LoRA uses rank 8, alpha 16, zero dropout, and the query/value projections in
every MOMENT T5 encoder block. Project-owned frequency and channel adapters use
residual zero-impact initialization, top-frequency fraction `1/4`, and bottleneck
width `d_model/8`. FourierFT uses PEFT 0.17.1 with 1,000 frequencies, scaling 150,
seed 777, and zero initialization.

## Installation

The tested environment is Python 3.11, PyTorch 2.4.1, CUDA 12.4, and one NVIDIA
A40 with 46 GB of memory.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
```

Model checkpoints, datasets, utility records, and run artifacts are downloaded
or generated outside Git under `.cache/`, `data/`, and `artifacts/`.

Run the complete synthetic CPU path first:

```bash
.venv/bin/utility-peft reproduce --updates 2
```

This exercises all seven deployable actions, utility persistence, the oracle
analysis, controller fitting, and report generation with a tiny backbone.

## A40 Experimental Protocol

The commands are resume-safe. Re-running a completed stage reuses immutable
episodes and utility records rather than repeating successful actions.

### 1. Prepare evaluation episodes

```bash
.venv/bin/utility-peft prepare-data --config pilot --download
```

This creates 24 chronological test episodes: four datasets, two horizons, and
three episodes per dataset/horizon. Support and query raw timestamps do not
overlap, and scaling statistics use support timestamps only.

### 2. Train source heads on ETTm2

```bash
.venv/bin/utility-peft train-source-head --config pilot --download
```

Each horizon-specific head receives 2,000 deterministic AdamW updates and is
selected by ETTm2 validation MSE. ETTm2 is excluded from evaluation.

### 3. Run the matched Time-PEFT-style subset

```bash
.venv/bin/utility-peft reproduce-time-peft --config pilot --protocol matched
```

This runs A2-A5 on the middle Lorenz and Weather horizon-96 episodes with three
seeds, producing 24 parity records. `--protocol paper` remains blocked until an
official implementation is pinned and parity-tested.

### 4. Generate the two-seed oracle screen

```bash
.venv/bin/utility-peft generate-utilities --config pilot
```

Expected output: 336 records from 24 episode groups, seven actions, and seeds 0
and 1. Controller development remains blocked at this stage even if the point
estimate screen passes.

### 5. Complete the oracle confirmation

```bash
.venv/bin/utility-peft generate-utilities --config pilot \
  -o 'experiment.seeds=[0,1,2]' \
  -o experiment.include_reference=true
```

This reuses the screen, adds seed 2 for A0-A6, and evaluates A7 on one middle
episode per dataset/horizon. Expected total: 512 utility records. OOM or NaN
actions receive one clean retry; a repeated failure becomes an explicit
infeasible record rather than changing the training budget.

### 6. Train and evaluate routing

```bash
.venv/bin/utility-peft train-controller --config pilot
.venv/bin/utility-peft evaluate-heldout --config pilot
.venv/bin/utility-peft build-report --config pilot --output reports
```

LODO trains on three datasets and evaluates on the fourth. It compares the full
controller with equal-capacity complexity-only, structure-only, and
structure-plus-mismatch controllers, random routing, the best source-fixed action,
always-on A5, and the target oracle.

## Training Protocol

- Lookback: 96.
- Forecast horizons: 96 and 336 in the pilot.
- Support/query sizes: 64/128.
- Adaptation updates: exactly 100, with no query-based early stopping.
- Effective batch size: 32.
- Optimizer: AdamW, weight decay 0.01, gradient clipping 1.0.
- Learning rates: head `1e-3`, adapters `1e-4`, full tuning `1e-5`.
- Precision: BF16 on CUDA.
- Primary outcome: normalized query-MSE gain without cost penalties.
- Primary routing metrics: NDCG and oracle regret.
- Confidence intervals: paired, dataset/horizon-stratified bootstrap.

## Data

| Dataset | Role | Acquisition |
| --- | --- | --- |
| Lorenz | Evaluation | Deterministic project-owned RK4 generator |
| ETTh1 | Evaluation | Official ETT repository, pinned source commit |
| Weather | Evaluation | Official THUML Hugging Face revision |
| Exchange | Evaluation | Official THUML Hugging Face revision |
| ETTm2 | Source head only | Official ETT repository; excluded from evaluation |
| ECGCA115 | Deferred | Manual acquisition required |

The broader MVP configuration also supports horizon 192 and five episodes per
dataset/horizon, but the reported A40 pilot uses only the preregistered smaller
protocol above.

## Artifacts

```text
artifacts/pilot/
|-- episodes/       local tensors and inspectable manifests
|-- utilities/      partitioned immutable Parquet action records
|-- checkpoints/    source heads and controllers
|-- runs/           resolved configs and environment metadata
|-- parity/         matched Time-PEFT-style records
|-- lodo/           fold controllers and held-out metrics
|-- reports/        local parity metadata
`-- oracle_gate.json
```

Every utility key includes dataset, horizon, episode, action, seed,
configuration hash, model revision, and preprocessing hash. Raw MSE, MAE,
parameters, FLOPs, peak memory, wall time, and normalized gain are retained.

## Validation

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest -q
```

The opt-in A40 smoke test loads MOMENT and executes every action on ETTh1:

```bash
UTILITY_PEFT_RUN_GPU_SMOKE=1 \
UTILITY_PEFT_DATA_ROOT=data \
.venv/bin/pytest -q -m 'gpu and model' tests/test_gpu_smoke.py
```

Tests cover tensor shapes, masks, chronological splits, timestamp separation,
deterministic IDs, utility arithmetic, budget filtering, zero-impact adapters,
trainable parameter sets, exact state restoration, support-only evidence,
target-query isolation, Parquet resume/deduplication, controller fitting, source
head provenance, expected pilot counts, and GPU execution.

## Repository Layout

```text
.
|-- src/utility_peft/          package implementation
|-- configs/                   Hydra experiment and model configurations
|-- tests/                     unit, integration, leakage, and GPU tests
|-- reports/                   versioned generated result tables
|-- utility_peft_latex_project original proposal and compiled PDF
|-- BASELINE_DISCREPANCIES.md  claim boundaries and parity gaps
|-- requirements.lock          committed Python dependency lock
`-- pyproject.toml             package and tool configuration
```

Build the preserved proposal independently with:

```bash
make -C utility_peft_latex_project
```
