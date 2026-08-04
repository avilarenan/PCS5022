# PCS5022: Utility-PEFT and Correlation-Routed Time-PEFT

Utility-PEFT is a research proof of concept for selecting how a time-series
foundation model should be adapted to a new forecasting episode. Instead of
assuming that every target needs the same adapter bank, it predicts the
counterfactual gain and cost of each candidate adaptation action from support
data only.

The current implementation uses MOMENT-base and evaluates frozen inference,
head tuning, LoRA, frequency and channel adapters, FourierFT, and full fine-tuning.
The original LaTeX proposal is preserved under
[`utility_peft_latex_project/`](utility_peft_latex_project/).

> **Research status: full development screen complete; mechanism not supported.**
> Across all 13 Time-PEFT datasets, the residual-correlation router beat always-on
> Time-PEFT (`LFC`) at 100 and 300 updates, but it lost decisively to source-fixed
> LoRA (`L`) and performed worse descriptively than histogram-matched random
> routing. The result supports selective omission of optional modules, not the
> proposed learned routing mechanism. See the
> [current-results report](docs/CURRENT_RESULTS_REPORT.md) for the complete history,
> controlled results, interpretation, related work, and next experiments.

## Time-PEFT versus LoRA reproduction lane

The accepted ICML 2026 paper, *Time-PEFT: Temporal and Multichannel
Complexity-Based Fine-Tuning for Time-Series Foundation Models*, reports that
full Time-PEFT improves MOMENT-base MSE over LoRA by roughly 5.7% to 38.0% on
seven complex datasets. It does **not** report universal superiority: on the
six standard datasets, results are generally similar and Time-PEFT is clearly
better only on Weather. Its temporal and multichannel complexity measures are
dataset diagnostics and proxies for fine-tuning gain; they do not route
adapters.

The separate conventional target-train/validation/test workflow is:

```bash
.venv/bin/utility-peft run-time-peft-reproduction \
  --config time_peft_reproduction_smoke --stage all
```

The uncapped workflow compares only `L` (LoRA + target head) with `LFC` (full
Time-PEFT + target head). It uses a fresh seed-matched forecasting head, a common
AdamW LR selected from validation across seeds, and one untouched test pass. It
does not run or compose the router.

The pinned ECGCA515/horizon-96 run is a **development/parity anchor**, for which
the paper reports MSE 0.199 (LoRA) and 0.125 (Time-PEFT). It may expose major
implementation or preprocessing discrepancies, but it is not confirmatory
evidence and cannot establish a general Time-PEFT or routing result. The primary
reconstruction uses `paper_count_inferred`, channel-adapter dropout 0.1, and
forecast-head dropout 0.1. Required one-factor sensitivities use the `paper`
variant, adapter dropout 0.0, and head dropout 0.0. Synthetic reproduction uses
`dysts==0.96`, length 12,000, seed 0, and `pts_per_period=100`; the package
version, trajectory length, and default arguments are local assumptions because
the paper omits them. Generated tensors are materialized once under
`data/.utility_peft/dysts/`; runtime numerical-library versions, effective
solver arguments, shapes, and tensor SHA-256 values are bound into the run and
protocol hashes. Confirmatory labeling is rejected for a reduced matrix or the
legacy proxy generator.

Before opening any test result from the seven-dataset, three-horizon matrix,
lock its composed config, implementation hash, splits, seed/LR grid,
sensitivities, and statistical decision rules. A change after test inspection
is a new experiment, not a replacement result. See
[`docs/TIME_PEFT_REPRODUCTION.md`](docs/TIME_PEFT_REPRODUCTION.md).

After the smoke passes, the prepared anchor command is:

```bash
./scripts/run_time_peft_anchor.sh
```

This is a long run. ECGCA515-h96 has 573,809 training windows and 18
method/LR/seed trials. A planning benchmark on the current A40 estimates about
9.75 GPU-hours per mean completed epoch across the grid: at least about 39
hours under patience 3, about 98 hours at ten mean epochs, and up to about 41
days if every trial reaches 100 epochs. Peak allocation was about 8.36 GiB.
The workflow atomically caches each method/LR/seed trial under its complete
identity and reconstructs the exact grid on resume; preserve that cache until
the final tuning artifact has been validated.

When a result is needed within 24 hours, run the separate preliminary screen:

```bash
./scripts/run_time_peft_budget24.sh
```

It first evaluates ECGCA515 for at most four epochs per trial under
`artifacts/time-peft-budget24/ecg`, then evaluates the five chaotic datasets for
at most eight epochs under `artifacts/time-peft-budget24/synthetic`. Both phases
use full, uncapped horizon-96 windows, seeds 0/1, `L`/`LFC`, and learning rates
`1e-3`/`1e-4`. They are explicitly labeled `development-parity`: useful
preliminary evidence, not a confirmatory reproduction of the complete paper
matrix.

## New Experiment: Correlation-Routed Time-PEFT

The accepted Time-PEFT method does **not** use temporal/multichannel complexity
scores to switch modules. Its training algorithm keeps the forecast head, LoRA,
frequency adapter, and channel adapter active together. The new contribution in
this repository is therefore an outer router, not a cheaper replacement for an
existing Time-PEFT routing rule.

The router makes one frozen forecast on labeled support windows and computes
vectorized residual autocorrelation and cross-channel correlation features. Two
source-dataset logistic gates independently choose the frequency and channel
modules. Evaluation is leave-one-dataset-out, so the held-out dataset contributes
neither gate parameters nor thresholds. Query labels are used only after the route
and adaptation are fixed.

| Arm | Trainable components |
| --- | --- |
| `L` | Forecast head + rank-8, alpha-32 Q/K/V LoRA |
| `LF` | `L` + top-3 frequency adapter; normalized F-both sum |
| `LC` | `L` + channel adapter with an explicit zero frequency slot |
| `LFC` | Algorithm 1 frequency → channel → LayerNorm stack |

`LFC` follows the accepted paper's full dataflow. `LF` and `LC` are explicit
routing ablations: the paper motivates both components but does not fully specify
the head-side fusion for every partial mask. Their exact definitions and remaining
parity limitations are recorded in `docs/EXPERIMENT.md`.

Run the self-contained CPU execution check after installation:

```bash
./scripts/run_cpu_smoke.sh
```

Run the five-dataset GPU pilot or the complete 13-dataset workflow:

```bash
./scripts/run_correlation_pilot.sh
./scripts/run_correlation_full.sh
```

All stages are resume-safe. Reports are written under
`artifacts/correlation-*/correlation/reports/`. See
[`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the protocol,
[`docs/COMPUTE_ANALYSIS.md`](docs/COMPUTE_ANALYSIS.md) for the derivation, and
[`docs/CODEX_HANDOFF.md`](docs/CODEX_HANDOFF.md) for local continuation.

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
| Paper-style Time-PEFT-versus-LoRA runner | Implemented; reduced h96 development screen complete; full paper matrix not run |
| Residual correlation evidence and strict support-only API | Implemented |
| L/LF/LC/LFC matched LODO benchmark and reports | Implemented |
| Full Time-PEFT 13-dataset manifest/workflow | Implemented |
| Correlation-routing five-dataset GPU pilot | Completed |
| Correlation-routing 13-dataset development result | Completed; beats `LFC`, loses to fixed `L` and random routing |
| Untouched confirmatory router experiment | Not warranted with the current action bank |

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

## Historical Pilot Motivation: Residual Correlation Nonstationarity

The current Time-PEFT-inspired complexity baseline combines normalized spectral
entropy with a support-only linear-Gaussian transfer-entropy proxy. That proxy is
auditable but computationally expensive because it fits directed regressions for
every channel pair. It also describes the dataset rather than what the frozen
model failed to learn.

Plain mean Pearson correlation is cheap, but it is not sufficient by itself.
High correlation can indicate exploitable channel structure or simple redundancy,
and a single average discards lag, topology, direction, and regime changes.

The implemented next signal is **residual correlation nonstationarity (RCN)**. For
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
correlation. The committed implementation benchmark now measures the complete lag-8
correlation suite against the existing lag-1 Gaussian transfer-entropy proxy. On
this CPU host, the proxy was 4.42x slower at 21 channels and 13.44x slower at 64
channels. Those isolated results do not establish end-to-end adaptation savings.

### Implemented RCN Experiment

The new implementation performs one support-only frozen forward; computes windowed
signed/absolute residual correlations, effective rank, nonstationarity, maximum
lagged cross-correlation, and residual autocorrelations; fits separate frequency
and channel gates on source datasets; and evaluates the resulting action against
always-on `LFC` in leave-one-dataset-out folds. The exhaustive four-action sweep is
retained as offline research/calibration cost and is never presented as deployment
cost. Deployment time includes the frozen support forecast, evidence extraction,
routing, and selected adaptation.

The locked primary criterion was routed MSE noninferiority within 1% of `LFC`
together with lower end-to-end adaptation time. The completed 13-dataset run beat
`LFC`, but failed the decisive routing-value control: source-fixed `L` and
histogram-matched random routing both did better. The immutable Parquet utility
store retains every paired raw record, while the
[current-results report](docs/CURRENT_RESULTS_REPORT.md) gives the complete audit.

## Foundation Model and Baseline Boundary

The MVP pins the official
[`AutonLab/MOMENT-1-base`](https://huggingface.co/AutonLab/MOMENT-1-base)
checkpoint at revision `5e44b0ea26376a176360f87831124e018f876d96`.
`momentfm` is installed from official commit
`38f7310ad594100747ca2a8357e9c7ca7d323e0e` (package version 0.1.5),
Transformers is pinned to 4.54.1, and PEFT is pinned to 0.17.1.

MOMENT's foundation checkpoint contains a reconstruction head, not a pretrained
long-horizon forecasting head. The historical A40 pilot trains horizon-specific
heads on ETTm2 and excludes ETTm2 from evaluation. The episodic correlation
workflow instead trains heads on Electricity, which is outside the target suite.
Each source checkpoint is bound to source-data, preprocessing, split, and
target-exclusion hashes. The paper-style reproduction deliberately follows a
different convention: it initializes a fresh target forecasting head per
dataset/horizon/seed and jointly trains it in both LoRA and Time-PEFT.

As of August 2, 2026, no public official Time-PEFT implementation was found on the
paper pages, author repositories, KAIST DMLab organization, or Hugging Face. The
historical implementation remains **Time-PEFT-style**, and the new matched stack is
a **paper-specified Time-PEFT reimplementation**. Neither can claim official
reproduction or superiority without parity and GPU evidence. See
[`BASELINE_DISCREPANCIES.md`](BASELINE_DISCREPANCIES.md), the
[`ICML paper page`](https://icml.cc/virtual/2026/poster/61767), and the
[`OpenReview entry`](https://openreview.net/forum?id=n8seTOinYs).

## Legacy Utility-PEFT Actions

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

These A0-A7 definitions preserve the reported historical pilot. The new experiment
uses the separate L/LF/LC/LFC definitions shown above; do not mix the two adapter
implementations in one result table.

## Installation

The principal runs used Python 3.11, PyTorch 2.4.1+cu121, and one NVIDIA A40 with
45.5 GiB of usable memory.

```bash
./scripts/setup_env.sh
```

The equivalent manual commands are in `docs/CODEX_HANDOFF.md`.

Model checkpoints, datasets, utility records, and run artifacts are downloaded
or generated outside Git under `.cache/`, `data/`, and `artifacts/`.

Run both CPU execution checks first:

```bash
.venv/bin/utility-peft reproduce --updates 2
PATH="$PWD/.venv/bin:$PATH" ./scripts/run_cpu_smoke.sh
```

The first preserves the seven-action Utility-PEFT path. The second exercises the
new four-arm residual-correlation workflow and LODO report generation. Both use
tiny random heads and validate plumbing only.

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
| Lorenz, CellCycle, DoublePendulum, Hopfield, LorenzCoupled | Evaluation | Deterministic compatible local generators; not paper-exact |
| ECGCA115, ECGCA515 | Evaluation | Pinned PhysioNet EDF files with hash verification |
| ETTh1, ETTh2, ETTm1, ETTm2 | Evaluation | Pinned official ETT repository files |
| Weather, Exchange | Evaluation | Pinned THUML Time-Series-Library files |
| Electricity | Correlation source head only | Pinned THUML file; excluded from all target folds |

Exact URLs, revisions, checksums, aliases, channel counts, and splits live in
[`datasets/manifest.yaml`](datasets/manifest.yaml). The five-dataset pilot is a
systems check; `configs/correlation.yaml` covers the paper's complete 13-dataset
suite. The historical A40 results above still use their original four-dataset
protocol and ETTm2 source head.

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
