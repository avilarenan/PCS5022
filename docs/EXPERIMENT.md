# Correlation-routed Time-PEFT experiment

## Research question

Can a cheap, support-only residual-correlation model decide whether Time-PEFT's
frequency and channel adapters are worth activating, preserving the forecasting
accuracy of always-on Time-PEFT while reducing target-adaptation cost?

This is an extension of Time-PEFT, not a reproduction of a routing mechanism in
that paper. The accepted paper's temporal complexity (spectral entropy) and
multichannel complexity (time-delayed transfer entropy) are dataset diagnostics.
Algorithm 1 always optimizes

\[
\Theta=\{\theta_{\mathrm{LoRA}},\theta_{\mathrm{Freq}},
\theta_{\mathrm{Chan}},\theta_{\mathrm{Head}}\}.
\]

The accepted paper and its canonical OpenReview entry are linked from
`papers/README.md`. Local PDF copies are intentionally ignored.

## Hypotheses and decision rules

Let \(u\) denote a matched held-out dataset/horizon/episode unit after averaging
adaptation seeds, and let

\[
d_u=\frac{\mathrm{MSE}_{\mathrm{router},u}-
\mathrm{MSE}_{\mathrm{LFC},u}}
{\mathrm{MSE}_{\mathrm{LFC},u}}.
\]

The preregistered hypotheses are:

1. **Accuracy noninferiority.** The correlation router is noninferior to always-on
   `LFC`; the upper bound of a paired 95% interval for the mean \(d_u\) is below
   `0.01` (one percent relative MSE).
2. **End-to-end efficiency.** Evidence extraction, routing, and selected-arm
   adaptation together take less wall time than `LFC` adaptation. Active trainable
   parameters and measured adaptation FLOPs are secondary efficiency endpoints.
3. **Routing value.** The router beats the best source-fixed arm among `L`, `LF`,
   and `LC`, rather than obtaining a favorable result merely because one smaller
   arm is globally sufficient.
4. **Non-degenerate selection.** At least two masks are selected across held-out
   units, and the accuracy result is not produced by routing every unit to `LFC`.

The point-estimate fields `relative_mse_difference` and
`noninferior_within_margin` implement the unit-level estimand above and the
configured margin. They are development diagnostics, not sufficient for a research
claim without the paired interval. A pilot is promising, but not confirmatory, when
median relative MSE degradation is at most two percent and end-to-end time falls by
at least ten percent.

## Accepted-paper audit and deliberate deviations

The accepted paper specifies the following Time-PEFT components:

- backbone: five TSFMs are studied; this repository's primary comparison uses
  MOMENT-base;
- LoRA: rank 8, scale 32, injected into query, key, and value projections for
  Transformer backbones;
- frequency adapter: FFT across patches, retain the three largest-amplitude
  frequency bins, inverse FFT, and an \(h_1\to h_2\) projection with \(h_2=h_1\);
- channel adapter: shared \((h_1+h_2)\to r\) down-projection and one
  channel-specific \(r\to h_1\) up-projection per channel, with \(r=h_1/2\);
- full-path fusion: `Efilt=Freq(Eback)`, `Ech=Channel(Eback, Efilt)`, followed by
  `ForecastHead(LayerNorm(Ech))` as written in Algorithm 1;
- forecast head: trained in every non-zero-shot arm;
- paper training: AdamW, learning rate selected between \(10^{-3}\) and
  \(10^{-5}\), default batch size 128, maximum 100 epochs with early stopping;
- forecasting protocol: lookback 96 and horizons 96, 192, and 336.

The repository makes several explicit choices where the paper or released assets
are insufficient:

- no official author implementation is included or validated in this repository,
  so this is a `paper-specified Time-PEFT reimplementation`;
- the paper does not state adapter initialization, so this repository uses seeded
  Xavier initialization; shared modules have matched weights across arms, but their
  initial predictions differ because no residual bypass is inserted around Algorithm 1;
- output LayerNorm is non-affine because the paper's optimized parameter set and
  simplified count omit a separate normalization term;
- cost experiments use a fixed update budget (100 by default) instead of
  early-stopped epochs, eliminating a convergence-dependent timing confound;
- the public MOMENT checkpoint has no useful pretrained long-horizon forecasting
  head, so horizon-specific heads are trained on Electricity, outside every target
  fold;
- local chaotic generators preserve the named systems, dimensions, deterministic
  integration, and experimental interface, but do not claim the exact `dysts`
  versions, sampling, initial conditions, or trajectories used by the authors.

These deviations prevent an exact numerical reproduction claim. They do not prevent
a matched internal comparison because every arm uses the same checkpoint, head,
episode, initialization, batches, optimizer, seed, and number of updates.

## Four matched arms

| ID | Always active | Optional adapter mask | Role |
| --- | --- | --- | --- |
| `L` | head + q/k/v LoRA | none | minimum matched arm |
| `LF` | head + q/k/v LoRA | frequency | F-both streams summed, normalized, then sent to the head |
| `LC` | head + q/k/v LoRA | channel | zero-filled absent-frequency slot, then `LayerNorm(Ech)` |
| `LFC` | head + q/k/v LoRA | frequency + channel | accepted-paper Algorithm 1 baseline |

All four arms must be generated for every episode and seed. The router is evaluated
offline by selecting one already matched arm. This avoids retraining a target arm
after observing its query result and makes action/seed pairing auditable.

`L` is the paper's LoRA comparator. `LF` and `LC` are routing arms rather than a
claim that the paper defines two deployable algorithms. Table 4 motivates these
factorial ablations, but it does not fully specify their head-side dataflow. For
`LF`, the two F-both streams have equal width and are added before non-affine
LayerNorm. For `LC`, a zero tensor explicitly occupies the absent frequency slot so
the shared down-projection retains the same \((h_1+h_2)\to r\) capacity. `LFC`
does not use either ablation rule; it follows Algorithm 1 directly.

## Correlation evidence

For support tensors \(X^{sup},Y^{sup}\), the source-initialized frozen model is
called exactly once in eval/no-grad mode:

\[
E_b=\widehat Y_b^{\mathrm{frozen}}-Y_b^{sup},\qquad
E\in\mathbb{R}^{B\times C\times H}.
\]

Each batch element \(b\) remains a separate forecast window. Statistics are never
computed by concatenating adjacent windows.

The **frequency gate** receives residual temporal features: signed and absolute
lag-1 autocorrelation, signed/absolute mean and maximum autocorrelation through
`max_lag` (8 by default), autocorrelation nonstationarity, and decay.

The **channel gate** receives residual cross-channel features: signed/absolute
zero-lag correlation, pairwise dispersion, correlation nonstationarity across
support windows, correlation-matrix effective rank and rank fraction, maximum
lagged cross-correlation.

Frozen MSE, residual scale, channel count, and horizon are retained in the
evidence record for diagnostics, but the default gates deliberately exclude them;
their inputs are strictly correlation/autocorrelation features.

Pairwise correlations use jointly finite observations. Undefined statistics, such
as cross-channel correlation for \(C=1\), map to zero. The extractor returns only
finite scalar features and restores the model's original device, train/eval mode,
and gradient flags.

This score is model-aware: it measures structure left in the source model's errors,
not just structure in the raw target data. It is nevertheless linear and
associational. High correlation does not prove that an adapter will improve the
forecast, which is why adapter activation is learned from source-only marginal
outcomes rather than hard-coded from correlation magnitude.

## Source-only marginal labels

For one source episode, let \(\ell_a\) be query MSE for action \(a\), averaged over
matched seeds. Frequency and channel benefits are estimated in both contexts:

\[
b_F=\frac{1}{2}\left[
\frac{\ell_L-\ell_{LF}}{\ell_L}+
\frac{\ell_{LC}-\ell_{LFC}}{\ell_{LC}}
\right],
\]

\[
b_C=\frac{1}{2}\left[
\frac{\ell_L-\ell_{LC}}{\ell_L}+
\frac{\ell_{LF}-\ell_{LFC}}{\ell_{LF}}
\right].
\]

The default positive label is \(b_m>0.002\), a 0.2% minimum relative MSE
benefit. Two independent class-balanced logistic gates are fitted with source-only
median imputation and standardization. Defaults are regularization `C=1`,
`liblinear`, maximum 1,000 iterations, and probability threshold 0.5. If a source
fold contains one class only, the corresponding gate uses a deterministic constant
probability; it must be reported as such.

These labels use query outcomes from source datasets and therefore belong to the
offline router-training stage. They must never be constructed for a target before
that target is evaluated.

## Leave-one-dataset-out protocol

For every held-out target dataset \(D^*\):

1. Remove all records from \(D^*\) before label construction, preprocessing, and
   gate fitting.
2. On complete source episodes only, average matched seed outcomes, construct
   \(b_F,b_C\), fit source-only feature preprocessing, and fit both gates.
3. Extract support evidence for a target episode without giving the router its
   query object or any `query_*` field.
4. Select `L`, `LF`, `LC`, or `LFC` from the two binary gate decisions.
5. Pair the selected action and `LFC` by the same target episode and seed; evaluate
   query MSE/MAE and measured costs.
6. Repeat until each of the 13 datasets has served exactly once as the outer target.

Thresholds and feature lists are global source-side hyperparameters. Any threshold
tuning must be nested inside the source datasets of each outer fold. A row-wise
split, target-assisted imputation, or target query calibration invalidates the
result.

## Episode and timestamp protocol

An episode consists of a labeled support block followed by a query block. Raw
timestamp ranges are disjoint: `query_start >= support_end`. Sliding windows inside
each block may overlap one another, but a raw timestamp cannot appear in both
support and query. Channel normalization is fitted on the support raw interval.

The current correlation configs set `episode_partition: test`. This is an explicit
few-shot/episodic adaptation protocol: labeled support and later query windows are
both carved from the conventional test partition, without overlap. It is not the
same as the paper's conventional target-train/target-test evaluation. Results must
be labeled accordingly. A paper-style experiment should instead train on the
target train split, select/stop on validation, and perform one untouched official
test evaluation; do not silently mix those two protocols in one table.

The meaningful frozen residual requires a non-random forecast head. The configured
Electricity source head is trained separately for every horizon and carries
provenance binding its source data, scaler, split, target-exclusion set, and model
revision. Electricity must remain outside the target suite whenever that head is
used.

## Dataset suite

The full workflow exposes all 13 datasets used in the Time-PEFT analysis.

| Family | Dataset | Channels | Repository source/fidelity |
| --- | --- | ---: | --- |
| chaotic | Lorenz | 3 | deterministic RK4 compatible substitute |
| chaotic | CellCycle | 6 | deterministic regulatory-oscillator substitute |
| chaotic | DoublePendulum | 4 | deterministic RK4 compatible substitute |
| chaotic | Hopfield | 6 | deterministic continuous-network substitute |
| chaotic | LorenzCoupled | 6 | deterministic coupled-Lorenz substitute |
| medical | ECGCA115 | 6 | pinned PhysioNet EDF, 2 thoracic + 4 abdominal signals |
| medical | ECGCA515 | 6 | pinned PhysioNet EDF, 2 thoracic + 4 abdominal signals |
| standard | ETTh1 | 7 | pinned ETT CSV and fixed ETT boundaries |
| standard | ETTh2 | 7 | pinned ETT CSV and fixed ETT boundaries |
| standard | ETTm1 | 7 | pinned ETT CSV and fixed ETT boundaries |
| standard | ETTm2 | 7 | pinned ETT CSV and fixed ETT boundaries |
| standard | Weather | 21 | pinned TSLib CSV, chronological 70/10/20 split |
| standard | Exchange | 8 | pinned TSLib CSV, chronological 70/10/20 split |

`datasets/manifest.yaml` is the source of truth for URLs, revisions, SHA-256
digests, aliases, dimensions, and splits. Electricity (321 channels) is listed
there only as a source-head dataset.

The full config uses 13 datasets, three horizons, three chronological episodes,
three seeds, and four arms: 1,404 matched utility records. The pilot uses Lorenz,
DoublePendulum, ECGCA515, ETTh1, and Weather at horizon 96: 180 records.

## Metrics and timing boundaries

### Accuracy

- Primary: query MSE.
- Secondary: query MAE, per-dataset/horizon results, negative adaptation relative
  to `L`, and route frequency.
- Do not average unnormalized MSE across families without also reporting the
  per-dataset table and paired relative differences.

### Computation

- `adaptation_wall_time_s`: fixed optimization updates only, measured after the
  model/action is built; GPU timing is synchronized.
- `evidence_wall_time_s`: one frozen support forecast plus residual-correlation
  extraction. Support-tensor and template placement on the target device, plus
  template restoration to its original device, are outside this synchronized
  timing region, matching the evaluator's exclusion of setup from adaptation
  time.
- `routing_wall_time_s`: preprocessing and two gate probability calls.
- router end-to-end: evidence + routing + selected-arm adaptation.
- always-on baseline end-to-end: `LFC` adaptation only; it needs no evidence or
  gate.
- `trainable_parameters`: parameters active for the chosen arm.
- `stored_adapter_parameters`: currently recorded per active arm; it does not prove
  storage savings for a deployment that serializes the complete four-arm bank.
- `profiled_flops`: one profiled forward/backward optimization step multiplied by
  the fixed update count. Report analytical FFT/projection costs alongside it.
- `peak_memory_mb`: peak allocated device memory after the evaluator resets its
  counter; the current boundary includes frozen query evaluation, adaptation, the
  profiler pass, and adapted query evaluation. Report hardware, software versions,
  precision, batch size, and whether the process was isolated.

Exclude one-time data/model downloads, episode preparation, source-head training,
offline generation of source utility labels, and offline logistic fitting from
target adaptation latency. They are still reproducibility costs and should be
reported separately in a run manifest.

For stable timing, warm the device, synchronize before and after the region, run
arms in balanced/randomized order, and pair seeds. Report both the mean and a
robust statistic such as the median. A parameter-count reduction alone is not a
runtime result.

### Uncertainty

For the final comparison, average seeds within each episode first and use a paired
hierarchical bootstrap that resamples datasets, then units within datasets. Report
the 95% interval for relative MSE and every cost reduction. A paired Wilcoxon test
over dataset/horizon summaries may be reported as secondary evidence; adjust
multiple secondary tests with Holm's method.

## Required controls and ablations

The four-arm table provides three fixed controls (`L`, `LF`, `LC`) and the
always-on baseline (`LFC`). At minimum report:

- correlation router versus `LFC`;
- correlation router versus the best source-fixed smaller arm among `L`, `LF`, and
  `LC`, chosen without the target;
- independent frequency/channel gate accuracy and one-class fallbacks;
- raw input-correlation features versus frozen-residual correlation features;
- `max_lag` and support-size sensitivity;
- routing with and without nonstationarity/effective-rank features;
- active parameters and end-to-end time by selected route, not only overall mean.

A random mask matched to the router's route histogram is a useful secondary
control. It must be repeated enough times to estimate routing variance.

## Claim limitations

Do not claim any of the following from this bundle alone:

- exact reproduction of accepted-paper numbers;
- official Time-PEFT parity;
- that the paper proposed module routing;
- that transfer entropy is part of always-on Time-PEFT's target training cost;
- that correlation is causal or universally identifies useful modules;
- that synthetic CPU smoke output is evidence of accuracy;
- that active-parameter savings equal checkpoint-size, memory, FLOP, or wall-time
  savings;
- generalization beyond MOMENT-base, long-horizon forecasting, and the evaluated
  dataset suite.

The defensible result is narrower: under a fully specified episodic protocol and
matched arms, a source-trained residual-correlation router either does or does not
retain `LFC` accuracy while saving measured target-adaptation resources.
