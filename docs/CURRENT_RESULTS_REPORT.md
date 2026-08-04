# Utility-PEFT and correlation-routed Time-PEFT: current results, lessons, and next steps

**Research snapshot:** 4 August 2026

**Backbone:** MOMENT-base

**Hardware for the principal runs:** one NVIDIA A40 (45.5 GiB), PyTorch 2.4.1, CUDA 12.1

**Evidence status:** completed development experiments; no confirmatory superiority claim

## Executive summary

This project began with a broad question: can a small support episode tell us which
parameter-efficient adaptation operation will help a time-series foundation model
before we pay to run every candidate? The proposed answer, Utility-PEFT, was an
action-conditioned predictor of counterfactual adaptation value. The project then
narrowed to a more testable mechanism: use frozen-model residual autocorrelation and
cross-channel correlation to decide whether the frequency and channel modules of a
paper-specified Time-PEFT implementation should be enabled.

The experiments produced a useful but negative mechanism result.

1. A conventional, reduced-budget Time-PEFT reproduction showed that the additional
   frequency and channel modules can help: full Time-PEFT beat LoRA on ECGCA515,
   Hopfield, and LorenzCoupled. It lost on CellCycle, DoublePendulum, and Lorenz.
   This establishes heterogeneity, but not universal Time-PEFT superiority.
2. In the full 13-dataset episodic experiment, the learned router beat always-on
   Time-PEFT (`LFC`) by **17.44%** relative MSE at 100 updates and **15.15%** at
   300 updates. Both paired hierarchical 95% intervals excluded zero.
3. That favorable comparison is not evidence that the router worked. A source-only
   fixed policy chose LoRA (`L`) in every outer fold and beat `LFC` by **35.70%**
   and **32.03%**, respectively. The learned router was significantly worse than
   this fixed policy and descriptively worse than histogram-matched random
   assignment.
4. The oracle selected `L` on 20/26 primary units and 15/26 sensitivity units.
   The channel-only extension (`LC`) was never oracle-optimal, but the router chose
   a channel-active route seven times in the primary and eight times in the
   sensitivity run.
5. Conditional activation reduced active trainable parameters by 39--48%, but it
   reduced end-to-end time by only 0.8--1.3%, far below the preregistered 10%
   promising target. Shared backbone, head, LoRA, data, and optimizer work dominates
   the optional adapter cost.

The defensible conclusion is therefore:

> Under short-budget, few-shot episodic adaptation with an Electricity-trained
> MOMENT forecasting head, omitting optional Time-PEFT modules is usually better
> than always activating them. The current residual-correlation gates do not predict
> the exceptions reliably and add no value over fixed LoRA.

This does **not** show that Time-PEFT is ineffective in its own conventional
target-train/validation/test setting. It shows that our candidate experts were not
sufficiently specialized or predictable in this different episodic setting. The
next research milestone must establish useful oracle headroom beyond `L` before
another router is trained.

## 1. How the project began

The original proposal, [Utility-PEFT](../utility_peft_latex_project/main.tex),
distinguished two concepts:

- **data complexity** describes the target series; and
- **adaptation utility** asks what a particular operation will improve for this
  target, backbone, support set, optimization budget, and resource constraint.

For episode `e` and action `a`, the proposal defined utility as normalized query-loss
reduction after support adaptation minus parameter, memory, FLOP, and wall-time
costs. Query outcomes are available only for offline utility-label construction and
final evaluation. At deployment, only support information may determine the action.

The initial action bank was deliberately broad:

| Historical ID | Operation |
| --- | --- |
| `A0` | Frozen/no update |
| `A1` | Forecast head |
| `A2` | Head + LoRA |
| `A3` | Head + LoRA + frequency adapter |
| `A4` | Head + LoRA + channel adapter |
| `A5` | Head + LoRA + frequency + channel adapters |
| `A6` | Head + FourierFT |
| `A7` | Full fine-tuning reference |

The proposal was intentionally falsifiable. It said the method was weakened if a
globally fixed action matched the learned policy, utility labels were unstable, or
inference overhead erased the saved adaptation cost. Those early-stop conditions
are now directly relevant: the experiments met several of them.

## 2. An important boundary: Time-PEFT is not a router

[Time-PEFT](https://openreview.net/forum?id=n8seTOinYs) augments LoRA and a forecast
head with a top-frequency adapter and a multichannel adapter. Its spectral-entropy
and information-flow measures characterize datasets and motivate specialized
adaptation; the published training path activates LoRA, frequency, channel, and
head together. We call that full path `LFC`.

Our later experiment is a separate outer decision system:

| Current arm | Trainable components |
| --- | --- |
| `L` | Forecast head + rank-8, alpha-32 Q/K/V LoRA |
| `LF` | `L` + frequency adapter |
| `LC` | `L` + channel adapter |
| `LFC` | `L` + frequency + channel adapters and output normalization |

The router does not replace a router inside Time-PEFT; none exists in the accepted
method. It attempts to improve Time-PEFT by conditionally omitting modules.

We kept two experimental lanes separate:

1. **Conventional reproduction:** fresh target head, target training, validation LR
   selection, then one test evaluation. No router is present.
2. **Episodic routing:** an Electricity-trained source head, 64 labeled support
   windows, support-only routing, fixed-update adaptation, then 128 query windows.
   The target dataset is excluded from every leave-one-dataset-out gate fit.

These lanes answer different questions and their MSE values must not be pooled.
No official author implementation was publicly available when this snapshot was
prepared. All local results are therefore labeled **paper-specified Time-PEFT
reimplementation**, never official reproduction.

## 3. Experiment chronology

| Stage | Matrix | Purpose | Outcome |
| --- | --- | --- | --- |
| Historical Utility-PEFT A40 pilot | 4 datasets, horizons 96/336, 3 episodes, 3 seeds, A0--A6 plus 8 A7 references; 512 records | Establish action heterogeneity and train an action-conditioned controller | Oracle heterogeneity existed, but fixed `A0` had much lower regret than the controller; matched-baseline CI crossed zero |
| Correlation-motivation screen | 24 historical episode groups | Test whether correlation dynamics relate to marginal channel utility | Exploratory associations motivated residual-correlation features |
| Evidence microbenchmark | CPU tensors with 21 and 64 channels | Compare vectorized correlation evidence with Gaussian transfer-entropy proxy | Correlation suite was 4.42x and 13.44x faster in isolation |
| Correlation CPU smoke | Tiny backbone, 3 datasets, h8, 2 episodes, 1 seed, 2 updates; 24 final-hash records | Exercise the complete support-only path | Plumbing passed; accuracy and timing are non-evidence |
| Five-dataset router GPU pilot | 5 datasets, h96, 3 episodes, 3 seeds, 100 updates; 180 records | First MOMENT residual-correlation router test | Apparently promising versus `LFC`, but missing decisive controls |
| Time-PEFT plumbing smoke | ECGCA515 h96 with severe caps | Validate conventional runner | Completed; values are explicitly not accuracy evidence |
| Runtime calibration/full-anchor attempt | ECGCA515, planned 18-trial LR/method/seed grid | Estimate faithful reproduction cost | At least about 39 A40 GPU-hours under patience 3; full anchor was not completed |
| Simplified conventional screen | ECGCA515 + 5 chaotic systems, h96, 2 seeds, `L`/`LFC`, 2 LRs | Obtain preliminary Time-PEFT parity evidence within the available budget | `LFC` won 3/6 cells |
| Full router primary | All 13 Time-PEFT datasets, h96, 2 raw-disjoint episodes, 3 seeds, 4 arms, 100 updates; 312 records | Test the router with paired uncertainty and controls | Beat `LFC`; lost decisively to fixed `L` and random routing |
| Longer-update sensitivity | Same 26 units, 2 seeds, 300 updates; 208 records | Check whether the finding was caused by very short adaptation | Same scientific conclusion |

Generated data, checkpoints, and utility records remain under `artifacts/` and are
ignored by Git. The committed configurations and launchers reproduce the matrices.

## 4. Historical Utility-PEFT results

The first A40 pilot used Lorenz, ETTh1, Weather, and Exchange at horizons 96 and
336. It produced 504 A0--A6 records and eight A7 references. The nondeployable
oracle found winning LoRA, channel, frequency+channel, and FourierFT families, so
no single trainable family won everywhere.

| Historical metric | Result |
| --- | ---: |
| Best fixed action | `A0` (no adaptation) |
| Best fixed mean oracle regret | 0.097327 |
| Learned controller mean oracle regret | 0.282610 |
| Historical always-on A5 mean oracle regret | 0.568695 |
| Learned controller mean LODO NDCG | 0.872973 |
| Complexity-only mean LODO NDCG | 0.808269 |
| Random-routing mean LODO NDCG | 0.726115 |
| Controller relative MSE versus historical A5 | -2.67% |
| Paired 95% interval | [-16.33%, +14.05%] |
| End-to-end time reduction versus A5 | 77.24% |

The controller ranked actions better than the complexity-only model, but ranking
quality did not translate into low regret or a significant MSE improvement. The
trivial `A0` policy remained substantially better. Thus the first implementation
already warned that predicting a detailed action ranking is not useful when a
simple conservative action dominates realized utility.

The historical adapters are not the later paper-specified Time-PEFT arms: their
LoRA targets/scales, adapter definitions, and head provenance differ. Historical
`A5` and current `LFC` must never appear in one pooled effect estimate.

## 5. Why residual correlation looked promising

A post-hoc screen defined the marginal contribution of the historical channel
module and compared it with support-only descriptors across 24 episode groups.

| Descriptor | Spearman association with marginal channel contribution |
| --- | ---: |
| Mean absolute input correlation | about +0.30 |
| Maximum lagged input correlation | +0.38 |
| Input-correlation nonstationarity | -0.53 |
| Gaussian transfer-entropy proxy | +0.52 |

The result was small and dataset-confounded, but it suggested that correlation
dynamics contain information discarded by a single average correlation. The new
extractor therefore made one frozen support forecast and measured residual
autocorrelation, cross-channel correlation, effective rank, nonstationarity, and
lagged cross-correlation. It was also substantially cheaper than fitting pairwise
Gaussian regressions: 4.42x faster at 21 channels and 13.44x at 64 channels in the
committed CPU microbenchmarks.

That microbenchmark justified a cheap evidence family. It did not establish that
the evidence predicts counterfactual adapter gain.

A separate tiny-backbone CPU smoke then exercised the complete support-only route
on Lorenz, DoublePendulum, and Hopfield at h8, with two episodes, one seed, and two
updates. Its final-hash records gave routed MSE 25.3746 versus `LFC` 25.3595
(+0.02%) and 5.09% fewer active parameters. The random head, tiny budget, and
unstable CPU timing make these plumbing values non-evidence.

## 6. Five-dataset correlation-router pilot

The first GPU router pilot used Lorenz, DoublePendulum, ECGCA515, ETTh1, and
Weather, with three h96 episodes and three seeds. It used the bias-free `paper`
adapter variant, zero dropout, BF16, legacy compatible synthetic trajectories, and
100 updates.

The original report looked encouraging:

- router versus always-on `LFC`: **-18.62%** relative MSE;
- active trainable parameters: **55.09% lower**;
- end-to-end time: **0.97% slower**, not faster;
- routes: `L=6`, `LF=4`, `LC=5`, `LFC=0`.

Applying the later control analysis to the immutable pilot records changes the
interpretation:

| Pilot comparison | Post-hoc controlled result |
| --- | ---: |
| Router vs `LFC` | -18.62%, 95% CI [-40.27%, +0.65%] |
| Source-fixed `L` vs `LFC` | -29.14% |
| Router vs source-fixed `L` | +14.02%, 95% CI [+1.75%, +30.65%] |
| Histogram-matched random vs `LFC` | -16.11% |

The pilot was useful because it prompted a full experiment, but it never supported
the mechanism. Its favorable point estimate against a weak always-on comparator
was compatible with the simpler explanation that optional adapters were usually
harmful. The full run was designed specifically to distinguish those stories.

## 7. Conventional Time-PEFT reproduction work

The project paused routing to answer a prerequisite: does the local Time-PEFT
implementation reproduce an advantage over LoRA at all?

The public MOMENT checkpoint contains a reconstruction head rather than a trained
long-horizon forecasting head, and the Time-PEFT paper omits several run-level
choices. The local reconstruction therefore pins a fresh target head, Q/K/V LoRA,
official `dysts==0.96` trajectories, split/scaler provenance, and explicit local
choices for dropout, bias, optimization, and early stopping.

The first severely capped ECGCA515 smoke produced MSE 0.3060 for `L` and 0.4837 for
`LFC`. It existed only to validate data, tuning, cache, and report plumbing;
512/256/256 capped windows and two batches per epoch make its apparent accuracy
difference invalid.

An uncapped ECGCA515 h96 grid would contain 18 method/LR/seed trials over 573,809
training windows. Calibration projected at least about 39 A40 GPU-hours under
patience three and roughly 98 hours at ten mean epochs. The complete seven-dataset,
three-horizon conventional matrix was therefore not run. A reduced screen retained
all stride-one windows but used h96, two seeds, two learning rates, and capped
epochs.

The reduced launcher also exposed an operational, not scientific, failure: after
the ECG phase finished, its follow-on command was unavailable in the inherited
shell environment. The cached ECG work remained valid, and the synthetic phase was
resumed manually. This created idle calendar time but did not change either result
matrix.

| Dataset | LoRA MSE | `LFC` MSE | `LFC` improvement over LoRA |
| --- | ---: | ---: | ---: |
| ECGCA515 | 0.146785 | 0.123508 | +15.86% |
| CellCycle | 0.236480 | 0.250609 | -5.97% |
| DoublePendulum | 0.009278 | 0.014776 | -59.27% |
| Hopfield | 0.304396 | 0.264376 | +13.15% |
| Lorenz | 0.182985 | 0.212302 | -16.02% |
| LorenzCoupled | 0.324336 | 0.296777 | +8.50% |

Full Time-PEFT won 3/6 cells. ECGCA515 is the most encouraging parity anchor:
local `LFC` MSE 0.123508 is close to the paper's 0.125 reference, although local
LoRA is also much better than the paper's 0.199, so the relative gain is smaller.
The synthetic absolute scales, particularly DoublePendulum, are not numerically
comparable without the authors' exact trajectories and preprocessing.

A test-informed selector that chose `L` or `LFC` after observing these six test
outcomes would improve on always-on `LFC` by about 9.44% on average. That number is
an oracle upper bound, not a routing result. The screen shows that selection could
matter, while also showing that Time-PEFT's advantage is sensitive to dataset and
local reconstruction details.

## 8. Full 13-dataset episodic router experiment

### 8.1 Locked protocol

The principal development experiment covered Lorenz, CellCycle, DoublePendulum,
Hopfield, LorenzCoupled, ECGCA115, ECGCA515, ETTh1, ETTh2, ETTm1, ETTm2, Weather,
and Exchange at lookback/horizon 96/96.

- Two chronological raw-disjoint episodes per dataset.
- 64 labeled support and 128 query windows.
- Primary seeds 0/1/2 and 100 fixed updates.
- Sensitivity seeds 0/1 and 300 fixed updates.
- Identical initialization, batches, and schedule across `L/LF/LC/LFC` within a
  seed; head LR `1e-3`, adapter LR `1e-4`, FP32 adaptation.
- `paper_count_inferred` adapters, rank-8/alpha-32 Q/K/V LoRA, top-k 3 frequency
  filtering, adapter/head dropout 0.1.
- An Electricity source head trained for 2,000 updates, selected at update 1,200
  with validation MSE 0.245988, and provenance-bound to exclude all 13 targets.
- Two LODO logistic gates using seven temporal residual features and seven channel
  residual features, threshold 0.5, lag 8, and a 0.2% marginal-benefit label.
- Dataset-then-episode hierarchical bootstrap with 10,000 samples.

The primary produced all 312 expected records; the sensitivity produced all 208.
The complete workflow ran from 03:22:57 to 07:25:16 UTC, 4h02m19s.

### 8.2 Headline comparisons

Negative values below mean lower MSE than the named control. Timing reductions are
point diagnostics without uncertainty intervals.

| Metric | 100 updates | 300 updates |
| --- | ---: | ---: |
| Router vs `LFC` | **-17.44%** | **-15.15%** |
| Hierarchical 95% CI | **[-30.63%, -6.29%]** | **[-27.30%, -4.31%]** |
| Source-fixed `L` vs `LFC` | -35.70% | -32.03% |
| Router vs source-fixed `L` | **+102.11%** | **+75.89%** |
| Router-vs-`L` 95% CI | **[+14.96%, +218.26%]** | **[+17.73%, +149.95%]** |
| Matched random vs `LFC` | -22.41% | -19.10% |
| Router vs matched random | +48.18% | +33.05% |
| Query oracle vs `LFC` | -36.67% | -34.98% |
| Router regret vs oracle | +103.44% | +78.96% |
| Active trainable-parameter reduction vs `LFC` | 48.43% | 39.63% |
| Profiled-FLOP reduction | 1.74% | 1.33% |
| Peak-memory reduction | 1.63% | 1.25% |
| End-to-end time reduction | 0.84% | 1.27% |

Both runs establish preliminary superiority over the predefined episodic `LFC`
benchmark. Both also establish that the learned router is worse than the strongest
valid source-only fixed control. The second result governs the mechanism claim.
Post-hoc hierarchical audits put fixed `L` versus `LFC` at -35.70% [95% CI
-51.16%, -22.01%] and -32.03% [-47.58%, -17.42%]. In contrast, the all-action
query oracle improved on `L` by only 1.26% [0.09%, 3.21%] and 2.84% [0.26%,
7.84%]. The addressable selection benefit was real but very small.

Fixed `L` also dominated the router operationally: it used about 71.45% fewer
active parameters, 2.53% fewer profiled FLOPs, and 2.26% less peak memory than
`LFC`; reduced time by roughly 2.93% and 3.09%; and required no evidence pass. The
router captured only about 49% and 47% of `L`'s accuracy gain over `LFC`. The
histogram-matched random comparison is descriptive and global; its randomization
interval is not a sampling confidence interval or a fold-specific permutation test.

### 8.3 Dataset-level routed effect versus `LFC`

These are equal-seed fold means of the unit-level relative-MSE estimand, not ratios
of pooled raw MSE across datasets.

| Held-out dataset | 100 updates | 300 updates |
| --- | ---: | ---: |
| CellCycle | -0.53% | -0.06% |
| DoublePendulum | -41.48% | -46.95% |
| ECGCA115 | -40.60% | +0.00% |
| ECGCA515 | -13.07% | +4.65% |
| ETTh1 | +4.30% | -24.81% |
| ETTh2 | -11.67% | -11.98% |
| ETTm1 | -16.80% | -13.56% |
| ETTm2 | +4.06% | -9.11% |
| Exchange | -22.55% | -21.74% |
| Hopfield | -49.11% | -47.86% |
| Lorenz | -31.84% | -24.11% |
| LorenzCoupled | -6.60% | -12.35% |
| Weather | -0.79% | +10.96% |

The overall direction is stable, but individual datasets are not. Weather becomes
clearly worse at 300 updates, while ETTh1 becomes much better. This variability is
consistent with budget-dependent adaptation utility and argues against a router
that ignores the adaptation budget.

### 8.4 Direct mechanism diagnostics

| Diagnostic | 100 updates | 300 updates |
| --- | ---: | ---: |
| Router routes | L=10, LF=9, LC=6, LFC=1 | L=6, LF=12, LC=3, LFC=5 |
| Oracle routes | L=20, LF=4, LC=0, LFC=2 | L=15, LF=8, LC=0, LFC=3 |
| Exact router-oracle matches | 9/26 | 8/26 |
| Units where `L` beat `LFC` | 24/26 | 22/26 |
| Router activated optional modules when oracle was `L` | 13 | 11 |
| Frequency probabilities within 0.1 of threshold | 12/26 | 11/26 |
| Channel probabilities within 0.1 of threshold | 5/26 | 6/26 |

Only 11/26 routed actions agreed across the two budgets, while oracle actions agreed
on 19/26. The comparison is partly confounded by using three primary seeds and two
sensitivity seeds, but the large route change still shows that the gate labels and
probabilities are not stable enough for a fixed budget-agnostic policy.

Each outer fold had only 24 source episode units. At 100 updates there were roughly
4--5 positive frequency examples and 1--2 positive channel examples per fold; at
300 updates there were about 9--11 and 2--3. There were no one-class fallback folds,
but the minority classes remained extremely small.

### 8.5 Post-hoc failure audit

The cached held-out predictions allow a more direct diagnostic of whether each gate
recognized its own marginal-benefit label. These statistics were computed after
the primary report and are therefore explanatory, not preregistered endpoints.

| Gate diagnostic | 100 updates | 300 updates |
| --- | ---: | ---: |
| Frequency true positives / predicted positives | 5 / 10 | 11 / 17 |
| Frequency balanced accuracy / AUC | 0.386 / 0.257 | 0.564 / 0.442 |
| Channel true positives / predicted positives | 2 / 7 | 3 / 8 |
| Channel balanced accuracy / AUC | 0.354 / 0.292 | 0.326 / 0.464 |
| Frequency probability--benefit Spearman | -0.192 | -0.180 |
| Channel probability--benefit Spearman | -0.284 | +0.009 |

The channel gate recovered zero true positives at either budget. The frequency
gate improved at 300 updates, but its probabilities still ranked continuous
benefit in the wrong direction. This is stronger evidence of selector failure than
route counts alone.

The target construction also creates avoidable errors. It averages each module's
marginal effect across contexts and then makes two independent decisions. In one
primary ETTh1 unit, for example, `L=1.231`, `LF=1.332`, `LC=2.130`, and
`LFC=1.803`. Frequency receives a positive marginal label because it improves the
bad `LC` path, even though `L` is optimal and adding frequency to `L` hurts. Even a
query-informed version of these two marginal labels would beat `L` by only 0.49%
and 2.10% and would still misroute four units at each budget. The issue is therefore
both the learned evidence and the factorized target.

Finally, the optional paths start at a major functional disadvantage. Before any
target update, `LF` was worse than `L` on 25/26 units (median +113%), `LC` on 26/26
(+84%), and `LFC` on 25/26 (+80%). The reconstructed paper path uses Xavier-initialized
non-residual projections and sends the channel output directly through LayerNorm to
a head trained on the unadapted Electricity representation. Initialization is not
specified by the paper, so this is a local path-compatibility hypothesis, not proof
of an author-method defect. Short support optimization often failed to repair it:
`L` itself worsened from its action-specific frozen start on 19/26 primary and
17/26 sensitivity units, consistent with support overfit or support/query regime
shift.

## 9. Why the approach did not improve as expected

### 9.1 What the data establish

The following are observations, not speculation.

1. **There was almost no useful headroom beyond fixed LoRA.** The post-hoc paired
   oracle-vs-`L` estimand was only -1.26% at 100 updates and -2.84% at 300 updates;
   restricting the oracle to `L/LF` reduced that to -0.96% and -1.92%. A router
   had little upside and substantial downside.
2. **The channel action lacked demonstrated utility.** `LC` was never oracle-optimal
   on any of the 26 units at either budget, yet channel-active routes were chosen
   repeatedly.
3. **The evidence-to-action association was wrong.** Gate AUCs were below 0.5 and,
   descriptively, keeping the same route histogram but permuting assignments
   produced better performance at both budgets.
4. **Decisions were brittle.** Many gate probabilities were close to 0.5 and only
   42% of routes agreed across budgets.
5. **Parameter savings did not imply systems savings.** Optional modules account
   for many trainable parameters but little of the shared forward/backward work.

### 9.2 Most plausible explanations

These are hypotheses to test, ordered roughly by current support.

#### A. Expert viability came before routing, and it failed

Successful conditional-computation systems need experts that are consistently
better in identifiable regions. Here, all arms began from the same target episode
and adapted for the same short budget. `L` already fit most support/query regimes;
the additional modules were often redundant, undertrained, or harmful. The router
cannot learn stable specialization that the experts do not possess.

This is the most important lesson from the result: **test oracle action headroom
before optimizing the selector**.

#### B. Residual structure is not counterfactual utility

Residual autocorrelation and channel correlation describe what the frozen
Electricity-trained head gets wrong. They do not directly reveal whether a
particular adapter will reduce query error after a specific optimizer trajectory.
The residuals may encode scale, dataset identity, or source-head mismatch instead
of frequency/channel adapter responsiveness.

#### C. Two independent gates mis-specify action interactions

Frequency and channel utility were converted into two marginal binary labels and
thresholded separately. This assumes near-additive module contributions. It cannot
naturally express synergy, interference, or uncertainty among all four actions.
The repeated selection of a never-oracle `LC` route and the ETTh1 counterexample
above are direct evidence against the current factorization.

#### D. The meta-dataset is too small and imbalanced

Twenty-four source episodes per fold are not 24 independent domains: episodes from
the same dataset are correlated. Seven features per gate, very few positive labels,
and a heterogeneous 13-dataset transfer problem make a 0.5 logistic threshold
high-variance even with regularization.

#### E. The optional path is initially incompatible with the source head

The source head was trained only on the base MOMENT representation. The local
paper-specified frequency/channel path is non-residual and randomly initialized,
so enabling it changes the representation delivered to that head before learning
begins. The very large step-zero losses make this a leading explanation for why
short episodic adaptation favors `L`. Conventional Time-PEFT jointly trains a fresh
target head with the adapters and does not inherit the same compatibility burden.

#### F. One schedule is matched but not equally suitable

Using the same fixed updates and optimizer protocol is fair for a controlled action
comparison, but not necessarily optimal for every module. Channel adapters can add
millions of channel-specific parameters while seeing only 64 support windows. They
may require different learning rates, regularization, warm-up, or more data. If so,
the action definition must include those settings; otherwise the router is choosing
architectures whose training quality is unequal.

The worsening from frozen start on many units also suggests support overfit. An
internal 48/16 support train/validation split and learning curves are needed before
attributing every loss to adapter capacity.

#### G. Conventional Time-PEFT and episodic transfer are different regimes

Time-PEFT's headline results use abundant target training, validation-based LR
selection, and a jointly trained fresh head. Our router starts from an
Electricity-trained head, receives 64 support windows, and uses a fixed adapter LR.
The conventional screen proves that `LFC` sometimes helps locally; episodic `L`
dominance does not refute the accepted paper or resolve exact baseline parity.

## 10. Comparable work and what it suggests

Published effect sizes below are not compared numerically with ours because the
backbones, data access, routing units, and compute budgets differ.

| Work | How adaptation is selected or composed | Relevance to the next design |
| --- | --- | --- |
| [LoRA](https://openreview.net/forum?id=nZeVKeeFYf9) (Hu et al., ICLR 2022) | Fixed low-rank weight updates | Our strongest operational baseline; every new selector must beat it |
| [MOMENT](https://proceedings.mlr.press/v235/goswami24a.html) (Goswami et al., ICML 2024) | Open TSFM backbone, not a router | Primary backbone and source of the forecasting-head boundary |
| [Time-PEFT](https://openreview.net/forum?id=n8seTOinYs) (Na et al., ICML 2026) | Fixed LoRA + frequency + channel stack motivated by complexity | Parent adapter architecture; its complexity metrics are not routing labels |
| [Time-LlaMA / DynaLoRA](https://aclanthology.org/2025.acl-srw.90/) (Zhang et al., ACL SRW 2025) | Chooses LoRA modules per input and Transformer layer | Closest published dynamic time-series PEFT analogue, though it uses an LLM backbone |
| [TRACE](https://arxiv.org/abs/2503.16991) (Li and Zhu, 2026) | Estimates LoRA-module importance and masks modules during target adaptation | A target-specific selective-PEFT baseline; trades amortized routing for target computation |
| [AT4TS](https://openreview.net/forum?id=U54YyLn8MX) (Tomar et al., TMLR 2025) | Target-specific HPO over fine-tuning choices | Strong search baseline with a different, higher target-time information budget |
| [MixFT](https://arxiv.org/abs/2603.02840) (Lee et al., 2026 preprint) | Discovers subdomains, trains one LoRA expert per component, then routes contexts | Key lesson: construct specialists and the routing representation from the same partition |
| [CoRA](https://openreview.net/forum?id=JRlNrcTllN) (Cheng et al., ICLR 2026) | Learns time-varying and invariant channel relations end to end | Tests whether learned correlation representations are better than hand summaries |
| [UniPELT](https://aclanthology.org/2022.acl-long.433/) (Mao et al., ACL 2022) | Jointly gates LoRA, prefix, and adapter submodules | Evidence for joint module/gate training rather than a post-hoc gate over weak experts |
| [AdaMix](https://aclanthology.org/2022.emnlp-main.388/) (Wang et al., EMNLP 2022) | Stochastically trains a mixture of adapters or LoRA experts | Expert diversity is created during training, not assumed after independent adaptation |
| [AdapterFusion](https://aclanthology.org/2021.eacl-main.39/) (Pfeiffer et al., EACL 2021) | Trains task adapters first, then learns contextual fusion | Separates knowledge extraction from composition; both stages have explicit objectives |
| [Moirai-MoE](https://arxiv.org/abs/2410.10469) (Liu et al., NeurIPS 2024) | Token-level sparse experts learned during TSFM pretraining | Demonstrates much richer specialization data and granularity than our 24-unit LODO fits |
| [ELF](https://proceedings.mlr.press/v267/lee25ag.html) (Lee et al., ICML 2025) | Combines a frozen FM forecast with a cheap online forecaster | A relevant efficiency alternative when full parameter adaptation saves little wall time |

The common pattern in successful routing/composition work is **selector--expert
alignment**: experts are trained to specialize, or selection is optimized jointly
with them, or target validation/search directly measures their utility. Our current
pipeline independently trains generic action variants and asks inexpensive summary
statistics to recover a weak and imbalanced counterfactual label afterward.

## 11. Recommended next experiments

### Phase A: zero-new-GPU decision analysis

Use the 520 cached full-run records before generating more utilities.

1. **Oracle-headroom audit.** Average matched seeds within each unit, then bootstrap
   `oracle - L` by dataset cluster and episode under the predefined paired estimand.
   Treat near-ties as equivalent rather than forcing a winner.
2. **Label-stability audit.** Estimate the probability that each action beats `L`
   from seed-level outcomes; quantify how often labels change across update budgets.
3. **Nested source-only model comparison.** Under the same LODO folds, compare
   constant `L`, current gates, a four-action utility regressor, an uncertainty-aware
   regressor, and an abstain-to-`L` rule. Hyperparameters and thresholds must be
   selected only inside source folds.
4. **Feature ablations.** Compare raw input statistics, frozen residual features,
   MOMENT embeddings, spectral entropy, early support gradients, and short probe
   updates. Report calibration and regret, not only action accuracy.

**Stop condition:** if held-out query-oracle headroom over `L` under the predefined
paired estimand does not reach a preregistered practical margin (proposed: 5%) after
tie handling, do not train another router on this action bank. This headroom is a
descriptive feasibility ceiling, not a deployable or cross-validated result.

### Phase B: establish useful experts on development data

Before touching a new confirmatory set, run a no-router path and learning-curve
diagnostic. Use `L/LF/LC/LFC`, steps `{0, 10, 30, 100, 300, 1000}`, a 48/16
internal support train/validation split, a small LR grid, source versus fresh heads,
and the current non-residual path versus an explicitly labeled residual/identity
extension. A compact first panel is ECGCA515, Lorenz, DoublePendulum, and Weather.
Then create a source/development protocol that tests whether optional modules can
specialize.

1. Start with `L` versus `LF`. Retire `LC` until a channel variant demonstrates
   positive oracle contribution on held-out development units.
2. Cross update budgets `{50, 100, 300, 1000}` with source-only learning-rate and
   regularization selection. Condition any later router on the selected budget.
3. Compare the Electricity head with a fresh target head and a support-precalibrated
   head. This isolates whether source-head mismatch dominates residual evidence.
4. Validate partial-arm architecture and optimization independently. A CoRA-style
   learned correlation adapter or ChaTSFM-style gated refinement is a more credible
   channel candidate than continuing to tune a gate over a never-winning expert.
5. Construct specialists explicitly: cluster residual/backbone embeddings or
   temporal subdomains, train adapters for those regions, and use the same
   representation for routing, following the design principle in MixFT.

### Phase C: Router v2, only if Phase B passes

1. Replace independent binary gates with a joint four-action continuous utility or
   regret predictor.
2. Predict uncertainty and default to `L` unless the lower confidence bound for an
   optional action exceeds a preregistered gain threshold.
3. Include support size, channel count, head provenance, and update budget as
   explicit context.
4. Compare hard routing with soft composition/joint gating inspired by UniPELT,
   AdaMix, and AdapterFusion.
5. Train on substantially more independent source regimes. More windows from the
   same two episodes do not substitute for more datasets, domains, or non-overlapping
   temporal regimes.

### Phase D: untouched confirmation

Reserve h192/h336 and at least one dataset-family split before Router v2 development
results are inspected. Preregister:

- datasets, episodes, seeds, source-head provenance, and target partitions;
- `L` as the primary baseline and `LFC` as a secondary always-on comparator;
- Time-LlaMA/DynaLoRA-, TRACE-, or AT4TS-inspired setting-matched baselines where
  implementation permits;
- seed-first paired estimands and dataset-hierarchical confidence intervals;
- timing repetition/randomized arm order and uncertainty for every cost endpoint;
- all thresholds and fallback rules.

Only after a MOMENT result survives this stage should the study extend to Chronos,
TimesFM, classification, or EMG.

### Proposed go/no-go gates

| Gate | Minimum requirement before expansion |
| --- | --- |
| Conventional viability | `LFC` is competitive with `L` under a target-trained, validation-tuned parity protocol |
| Expert viability | Optional action has at least 5% reproducible oracle gain over `L`, not just `LFC` |
| Routing value | Upper 95% bound for router-vs-`L` relative MSE is below 0 |
| Calibration | Optional activation is suppressed on near-ties; action stability reaches 80%; regret improves over histogram-matched random |
| Budget robustness | Direction holds at two preregistered budgets, or budget is an explicit router input |
| Non-degeneracy | More than one action is selected and no selected action is never oracle-useful |
| Efficiency | End-to-end reduction reaches 10%, or the paper explicitly pivots away from a runtime claim |

## 12. Viable paper directions from here

### Direction 1: revised conditional-PEFT method

This remains viable only if expert specialization is created first and Router v2
beats fixed `L` on an untouched set. The novelty would be uncertainty-aware,
budget-conditioned counterfactual utility prediction for genuinely specialized
time-series adapters.

### Direction 2: rigorous negative-results/benchmark paper

The current result already supports a useful thesis:

> Routing cannot rescue an action bank whose optional experts lack stable oracle
> headroom; beating an always-on baseline is insufficient evidence of routing value.

A strong version would add repeated timings, one more backbone, stronger dynamic
PEFT/search baselines, label-stability analysis, and a preregistered replication.
The contribution would be the matched episodic benchmark, leakage-safe utility
store, control hierarchy, and empirical account of selector--expert misalignment.

### Direction 3: simpler adaptation-budget controller

If optional F/C adapters remain unhelpful, route over decisions with real systems
headroom: no update, head-only, LoRA, and update count. The historical result that
`A0` was best fixed and the current result that `L` dominates suggest that deciding
**whether and how long to adapt** may be more valuable than selecting small optional
branches.

## 13. What may and may not be claimed now

### Supported as preliminary development evidence

- Full `LFC` is not the best episodic policy under the tested short budgets.
- The routed mixture is superior to always-on `LFC` in the full development suite.
- Fixed `L` is superior to the current learned router under the same support-only
  information boundary.
- The current residual-correlation features do not provide useful assignment beyond
  route frequency in this action bank.
- Large active-parameter reductions can coexist with negligible end-to-end savings.

### Not supported

- The router improves Time-PEFT beyond LoRA.
- Residual correlations predict frequency/channel adapter utility.
- Time-PEFT generally loses to LoRA.
- The accepted Time-PEFT paper has been exactly reproduced.
- The result is confirmatory, backbone-general, horizon-general, or deployment-ready.

## 14. Reproducibility map and limitations

Principal committed entry points:

- [Time-PEFT reproduction protocol](TIME_PEFT_REPRODUCTION.md)
- [Eight-hour routing protocol](ROUTER_TIMEPEFT_8H.md)
- [`time_peft_budget24` configuration](../configs/time_peft_budget24.yaml)
- [`router_timepeft_8h` configuration](../configs/router_timepeft_8h.yaml)
- [24-hour conventional launcher](../scripts/run_time_peft_budget24.sh)
- [Full routing launcher](../scripts/run_router_timepeft_8h.sh)
- [Historical MVP report](../reports/mvp_report.md)

Local raw evidence paths:

- `artifacts/time-peft-budget24/*/paper-reproduction/reports/`
- `artifacts/correlation-pilot/correlation/reports/`
- `artifacts/router-timepeft-8h/correlation/reports/`
- `artifacts/router-timepeft-8h-u300/correlation/reports/`

The main limitations are material. The conventional screen is reduced to h96, two
seeds, two LRs, and capped epochs. The routing suite has only two independent
episodes per dataset, one backbone, one source head, and a protocol designed after
earlier development outcomes. The 300-update sensitivity also changes the seed set,
so budget and seed effects are partly confounded. Cost reductions have point
estimates but no paired uncertainty, and the paper-specified adapter necessarily
contains local choices absent from public Time-PEFT materials. These facts make the
current work a development result and design audit, not a final paper experiment.

## 15. References

- Y. Mao et al. [UniPELT: A Unified Framework for Parameter-Efficient Language
  Model Tuning](https://aclanthology.org/2022.acl-long.433/). ACL, 2022.
- J. Pfeiffer et al. [AdapterFusion: Non-Destructive Task Composition for Transfer
  Learning](https://aclanthology.org/2021.eacl-main.39/). EACL, 2021.
- Y. Wang et al. [AdaMix: Mixture-of-Adaptations for Parameter-efficient Model
  Tuning](https://aclanthology.org/2022.emnlp-main.388/). EMNLP, 2022.
- E. J. Hu et al. [LoRA: Low-Rank Adaptation of Large Language
  Models](https://openreview.net/forum?id=nZeVKeeFYf9). ICLR, 2022.
- M. Goswami et al. [MOMENT: A Family of Open Time-series Foundation
  Models](https://proceedings.mlr.press/v235/goswami24a.html). ICML, 2024.
- X. Liu et al. [Moirai-MoE: Empowering Time Series Foundation Models with Sparse
  Mixture of Experts](https://arxiv.org/abs/2410.10469). NeurIPS, 2024.
- J. Zhang et al. [Time-LlaMA: Adapting Large Language Models for Time Series
  Modeling via Dynamic Low-rank Adaptation](https://aclanthology.org/2025.acl-srw.90/).
  ACL Student Research Workshop, 2025.
- S. Tomar et al. [AT4TS: Autotune for Time Series Foundation
  Models](https://openreview.net/forum?id=U54YyLn8MX). TMLR, 2025.
- T. L. Lee et al. [Lightweight Online Adaptation for Time Series Foundation Model
  Forecasts](https://proceedings.mlr.press/v267/lee25ag.html). ICML, 2025.
- Y. Li and W. Zhu. [TRACE: Time Series Parameter Efficient
  Fine-Tuning](https://arxiv.org/abs/2503.16991). Neurocomputing, 2026.
- J. Na et al. [Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning
  for Time-Series Foundation Models](https://openreview.net/forum?id=n8seTOinYs).
  ICML, 2026.
- H. Cheng et al. [CoRA: Boosting Time Series Foundation Models for Multivariate
  Forecasting through Correlation-aware Adapter](https://openreview.net/forum?id=JRlNrcTllN).
  ICLR, 2026.
- T. L. Lee, E. M. Ponti, and A. Storkey. [Adapting Time Series Foundation Models
  through Data Mixtures](https://arxiv.org/abs/2603.02840). arXiv preprint, 2026.
