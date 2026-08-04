# Eight-hour router-on-Time-PEFT development protocol

## Evidence status

This is a **preliminary, post-hoc development experiment**. The horizon and routing
protocol were finalized after inspecting the conventional horizon-96 LoRA/Time-PEFT
development results and an earlier router pilot on Lorenz, DoublePendulum,
ECGCA515, ETTh1, and Weather, while the matrix retains all 13 paper datasets. It
can support a paper direction and decide whether an untouched, preregistered
protocol is warranted; it cannot provide a confirmatory claim.

The conventional reproduction and this episodic routing extension remain separate
protocols. The reproduction trains on target train windows, selects on validation,
and evaluates the conventional test split once. This experiment uses labeled,
raw-disjoint support/query episodes inside the test partition and fits every router
fold without the held-out dataset.

## Locked primary matrix

- datasets: all 13 Time-PEFT datasets at horizon 96: Lorenz, CellCycle,
  DoublePendulum, Hopfield, LorenzCoupled, ECGCA115, ECGCA515, ETTh1, ETTh2,
  ETTm1, ETTm2, Weather, and Exchange;
- lookback/horizon: 96/96;
- two chronological non-overlapping episodes per dataset;
- 64 labeled support windows followed by 128 query windows;
- seeds 0, 1, and 2;
- arms `L`, `LF`, `LC`, and `LFC`, with identical initialization, mini-batches,
  optimizer schedule, and 100 fixed updates;
- count-inferred paper adapter implementation, rank-8/alpha-32 Q/K/V LoRA,
  frequency top-k 3, adapter/head dropout 0.1, and FP32 adaptation;
- official pinned `dysts==0.96` trajectories of length 12,000, seed 0, and 100
  points per period;
- an Electricity-trained horizon-96 source head whose provenance excludes all
  13 evaluation datasets;
- fixed router choices: lag 8, 0.2% minimum marginal benefit, gate threshold 0.5,
  and random state 17.

The primary matrix contains 26 held-out dataset/episode units and 312 matched
action/seed records. Seeds are averaged inside a unit before any relative
comparison.

## Sensitivity matrix

After the primary report, a separate artifact tree repeats the same 26 units with
seeds 0 and 1 (208 matched records) and 300 fixed updates. It reuses only the
provenance-bound source head. This tests whether the routing conclusion is an
artifact of short target adaptation.

## Estimand and controls

For held-out unit `u`, the primary accuracy estimand is

`d_u = (MSE_router,u - MSE_LFC,u) / MSE_LFC,u`,

where `LFC` is always-on paper-specified Time-PEFT. Report the equal-unit mean and
a paired 95% hierarchical bootstrap interval that resamples datasets and then
episodes. Query labels never enter evidence, preprocessing, gate fitting, or
threshold selection for the held-out dataset.

Required controls are always-on `LFC`, the best source-selected fixed arm, and a
query-informed oracle explicitly labeled as an unattainable upper bound. Report
all route counts, gate one-class folds, MSE/MAE, evidence + routing + adaptation
time, active trainable parameters, peak memory, and profiled FLOPs.

## Interpretation locked before execution

- **Preliminary superiority:** the upper 95% bound for mean `d_u` is below zero.
- **Preliminary noninferiority:** the upper 95% bound is below +1%.
- **Directional only:** the point estimate is at most +2%, but its upper interval
  does not establish noninferiority.
- **Directional efficiency diagnostic:** router evidence, decision, and
  selected-arm adaptation together are faster than `LFC`; a 10% point reduction
  is the promising target. This run does not attach an uncertainty interval to
  timing, so it cannot by itself establish a formal efficiency claim.
- **Routing value:** at least two actions are selected, routing does not collapse
  to `LFC`, and the router improves on the source-selected fixed arm.
- **Robustness:** the 100- and 300-update conclusions have the same direction.

If only the query-informed oracle improves, or if the learned router loses to the
source-fixed arm, the experiment does not support the proposed routing mechanism.

## Completed result

Both matrices completed on one NVIDIA A40. At 100 updates, the router improved
relative MSE over `LFC` by 17.44% (hierarchical 95% CI 6.29% to 30.63%), but was
102.11% worse than source-fixed `L` and 48.18% worse than histogram-matched random
routing. At 300 updates, the corresponding results were a 15.15% improvement over
`LFC`, a 75.89% loss to fixed `L`, and a 33.05% loss to random routing. End-to-end
time fell by only 0.84% and 1.27%.

The locked routing-value criterion therefore failed at both budgets. The result
supports omitting optional Time-PEFT modules in this episodic regime, but does not
support the residual-correlation assignment mechanism. See the
[current-results report](CURRENT_RESULTS_REPORT.md) for the full analysis and next
decision gates.
