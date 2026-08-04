# Time-PEFT versus LoRA reproduction protocol

This workflow addresses a prerequisite for the routing study:

> Does a paper-specified implementation recover the accepted paper's finding
> that full Time-PEFT improves on LoRA on complex time-series datasets?

It is independent of support/query episodes, the Electricity source head,
correlation evidence, gates, and leave-one-dataset-out routing. The accepted
method is not a router: its complexity measures diagnose datasets, while its
training path jointly activates LoRA, the frequency adapter, the channel
adapter, and the forecasting head.

No author implementation has been located or numerically matched. Every local
result must therefore be called a **paper-specified Time-PEFT
reimplementation**, never an official reproduction.

## Accepted-paper claim and stated protocol

The accepted ICML 2026 paper is Jihye Na, Patara Trirat, Chanyoung Park, and
Jae-Gil Lee, *Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning
for Time-Series Foundation Models* (OpenReview `n8seTOinYs`). Its abstract says
that temporal complexity is based on spectral entropy, multichannel complexity
captures cross-channel information flow, and the two quantities are proxies for
fine-tuning gains. Time-PEFT itself uses a frequency adapter for top-k filtering
and a channel adapter for multichannel modeling. With MOMENT-base, the reported
improvement is up to 38% over LoRA on complex datasets.

The accepted paper states the following details used by this workflow:

- forecasting lookback 96 and horizons 96, 192, and 336;
- rank-8, scale-32 LoRA on query, key, and value projections for Transformer
  backbones;
- FFT over patches, retention of the three largest-amplitude frequency bins,
  inverse FFT, and an `h1 -> h2` frequency projection with `h2 = h1`;
- a channel adapter that concatenates backbone and filtered embeddings, uses a
  shared `(h1+h2) -> r` projection and channel-specific `r -> h1` projections,
  with `r = h1/2`;
- the full dataflow `Efilt=Freq(Eback)`, `Ech=Channel(Eback, Efilt)`, followed by
  `ForecastHead(LayerNorm(Ech))`;
- joint optimization of LoRA, the frequency adapter, the channel adapter, and
  the forecasting head;
- AdamW, a learning rate selected in the range `1e-3` to `1e-5`, default batch
  size 128, at most 100 epochs, and early stopping.

The paper reports these MOMENT-base MSE values for the complex suite:

| Dataset | LoRA h96/h192/h336 | Time-PEFT h96/h192/h336 |
| --- | --- | --- |
| Lorenz | .423 / .616 / .742 | .134 / .408 / .614 |
| CellCycle | .514 / .703 / .855 | .249 / .484 / .675 |
| DoublePendulum | .907 / .984 / 1.023 | .817 / .938 / .994 |
| Hopfield | .499 / .676 / .786 | .244 / .440 / .599 |
| LorenzCoupled | .619 / .829 / .943 | .228 / .532 / .723 |
| ECGCA115 | .573 / .788 / 1.003 | .462 / .683 / .900 |
| ECGCA515 | .199 / .255 / .323 | .125 / .181 / .250 |

These are post-run parity references, not tuning targets. Local learning rates
are selected from validation data only, and each selected checkpoint receives
one test evaluation.

## Details the accepted paper does not fix

The paper and currently available assets do not determine:

- the exact learning-rate candidates inside the stated range;
- weight decay, early-stop patience or minimum delta, scheduler, gradient
  clipping, random seeds, and numeric precision;
- adapter dropout and forecasting-head dropout probabilities;
- adapter and forecasting-head initialization;
- whether adapter projections have bias or output LayerNorm is affine;
- how frequency amplitudes are aggregated before top-k selection;
- exact dataset split boundaries, window stride, scaling, or whether
  validation/test lookback may use prior-split context;
- exact `dysts` version, trajectory parameters, initial conditions, and sampled
  trajectories for the five chaotic datasets.

These omissions prevent exact reproduction claims. The choices below are local
reconstruction assumptions and must not be attributed to the authors.

## Primary local reconstruction

The primary full-matrix configuration is locked to the following choices before
any full-matrix test result is inspected:

| Item | Primary local choice | Status |
| --- | --- | --- |
| Backbone | Pinned MOMENT-base revision | Paper-aligned and versioned |
| Target head | Fresh per dataset/horizon/seed; seed-matched across `L` and `LFC` | Local choice required by the public checkpoint |
| LR grid | `{1e-3, 1e-4, 1e-5}`; one LR per method/cell selected by mean validation MSE across seeds | Local reconstruction of the stated range |
| Optimizer | AdamW, weight decay `0.01`, no scheduler or gradient clipping | Local choice |
| Stopping | Maximum 100 epochs, patience 3, minimum delta 0 | Partly paper-stated, partly local |
| Precision | FP32 | Local choice |
| Windows | Stride one; batch 128 | Batch paper-stated, stride local |
| Scaling | Per-channel standardizer fitted only on target-train timestamps | Local choice |
| Complex-data split | Chronological 70/10/20; validation/test targets remain inside their split and may use prior history as context | Local choice |
| Synthetic generation | `dysts==0.96`; system-metadata parameters, initial conditions, and integration step; Fourier resampling; `postprocess=True`; no burn-in or noise; Radau at `rtol=atol=1e-12`; length 12,000 and `pts_per_period=100` | Pinned local reconstruction; trajectory details and length are not paper-stated |
| Adapter structure | `paper_count_inferred`: projection biases and affine output LayerNorm | Inferred from the paper's rounded parameter table |
| Channel-adapter dropout | `0.1` | Unstated; primary TSLib-style assumption |
| Forecast-head dropout | `0.1` | Unstated; primary TSLib/MOMENT-style assumption |
| Seeds | `0, 1, 2` | Local choice |

The resolved run manifest must contain every value above. Before a primary run,
verify that the runtime model actually consumes the resolved dropout fields; a
configuration entry by itself is not implementation evidence.

The reproduction workflow uses the official `dysts` package pinned at version
0.96 rather than the historical episodic workflow's local compatible
generators. The paper names the systems but does not publish the `dysts`
version, constructor arguments, initial conditions, random seed, sampling
density, trajectory length, or exact saved trajectories. The choices above are
therefore explicit local assumptions, not exact paper trajectories. With fixed
metadata initial conditions and zero noise, the configured seed does not change
these five trajectories; it is retained as explicit provenance. In particular,
`postprocess=True` maps the DoublePendulum angular coordinates through the
package's postprocessor.

The lock records the installed `dysts`, NumPy, SciPy, and optional Numba
versions, every effective trajectory argument, and the final tensor SHA-256 and
shape. Each generated tensor is atomically materialized under
`data/.utility_peft/dysts/` and its identity and SHA are revalidated on reuse,
so tune and test processes consume identical bytes without repeating the
expensive ODE solve. Numeric parity remains most meaningful first on the pinned
ECG recordings.

## Preregistered sensitivities

The following one-factor-at-a-time variants are required; they are not optional
post-hoc troubleshooting choices:

| ID | Change from the primary configuration | Ambiguity tested |
| --- | --- | --- |
| S1 | `model.adapter_implementation=paper` | Bias-free projections and non-affine LayerNorm versus the count-inferred structure |
| S2 | `model.adapter_dropout=0.0` | Channel-adapter dropout `0.1` versus `0.0` |
| S3 | `model.head_dropout=0.0` | Forecast-head dropout `0.1` versus `0.0` |

All other fields remain fixed. Run the variants first on the development anchor
to catch implementation failures. A paper claim that depends on any omitted
detail above is robust only if the relevant sensitivity is also executed on the
locked full matrix and preserves the direction of the `LFC`-versus-`L` result.
Anchor-only sensitivities must be reported as development checks, not evidence
of full-matrix robustness. If an interaction between dropout choices is visible
on the anchor, preregister the joint `adapter_dropout=0.0, head_dropout=0.0`
variant before opening any full-matrix test result.

## Development anchor and confirmatory boundary

ECGCA515 at horizon 96 is a **development/parity anchor**. The accepted paper's
MSE values, 0.199 for LoRA and 0.125 for Time-PEFT, make it useful for detecting
large implementation or preprocessing discrepancies. Its local test result may
be used to debug the reconstruction and decide whether the full experiment is
ready. It is not confirmatory evidence for the routing hypothesis, for a general
Time-PEFT advantage, or for a paper-level effect size.

Before the first test evaluation of the seven-dataset matrix, lock and archive:

- datasets, horizons, seeds, actions, checkpoint revisions, `dysts` version and
  trajectory arguments, and implementation hash;
- every primary assumption in the table above;
- sensitivity variants and their execution scope;
- split hashes, standardization rule, LR-selection rule, stopping rule, and
  aggregation/statistical decision rules.

Full-matrix tuning may be inspected because it uses train and validation data.
The full script uses the explicit `confirmatory` test role and requires a
matching protocol-lock artifact before test access. The anchor uses
`development-parity`, and the capped run uses `plumbing-smoke`; those roles are
part of run provenance and are not interchangeable. The CLI rejects a
`confirmatory` label unless the complete seven-dataset, three-horizon,
three-seed preregistered profile and official `dysts` path are intact. Once any
full-matrix test result is opened, a changed choice is a new experiment with a
new config and dataset hash; it cannot silently replace the locked result.

## Measured A40 workload for the anchor

The uncapped ECGCA515-h96 split contains 573,809 train windows, 81,905 validation
windows, and 163,905 test windows. At batch 128 this is 4,483 training batches
and 640 validation batches per epoch, plus 1,281 test batches per selected model.
The LR/method/seed Cartesian grid contains 18 trials.

On the current NVIDIA A40, an FP32 batch-128 planning benchmark measured:

| Method | Training batch, including measured data placement | Validation batch, including measured data placement | Estimated epoch |
| --- | ---: | ---: | ---: |
| `L` | about 0.404 s | about 0.179 s | about 32.1 min |
| `LFC` | about 0.414 s | about 0.183 s | about 32.9 min |

Peak allocated GPU memory was about 8.36 GiB. One epoch across all 18 trials is
about 9.75 GPU-hours. Patience 3 makes four epochs per trial the theoretical
minimum, so tuning is at least about 39 GPU-hours. Representative planning
scenarios are about 49 hours for five mean epochs, 98 hours for ten, 8.1 days
for twenty, and 41 days if all trials reach 100 epochs. Test evaluation is
approximately 23--30 minutes. These are planning measurements, not paper timing
results, and may vary by roughly 15% with host and device conditions.

The final tune artifact stores selected trainable states and should be about
64.4 MiB for three seeds. In addition, the required resume cache retains the
best state from every method/LR/seed trial: approximately 193 MiB of raw FP32
tensors, or about 194 MiB plus serialization overhead. Expect roughly 258 MiB
per completed ECGCA515-h96 cell for the trial cache and final tuning artifact
together.

Completed trials are written atomically and keyed by config fingerprint,
dataset and data hash, method, learning rate, seed, and template fingerprint.
Rerunning the identical tune stage reuses them; a mismatched identity is
rejected. The final LR selection occurs only after the exact trial grid is
reconstructed. Preserve the trial-cache directory until the tuning artifact is
validated. Per-trial resume is a scientific requirement for the anchor and full
matrix, not merely an execution optimization.

## Preliminary 24-hour screen

For a decision-quality result within one day, use the dedicated
`time_peft_budget24` configuration and its two-phase runner. Both phases keep
all stride-one train, validation, and test windows and share:

- horizon 96 only;
- seeds 0 and 1;
- `L` and `LFC`, with learning rates `1e-3` and `1e-4`;
- early-stopping patience three.

Phase 1 runs ECGCA515 for at most four epochs per trial, downloads the pinned
recording if necessary, and writes to `artifacts/time-peft-budget24/ecg`. Its
eight trials project to roughly 17.5 A40 hours from the measured anchor rate.
Only after its tune, test, and report stages finish does phase 2 run Lorenz,
CellCycle, DoublePendulum, Hopfield, and LorenzCoupled for at most eight epochs
per trial under `artifacts/time-peft-budget24/synthetic`. The longer synthetic
budget makes these short cells more informative while keeping the total close
to one day. Completion within 24 hours remains a planning target and requires
exclusive GPU access; it is not a hard runtime guarantee.

Run the isolated, resume-safe workflow with:

```bash
./scripts/run_time_peft_budget24.sh
```

The two phase-specific trees each contain an independent protocol lock, resume
cache, and report. Results are development/parity evidence: they can show
whether `LFC` beats `L` on this reduced matrix and can motivate the next run,
but they are preliminary and non-confirmatory. They do not estimate the paper's
complete seven-dataset, three-horizon effect and must not be substituted into
the locked full-matrix claim.

## Execution order

1. Run the explicitly capped plumbing smoke:

   ```bash
   ./scripts/run_time_peft_reproduction_smoke.sh
   ```

2. Run tune, test, and report for the uncapped ECGCA515-h96 development anchor:

   ```bash
   ./scripts/run_time_peft_anchor.sh
   ```

3. Run S1--S3 on the anchor and resolve implementation defects. Do not describe
   any anchor outcome as confirmatory.
4. Freeze the complete primary and sensitivity protocols, including statistical
   decision rules, before opening any full-matrix test value.
5. Run the locked full tune stage. Inspect validation metadata and completeness.
6. Run the locked full test stage exactly once, then render the report:

   ```bash
   ./scripts/run_time_peft_full.sh
   ```

Standard datasets form a later negative-transfer evaluation. The accepted paper
does not claim that Time-PEFT universally improves over LoRA on those datasets.
