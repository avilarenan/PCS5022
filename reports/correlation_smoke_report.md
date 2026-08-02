# Correlation-routing CPU smoke result

This run is an execution check, not scientific evidence. It used the tiny random-head
backbone, three deterministic compatible generators (Lorenz, DoublePendulum, Hopfield),
two episodes per dataset, one seed, horizon 8, and two adaptation updates. The accepted
Time-PEFT experiment requires a pretrained MOMENT backbone, a provenance-checked source
forecasting head, horizons 96/192/336, and three seeds.

## End-to-end result

| Quantity | Correlation route | Always-on L+F+C | Relative change |
| --- | ---: | ---: | ---: |
| Query MSE | 25.3746 | 25.3595 | +0.02% mean paired/unit |
| Query MAE | 4.12271 | 4.11136 | +0.53% mean paired/unit |
| Adaptation time | 0.01663 s | 0.01082 s | 53.64% higher |
| End-to-end time | 0.02074 s | 0.01082 s | 91.66% higher |
| Trainable parameters | 17,488 | 18,426.7 | 5.09% lower |

The LODO gates chose `LC` three times, `LF` once, and `LFC` twice. The corrected
paired estimand first averages seeds inside each dataset/horizon/episode and then
gives each unit equal weight. Its MSE point estimate was inside the configured 1%
development margin, but this smoke run has no confidence interval and cannot
establish noninferiority. The support forecast, evidence overhead, and noisy two-step
arm timings made the routed path slower on CPU. That unfavorable timing is retained;
only the 100-update GPU workflow can test whether skipped adapter work amortizes the
selection pass.

This result was regenerated after correcting the `LFC` implementation to the
accepted paper's Algorithm 1 dataflow: frequency output and backbone embeddings feed
the channel adapter, whose output is normalized and passed directly to the forecast
head. The partial `LF` and `LC` masks remain explicitly documented routing ablations.

## Evidence microbenchmark

The warmed CPU comparison uses PyTorch 2.4.1 with one thread on an AMD EPYC 9V74 host.
The correlation path computes its complete feature suite through lag 8. The existing
linear-Gaussian transfer-entropy proxy computes lag 1 only, making this conservative
relative to the paper's maximum over all lags.

| Shape `[B,C,T]` | Correlation suite | Gaussian TE, lag 1 | TE / correlation |
| --- | ---: | ---: | ---: |
| `[64,21,96]` | 0.0202 s | 0.0892 s | 4.42x |
| `[32,64,96]` | 0.0475 s | 0.6386 s | 13.44x |

Both implementations are quadratic in channel count for fixed lag count. The observed
benefit is vectorization and avoiding one regression solve per ordered channel pair, not
an asymptotic improvement. The stronger end-to-end hypothesis is that routing can skip
frequency/channel training after this cheap evidence pass; only the GPU workflow can test
whether the saved adapter work exceeds the evidence overhead.

Raw microbenchmark outputs are in `evidence_microbenchmark_cpu.json` and
`evidence_microbenchmark_cpu_c64.json`. The full smoke utility store is reproducible with
`./scripts/run_cpu_smoke.sh` and is intentionally excluded from Git.
