# Compute and parameter analysis

This note separates quantities that can be derived from the accepted Time-PEFT
architecture from quantities that must be measured. The analytical result is a
capacity and operation-count argument; it is not a substitute for synchronized
GPU timing.

## Symbols

| Symbol | Meaning |
| --- | --- |
| \(B\) | adaptation batch size |
| \(C\) | number of channels |
| \(K\) | number of backbone patches |
| \(h_1\) | backbone hidden size |
| \(h_2\) | frequency-adapter output size; paper default \(h_2=h_1\) |
| \(r_c\) | channel-adapter rank; paper default \(r_c=h_1/2\) |
| \(r_L\) | LoRA rank; paper default 8 |
| \(M\) | number of attention layers receiving LoRA |
| \(U\) | fixed number of adaptation updates |
| \(q\) | maximum correlation lag; default 8 |
| \(P_0\) | head + q/k/v LoRA parameters active in every arm |
| \(g_F,g_C\) | binary frequency and channel activation decisions |

Multiply-add FLOP formulas below count one multiplication and one addition as two
operations. Libraries and profilers sometimes use a one-operation convention, so
compare ratios only after checking the convention.

## Exact adapter parameter formulas

Section 4.5 of the accepted paper excludes bias and normalization parameters and
gives

\[
P_F=h_1h_2
\]

for the frequency projection and

\[
P_C=r_c(h_1+h_2)+C r_c h_1
\]

for the shared channel down-projection plus \(C\) channel-specific up-projections.
The repository's paper adapters use bias-free projections, so these formulas are
exact for those adapter tensors.

For square q/k/v attention projections, LoRA contributes

\[
P_{\mathrm{LoRA}}=M\cdot 3\cdot r_L(h_1+h_1)
=6Mr_Lh_1.
\]

For non-square projections, use the exact sum

\[
P_{\mathrm{LoRA}}
=\sum_{j\in\mathcal{J}_{qkv}} r_L
\left(d_{in,j}+d_{out,j}\right)
\]

over injected modules. The forecasting-head term depends on horizon and backbone,
so the implementation counts it directly. Define

\[
P_0=P_{\mathrm{head}}+P_{\mathrm{LoRA}}.
\]

The active trainable parameters by route are therefore

| Route | Active trainable parameters |
| --- | ---: |
| `L` | \(P_0\) |
| `LF` | \(P_0+P_F\) |
| `LC` | \(P_0+P_C\) |
| `LFC` | \(P_0+P_F+P_C\) |

The paper defaults \(h_2=h_1\) and \(r_c=h_1/2\) reduce the optional terms to

\[
P_F=h_1^2,
\qquad
P_C=h_1^2\left(1+\frac{C}{2}\right),
\qquad
P_F+P_C=\frac{C+4}{2}h_1^2.
\]

For MOMENT-base, \(h_1=768\), so \(h_1^2=589{,}824\):

| Channel count | Datasets in suite | \(P_F\) | \(P_C\) | Optional `LFC` total |
| ---: | --- | ---: | ---: | ---: |
| 3 | Lorenz | 589,824 | 1,474,560 | 2,064,384 |
| 4 | DoublePendulum | 589,824 | 1,769,472 | 2,359,296 |
| 6 | CellCycle, Hopfield, LorenzCoupled, ECG datasets | 589,824 | 2,359,296 | 2,949,120 |
| 7 | ETT datasets | 589,824 | 2,654,208 | 3,244,032 |
| 8 | Exchange | 589,824 | 2,949,120 | 3,538,944 |
| 21 | Weather | 589,824 | 6,782,976 | 7,372,800 |

As a sanity check, the paper reports approximately 2.360M trainable parameters for
LoRA and 4.429M for Time-PEFT on three-channel Lorenz. The analytical optional
increment above is 2.064M; the small remaining difference is consistent with
terms excluded from the paper's simplified formula.

## Expected active-parameter saving

Let \(\pi_F=\Pr(g_F=1)\) and \(\pi_C=\Pr(g_C=1)\) over deployment episodes. Because
the optional terms are additive, their dependence does not affect the expectation:

\[
\mathbb{E}[P_{\mathrm{router}}]
=P_0+\pi_F P_F+\pi_C P_C.
\]

Relative to always-on `LFC`, the expected saving is

\[
\Delta P=(1-\pi_F)P_F+(1-\pi_C)P_C,
\]

\[
R_P=\frac{\Delta P}{P_0+P_F+P_C}.
\]

`paper_time_peft_parameter_savings` implements these formulas and also returns the
four route-specific counts.

This is an **active trainable-parameter** result. If a service stores a complete
bank containing both adapters and only toggles execution, the stored model remains
\(P_0+P_F+P_C\). Storage falls only when the deployed artifact omits inactive
modules or stores route-specific checkpoints.

## Adapter operation counts

Let \(n=B C K\) be the number of channel-patch tokens processed by a projection.

### Frequency adapter

The frequency path performs:

1. FFTs over the patch axis and an inverse FFT after masking:
   \(\Theta(B C h_1 K\log K)\);
2. amplitude reduction and top-k selection:
   \(\Theta(B C h_1 K)\) plus selection overhead;
3. the \(h_1\to h_2\) projection:

\[
F_{F,\mathrm{forward}}^{\mathrm{linear}}=2nh_1h_2.
\]

For a trainable dense projection, forward plus gradients with respect to weights
and inputs are approximately three forward-linear costs:

\[
F_{F,\mathrm{train}}^{\mathrm{linear}}\approx6nh_1h_2.
\]

FFT backward work is also present and retains
\(\Theta(B C h_1 K\log K)\) scaling. Constants depend on the complex FFT
kernel.

### Channel adapter

The shared down-projection maps \(h_1+h_2\) to \(r_c\); each channel-specific
up-projection maps \(r_c\) to \(h_1\). Its forward projection work is

\[
F_{C,\mathrm{forward}}
=2 n r_c(h_1+h_2)+2 n r_c h_1
=2 n r_c(2h_1+h_2).
\]

The corresponding training approximation is

\[
F_{C,\mathrm{train}}\approx6 n r_c(2h_1+h_2),
\]

excluding ReLU, dropout, concatenation, layer normalization, and optimizer work.

### LoRA

For one \(d_{in}\to d_{out}\) projection with rank \(r_L\), LoRA adds

\[
F_{L,\mathrm{forward}}=2 N r_L(d_{in}+d_{out})
\]

for \(N\) tokens and roughly three times that amount for forward plus backward.
LoRA and the forecasting head are common to all four matched arms and therefore
cancel in the optional-adapter comparison.

The total per-step training work can be represented as

\[
F_{\mathrm{step}}(g_F,g_C)
=F_0+g_F F_F+g_C F_C,
\]

where \(F_0\) includes the frozen backbone execution, activation-gradient path,
LoRA, head, loss, and optimizer work. Since \(F_0\) can dominate, the percentage
FLOP and wall-time reductions will normally be smaller than the optional-parameter
reduction.

An inactive adapter must be skipped in the forward graph to realize these savings.
Setting `requires_grad=False` while still executing the adapter saves optimizer and
weight-gradient work but not its forward or activation-gradient work. The current
model branches on `frequency_enabled` and `channel_enabled`.

## Correlation-selector cost

Evidence extraction includes one frozen support forecast and residual statistics.
For residual tensor \(E\in\mathbb{R}^{B\times C\times H}\):

- zero-lag pairwise channel correlations cost \(O(BC^2H)\);
- correlations through \(q\) lags cost \(O(qBC^2H)\);
- the channel correlation effective rank adds a \(C\times C\) eigendecomposition,
  \(O(C^3)\);
- two logistic probability calls cost \(O(d_F+d_C)\) after feature preprocessing.

Thus, excluding the frozen forecast shared with the evidence stage,

\[
F_{\mathrm{corr}}=O((q+1)BC^2H+C^3).
\]

The maximum channel count in the evaluation suite is 21, so this is small relative
to MOMENT-base adaptation. Electricity has 321 channels but is source-head-only
and is not routed in the target benchmark.

The accepted paper's multichannel-complexity diagnostic evaluates pairwise transfer
entropy and maximizes it over every lookback lag. Correlation has a cheaper
estimator and uses only eight lags by default. However, Time-PEFT does not require
that diagnostic inside each always-on target training run. Therefore, the valid
end-to-end comparison charges the router for its correlation evidence and charges
`LFC` no transfer-entropy cost.

## End-to-end break-even condition

Let \(t_0,t_F,t_C\) be per-update time for the common path and two optional paths,
and let \(T_E,T_R\) be evidence and routing time. Then

\[
T_{\mathrm{LFC}}=U(t_0+t_F+t_C),
\]

\[
T_{\mathrm{router}}=T_E+T_R
+U(t_0+g_Ft_F+g_Ct_C).
\]

For one selected route, the router is faster exactly when

\[
U\big[(1-g_F)t_F+(1-g_C)t_C\big]>T_E+T_R.
\]

Its update-count break-even point is

\[
U_{BE}=\frac{T_E+T_R}
{(1-g_F)t_F+(1-g_C)t_C},
\]

provided at least one adapter is inactive. `LFC` selections cannot produce a
per-target speedup because they add selection overhead to the same adaptation.
Report route-conditioned latency and the proportion routed to `LFC`; an overall
mean alone can hide this fact.

## Training-memory bounds

Skipping optional trainable parameters removes their gradients and AdamW moment
states. A useful implementation-independent expression is

\[
M_{\mathrm{persistent}}\approx
(b_w+b_g+2b_s)P_{\mathrm{active}},
\]

where \(b_w,b_g,b_s\) are bytes per weight, gradient, and optimizer-state value.
With FP32 weights, gradients, and moments this is approximately 16 bytes per active
parameter, excluding allocator overhead. Mixed-precision master copies can change
the constant.

Conditional execution can also avoid optional activations with leading shapes

- frequency output: \(B\times C\times K\times h_2\);
- channel bottleneck: \(B\times C\times K\times r_c\);
- channel output: \(B\times C\times K\times h_1\).

Autograd retention and kernel fusion determine which tensors dominate in practice.
Use `torch.cuda.max_memory_allocated`, and report `max_memory_reserved` separately
if allocator behavior matters. Do not infer a peak-memory percentage from parameter
counts.

## What the current measurements include

`evaluate_action` currently measures:

- `wall_time_s`: the fixed optimization loop only;
- `profiled_flops`: a separate profiler pass for one optimization step, multiplied
  by \(U\); the profiler pass is outside `wall_time_s`;
- `peak_memory_mb`: peak allocated CUDA memory after reset, including the frozen
  query pass, optimization, profiler/query work observed before the final read;
- `evidence_wall_time_s`: separately timed evidence extraction, excluding support-
  tensor/template placement on the target device and template restoration to its
  original device;
- query MSE/MAE: evaluation outside adaptation wall time.

Model cloning, LoRA injection, optimizer construction, source-head loading, data
loading, and query evaluation are not in adaptation time. Router aggregate
end-to-end time is evidence + gate inference + adaptation; it is a warm target-
adaptation boundary, not cold process startup. If deployment startup matters, add
a second outer wall-clock metric rather than redefining the existing field.

PyTorch's FLOP profiler does not guarantee complete accounting for FFT, top-k,
normalization, elementwise operations, or optimizer kernels. Treat its output as a
consistent instrumentation estimate and publish the formulas above beside it.

## Measurement checklist

For every reported timing table record:

- GPU model, driver, CUDA, PyTorch, precision, and power mode;
- batch size, support size, horizon, patch count, update count, and channel count;
- warm-up procedure and explicit CUDA synchronization;
- isolated process or co-tenant status;
- matched seed and batch-index stream across arms;
- route execution order or randomized/balanced ordering;
- evidence, routing, adaptation, and cold-start boundaries separately;
- mean, median, dispersion, and paired confidence interval;
- analytical parameters, counted parameters, profiler FLOPs, and measured memory
  as distinct columns.

Only synchronized end-to-end timing can establish a computational benefit.
