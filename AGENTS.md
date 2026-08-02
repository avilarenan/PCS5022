# PCS5022 agent guide

This repository is a research prototype for testing whether a cheap, support-only
correlation router can omit unnecessary Time-PEFT adapters while retaining
forecasting accuracy. Treat reproducibility and leakage prevention as code-level
requirements, not documentation conventions.

## Scientific ground truth

- The accepted Time-PEFT paper is linked from `papers/README.md` through its
  canonical OpenReview entry. Local PDF copies are intentionally ignored.
- Time-PEFT does **not** route or selectively activate modules. Its temporal and
  multichannel complexity scores are dataset diagnostics. Its training algorithm
  always optimizes LoRA, the frequency adapter, the channel adapter, and the
  forecasting head together.
- The correlation router is this repository's extension. Never describe it as an
  implementation detail of the accepted paper.
- No official author implementation is included or validated in this repository.
  Use `paper-specified Time-PEFT reimplementation`, not `official reproduction`,
  until numerical parity against an authoritative release has been demonstrated.
- The five local chaotic generators are deterministic, protocol-compatible
  substitutes. They are not the paper's exact `dysts` trajectories.

Read `docs/EXPERIMENT.md` before changing the protocol and
`docs/COMPUTE_ANALYSIS.md` before making an efficiency claim.

## Matched comparison arms

Preserve these four arms and their shared initialization, data, optimizer,
mini-batches, seeds, and update count:

| Arm | Trainable modules |
| --- | --- |
| `L` | forecast head + LoRA |
| `LF` | `L` + frequency adapter; normalized `Eback + Efilt` F-both ablation |
| `LC` | `L` + channel adapter; zero tensor marks the absent frequency branch |
| `LFC` | `L` + frequency + channel adapters; exact Algorithm 1 dataflow |

Paper mode uses LoRA rank 8, scale 32, and query/key/value projections; frequency
top-k 3 with `h2 = h1`; and channel rank `h1 / 2`. Do not silently substitute the
older MVP adapters, q/v-only LoRA, alpha 16, a top-frequency fraction, or a
`d_model / 8` bottleneck.

All arms clone one seeded template, so modules shared by two arms have identical
initial weights and mini-batches. Their initial predictions are intentionally not
forced to match: Algorithm 1 passes channel embeddings directly to LayerNorm and
the forecast head, so a residual zero-impact bypass would change the accepted-paper
architecture. The paper does not publish initialization; the paper adapters use a
documented Xavier choice. Preserve the explicit arm-dataflow tests when changing it.

## Leakage and evaluation invariants

- Selection-time code accepts `SupportView`, never `EvaluationEpisode` or query
  tensors. Do not weaken that API boundary.
- Correlation evidence is computed from support residuals after exactly one
  eval/no-grad frozen forecast. Preserve model mode, device, and parameter
  `requires_grad` flags.
- Each support example remains a separate correlation window. Never concatenate
  adjacent windows and create false temporal transitions.
- Query labels may create offline utility labels only for **source** datasets.
  In every LODO fold, exclude the complete held-out dataset before constructing
  labels, fitting imputers/scalers, fitting gates, or tuning thresholds.
- Group by dataset for outer validation. A row-wise random split is leakage.
- Support and query raw timestamp intervals must remain disjoint, and
  normalization statistics must not use query timestamps.
- The source forecasting head is trained on Electricity, which is outside the
  13-dataset evaluation suite. Do not add Electricity to a target fold while
  reusing that head.
- Require a complete `L/LF/LC/LFC` action set and exact seed pairing for every
  evaluated episode. Incomplete action sets are not evidence for a routing claim.

## Cost-accounting invariants

- Router end-to-end time is evidence extraction + gate inference + selected-arm
  adaptation. The always-on `LFC` baseline has no selection cost.
- Adaptation time excludes data download, model download, episode materialization,
  source-head training, and offline router fitting. Report those separately when
  relevant.
- `profiled_flops` is a PyTorch-profiler estimate for one optimization step scaled
  by the fixed update count. FFT/top-k operations may be undercounted; accompany
  profiler values with the analytical formulas in `docs/COMPUTE_ANALYSIS.md`.
- Analytical savings are active trainable-parameter savings. They are not stored
  checkpoint-byte savings if all optional adapters remain serialized.
- Parameter savings do not imply the same wall-time percentage because the frozen
  backbone and forecasting head are shared by all arms.

## Code map

- `src/utility_peft/actions.py`: matched action definitions.
- `src/utility_peft/adapters/modules.py`: paper-specified frequency/channel paths.
- `src/utility_peft/correlation.py`: support-only residual-correlation evidence.
- `src/utility_peft/correlation_benchmark.py`: marginal labels, binary gates,
  LODO evaluation, and paper parameter formulas.
- `src/utility_peft/episodes.py`: chronological, raw-disjoint support/query data.
- `src/utility_peft/data/datasets.py` and `datasets/manifest.yaml`: pinned sources,
  aliases, hashes, splits, and synthetic generators.
- `src/utility_peft/evaluator.py`: fixed-update adaptation and measured costs.
- `configs/correlation_pilot.yaml` and `configs/correlation.yaml`: pilot and full
  experiment entry points.

## Local development

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/utility-peft --help
```

The lock file includes `pyedflib`, which is needed for ECGCA115/ECGCA515. Keep
downloads, model caches, episodes, checkpoints, and utility records outside Git
under `data/`, `.cache/`, and `artifacts/`.

Before a GPU run, first run the CPU tests covering paper adapters, correlation
evidence, the four-arm router, parameter formulas, dataset manifests, and episode
non-overlap. Then use the command sequence in `docs/CODEX_HANDOFF.md`.

## Safe continuation rules

- The worktree may contain user or agent changes. Inspect `git status` and never
  reset, discard, or overwrite unrelated modifications.
- Utility generation is intentionally resume-safe. Re-run the same resolved
  config instead of deleting partially completed artifacts.
- A changed protocol must produce a changed config hash and a new artifact root.
  Do not mix incompatible records in one result table.
- Add or update tests with every protocol change. At minimum verify finite
  evidence, query rejection, held-out exclusion, exact arm pairing, deterministic
  routing, parameter counts, and timing-field semantics.
- Never claim that a CPU synthetic smoke test establishes accuracy or research
  promise. It validates plumbing only.
