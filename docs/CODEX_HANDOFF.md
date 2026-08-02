# Codex handoff for local VS Code development

## Goal and current state

The repository now contains a self-contained experimental path for comparing:

- always-on, paper-specified Time-PEFT (`LFC`); and
- a source-trained residual-correlation router that selects `L`, `LF`, `LC`, or
  `LFC` for each held-out target episode.

The accepted paper is linked from `papers/README.md`, the paper adapter equations
have executable tests, the four matched arms are defined, support-only correlation
evidence and LODO gates are implemented, all 13 datasets have manifest
entries/loaders, and smoke, pilot, and full configs/scripts are present.

This checkout may be intentionally uncommitted. Start by inspecting it; do not
reset or discard changes:

```bash
git status --short
git diff --check
```

Then read, in order:

1. `AGENTS.md`
2. `docs/EXPERIMENT.md`
3. `docs/COMPUTE_ANALYSIS.md`
4. `BASELINE_DISCREPANCIES.md`
5. `papers/README.md`, then the canonical OpenReview paper linked there

The paper linked from `papers/README.md` and `datasets/manifest.yaml` are
inputs/provenance, not generated artifacts. Do not edit them to make a result
match expectations.

## VS Code setup

Open the repository root as the VS Code workspace and select `.venv/bin/python`
as the Python interpreter. Use Python 3.11; the package excludes 3.12.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
.venv/bin/pip check
```

The lock file includes `pyedflib`, which is required for ECGCA115/ECGCA515.
MOMENT is pinned to its official repository commit, the model checkpoint is pinned
by revision, and Transformers/PEFT/PyTorch versions are constrained.

Keep these locations out of Git:

- `.cache/huggingface/`: model cache;
- `data/`: pinned public datasets;
- `artifacts/`: episodes, source heads, immutable Parquet utilities, run metadata,
  and reports.

Before starting a long job, confirm the GPU environment:

```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Required checks before experiments

Run the complete suite, then the focused protocol checks if debugging:

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests
```

```bash
.venv/bin/pytest \
  tests/test_time_peft_paper.py \
  tests/test_correlation.py \
  tests/test_correlation_benchmark.py \
  tests/test_datasets.py \
  tests/test_episodes.py
```

The critical assertions are:

- paper adapter parameter formulas and tensor shapes;
- q/k/v LoRA rank/scale and exact trainable-module membership;
- exact Algorithm 1 `LFC` dataflow and explicit `LF`/`LC` ablation dataflows;
- seeded shared-module initialization without a residual zero-impact bypass;
- evidence extractor rejects query/evaluation objects;
- finite signed/absolute, lagged, nonstationarity, and effective-rank features;
- complete matched action sets and source-only LODO fitting;
- correct one-class gate fallback;
- exact active-parameter formulas;
- dataset hashes, aliases, channel counts, and chronological splits;
- no raw support/query timestamp overlap.

Do not proceed to a long GPU sweep with a failing test or a dirty `diff --check`.

## Three ready-to-run workflows

Run commands from the repository root after activating the virtual environment or
placing `.venv/bin` on `PATH`:

```bash
source .venv/bin/activate
```

### 1. CPU plumbing smoke

```bash
bash scripts/run_cpu_smoke.sh
```

Equivalent commands:

```bash
utility-peft prepare-data --config correlation_smoke
utility-peft run-correlation-benchmark --config correlation_smoke
```

This uses a tiny random backbone/head, three generated systems, horizon 8, two
episodes, one seed, two updates, and 24 matched records. It validates preparation,
all four arms, evidence, storage, LODO routing, and report generation. It cannot
support an accuracy or Time-PEFT claim.

Outputs:

- `artifacts/correlation-smoke/correlation/utilities/`
- `artifacts/correlation-smoke/correlation/reports/correlation_benchmark.json`
- `artifacts/correlation-smoke/correlation/reports/correlation_benchmark.md`

### 2. GPU pilot

```bash
bash scripts/run_correlation_pilot.sh
```

Equivalent staged commands:

```bash
utility-peft prepare-data --config correlation_pilot --download
utility-peft train-source-head --config correlation_pilot --download
utility-peft run-correlation-benchmark --config correlation_pilot
```

The pilot uses MOMENT-base, Lorenz, DoublePendulum, ECGCA515, ETTh1, and Weather;
horizon 96; three episodes; three seeds; four arms; and 100 fixed updates. It
generates 180 action records after training a horizon-96 source head on Electricity.

Outputs:

- source head and provenance under
  `artifacts/correlation-pilot/checkpoints/`;
- episodes under `artifacts/correlation-pilot/episodes/`;
- immutable records under
  `artifacts/correlation-pilot/correlation/utilities/`;
- final JSON/Markdown under
  `artifacts/correlation-pilot/correlation/reports/`.

Treat this as a feasibility and systems pilot. In each LODO fold the gate has only
the other four datasets as sources, so routing estimates are high variance.

### 3. Full 13-dataset suite

```bash
bash scripts/run_correlation_full.sh
```

Equivalent staged commands:

```bash
utility-peft prepare-data --config correlation --download
utility-peft train-source-head --config correlation --download
utility-peft run-correlation-benchmark --config correlation
```

This uses all 13 datasets, horizons 96/192/336, three episodes, three seeds, four
arms, and 100 updates: 1,404 matched records plus three Electricity source-head
training jobs. It is a substantial GPU run. Estimate storage and runtime from the
pilot before launching.

Outputs:

- `artifacts/correlation-full/correlation/utilities/`
- `artifacts/correlation-full/correlation/reports/correlation_benchmark.json`
- `artifacts/correlation-full/correlation/reports/correlation_benchmark.md`

The commands use `bash` explicitly so they also work when archive extraction does
not preserve executable bits.

## Dataset preparation

`utility-peft prepare-data --download` reads `datasets/manifest.yaml` and:

- generates the five deterministic chaotic substitutes locally;
- downloads pinned ETT, Weather, Exchange, ECG, and Electricity sources;
- verifies SHA-256 for manifest-pinned files;
- validates row/channel shape and EDF signal layout;
- applies fixed ETT boundaries, 70/10/20 standard ratios, or 60/20/20
  synthetic/medical ratios;
- creates deterministic episode tensors and inspectable JSON manifests.

For intentional alternative data, the Python loader accepts `local_path`,
`source_url`, `manifest_path`, and `verify_hash=False`. Such a run is no longer the
pinned protocol and needs a new artifact root plus explicit provenance. Never
disable hash verification merely to bypass a corrupt download.

The ECG loader selects exactly two Thorax and four Abdomen signals and requires
equal sample rates. The local chaotic generators are not exact paper trajectories;
retain that limitation in every report.

## Source-head prerequisite

The public MOMENT-base checkpoint is reconstruction-pretrained; its long-horizon
forecast head is not a meaningful frozen predictor. Correlation evidence based on
a random head would be uninterpretable. The GPU configs therefore train one
horizon-specific forecast head on Electricity and bind it to:

- source file hash and split;
- source scaling/preprocessing;
- horizon and checkpoint-file digest;
- the complete evaluation-dataset exclusion set.

The utility-record model revision separately combines the configured backbone
revision with the source-head digest. The source-head sidecar does not itself prove
that a same-shape head was trained from that backbone revision; preserve the pinned
config and regenerate the head after changing the backbone.

The benchmark refuses provenance-incompatible checkpoints. Electricity is not one
of the 13 targets. If a different source dataset is introduced, it must remain
outside every target fold and the exclusion metadata must be regenerated.

## Resume and analysis-only behavior

Utility records are one immutable Parquet file per
dataset/horizon/episode/action/seed/config/model/preprocessing key. An interrupted
benchmark can be resumed by rerunning only its final command with the same config
and unchanged source-head checkpoint:

```bash
utility-peft run-correlation-benchmark --config correlation_pilot
```

or:

```bash
utility-peft run-correlation-benchmark --config correlation
```

Existing successful keys are skipped. Do **not** retrain or overwrite the source
head while resuming; its digest is part of the model revision and would define a
different experiment.

After every expected successful record exists, rebuild LODO analysis and reports
without running adaptations:

```bash
utility-peft run-correlation-benchmark --config correlation_pilot --analyze-only
```

`--analyze-only` deliberately fails on an incomplete table. The normal command
also requires the exact successful record count before reporting.

Current caveat: an action that exhausts its clean retry and is persisted with
`status=failed` is explicit but its key may block a same-config retry. Inspect the
record/error first. The safe current recovery is to fix the cause and use a new
artifact root/config hash; a future improvement should make failed-key retry an
explicit CLI operation rather than encouraging manual Parquet deletion.

Changing an update count, action definition, head, dataset, or preprocessing
requires a new config hash and preferably a new `paths.artifacts` root. Never merge
records from different protocols just to satisfy the expected count.

## Reading the outputs

The Markdown report is the quick view. The JSON report is authoritative and
contains:

- overall and fold-level router versus `LFC` MSE/MAE;
- evidence, routing, adaptation, and combined wall time;
- trainable/stored parameters, peak memory, and profiler FLOPs;
- route counts and gate class counts;
- implementation/parity guard fields;
- analytical paper-formula parameter savings by dataset.

Before interpreting a run, check:

1. every fold excluded its held-out dataset from `training_datasets`;
2. both gates had enough positive and negative source labels, or the one-class
   fallback is clearly visible;
3. route counts did not collapse entirely to `LFC`;
4. all arms have the same episode/seed count and update budget;
5. MSE and time are inspected per dataset/horizon, not only globally;
6. router time includes evidence and gate inference, while `LFC` has no selection
   charge;
7. measured savings agree in direction with the active-parameter formulas;
8. no claim is based on the CPU smoke label.

## High-priority continuation tasks

The bundle is ready for a small matched experiment, but these items should be
completed before a publication-strength claim:

1. **Run and archive the GPU pilot.** Verify downloads, source-head provenance,
   MOMENT q/k/v injection, memory headroom, and fixed-update timing on the actual
   GPU environment.
2. **Add paired uncertainty.** The configured margin is wired into the unit-level
   point diagnostic, but the current report has no confidence interval. Implement a
   hierarchical paired bootstrap over datasets and units, with seeds averaged inside
   units, and only then add bootstrap controls to the config and intervals to
   JSON/Markdown.
3. **Add source-fixed controls.** Report the best smaller arm among `L`, `LF`, and
   `LC`, selected using source folds only, plus a route-histogram-matched random
   mask. The four exhaustive arms are already available, so this is analysis work
   rather than new GPU adaptation.
4. **Harden timing.** The current arm loop order is fixed. Add warm-up and balanced
   or randomized arm order, record cold-start separately, and verify FFT/top-k
   undercounting in profiler FLOPs.
5. **Test support-size and feature ablations.** At minimum compare raw input
   correlation, frozen-residual correlation, removal of nonstationarity/effective
   rank, and `max_lag` values. Tune only inside source folds.
6. **Run the full suite only after the pilot passes.** Preserve the pilot as a
   systems checkpoint; do not repeatedly tune on full-suite held-out outcomes and
   then call the same outcomes confirmatory.
7. **Revisit parity if an authoritative implementation is available.** Numerically
   compare adapter placement, initialization, preprocessing, head, learning-rate
   selection, and splits. Until then retain the reimplementation label.

## Suggested first prompt for local Codex

Use a prompt like:

> Read AGENTS.md and all files in docs/. Inspect git status without discarding
> changes. Run the focused tests and CPU correlation smoke. Diagnose any failure
> before editing. Then summarize the exact GPU pilot commands, expected record
> count, current parity limitations, and any blocker you find. Do not launch the
> full suite yet.

That gives local Codex enough context to continue safely while keeping expensive
experiments and scientific claims under deliberate control.
