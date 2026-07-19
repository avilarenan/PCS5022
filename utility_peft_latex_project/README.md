# Utility-PEFT LaTeX project

This is a self-contained ICLR-style research and implementation proposal for:

**Utility-PEFT: Learning the Counterfactual Value of Adaptation Modules for Time-Series Foundation Models**

The proposal is designed to be handed to Codex or another coding agent for later implementation in a GPU environment. It contains:

- the scientific novelty and positioning relative to Time-PEFT, TRACE, TS-PET, AT4TS, MixFT, AutoLoRA, and related work;
- a mathematical definition of counterfactual adaptation utility;
- a minimum viable adapter/action space;
- implementation architecture and recommended repository layout;
- exact staged experiments, datasets, baselines, metrics, ablations, and falsification criteria;
- NinaPro DB6, NinaPro DB3, and EMGBench validation plans;
- a glossary, notation table, pseudocode, and Codex handoff checklist.

## Compile

The package is self-contained and includes the ICLR-style file.

```bash
make
```

Equivalent manual commands:

```bash
pdflatex main.tex
bibtex8 main
pdflatex main.tex
pdflatex main.tex
```

Clean generated files:

```bash
make clean
```

## Important scope decision

The minimum viable implementation should **not** begin with dynamic layer, rank, optimizer, loss, and online TTT decisions simultaneously. First validate the central claim with a small action space:

1. frozen/no-op;
2. head-only;
3. head + fixed LoRA;
4. head + LoRA + frequency adapter;
5. head + LoRA + channel adapter;
6. head + LoRA + frequency + channel adapters;
7. head + FourierFT.

Only expand to rank/layer selection after the oracle-map and utility-ranking experiments show that target-specific selection is meaningful.

## Deliverables

- `main.tex`: proposal source
- `references.bib`: bibliography
- `iclr2026_conference.sty`: local style file
- `main.pdf`: rendered proposal
- `Makefile`: build commands
