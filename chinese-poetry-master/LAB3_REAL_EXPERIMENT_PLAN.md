# Lab3 Real Experiment Plan

This plan turns the current prototype phase into a first formal lab3 run that matches the PDF much more closely.

## Why this plan

- The current repository already has working corpus export, character-level batching, loss computation, validation, CSV logging, and single-block decoder prototypes.
- The main missing pieces are multi-head attention, multi-layer depth, periodic generation, checkpoints, and a repeatable training entrypoint.
- The new `lab3_train.py` and `lab3_attention_benchmark.py` scripts are meant to cover exactly that gap without deleting the earlier prototype scripts.

## First formal run

- Corpus file: `songci_main_5mb.txt`
- Context length: `256`
- Hidden dimension: `256`
- Number of layers: `6`
- Number of heads: `8`
- MLP expansion: `4`
- Dropout: `0.1`
- Batch size: `32`
- Learning rate: `3e-4`
- Warmup steps: `500`
- Training steps: `20000`
- Validation interval: `200`
- Generation interval: `1000`
- Checkpoint interval: `2000`
- Attention backend: `sdpa`

## Why these choices are reasonable

- `songci_main_5mb.txt` matches the PDF's recommendation for a main experiment corpus size.
- Context length `256` is the exact value suggested in the PDF and is long enough to include several poetic lines instead of only one short line.
- Hidden dimension `256`, `6` layers, and `8` heads match the PDF's suggested baseline closely enough to count as a real experiment rather than another toy run.
- Dropout `0.1`, AdamW, and warmup are standard stabilisation choices for a small GPT-like decoder.
- Validation every `200` steps is frequent enough to see trends without making training mostly evaluation.
- Generation every `1000` steps matches the PDF's recommendation to monitor how output quality evolves.
- Checkpoints every `2000` steps keep recovery and later analysis practical.

## Immediate next experiments after the baseline

- Attention comparison: run `lab3_attention_benchmark.py` with `sdpa` and `classic`.
- Hyperparameter study 1: vary context length, for example `128` vs `256`.
- Hyperparameter study 2: vary depth, for example `4` vs `6` layers.
- Hyperparameter study 3: vary dropout, for example `0.0` vs `0.1`.

## What is still not completed even after these code changes

- Long full training runs still need to be executed.
- Generated poem structural analysis still needs a dedicated analysis script or notebook.
- Probing and internal representation analysis still need to be designed and implemented.
- Final report and repository cleanup are still separate tasks.
