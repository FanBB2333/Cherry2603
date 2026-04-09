# Lab3 Status Checklist

This checklist reflects the repository state after adding the new formal training scaffold.

## Done

- Assignment scope is understood: decoder-only Transformer, training, monitoring, and analysis.
- Poetry corpus extraction and export are already implemented.
- Corpus subsets by size already exist, including pilot and main-sized files.
- Character-level vocabulary building, encoding, and train/validation split are implemented.
- Random mini-batch sampling and tensor batching are implemented.
- Prototype baselines exist for embedding-only, single-head attention, and single decoder block models.
- Training metrics already include loss, validation loss, BPC, CSV logging, and plotting scripts.
- A formal reusable multi-head decoder implementation now exists in `lab3_transformer.py`.
- A formal training entrypoint with periodic generation and checkpoints now exists in `lab3_train.py`.
- An attention backend comparison script now exists in `lab3_attention_benchmark.py`.

## Partially Done

- Final architecture alignment:
  multi-head and multi-layer support are now implemented, but the real full baseline run still needs to be executed.
- Monitoring for the final lab:
  periodic generation and checkpoints are now implemented, but they have only been smoke-tested, not used in a long training run yet.
- Hyperparameter study readiness:
  the code now supports structured experiments, but the actual controlled runs and analysis are still missing.
- Repository readiness:
  the key scripts are now present, but final cleanup and report-facing organisation are still needed.

## Not Started

- Full main training run at the intended scale, for example 20k or more steps on the 5 MB corpus.
- Systematic hyperparameter exploration with interpretation.
- Structural analysis of generated poems against corpus statistics.
- Probing or internal representation analysis.
- Final report writing and final submission packaging.

## Safest next execution order

1. Run the first formal baseline with `lab3_train.py` on `songci_main_5mb.txt`.
2. Run a small number of controlled hyperparameter variations.
3. Collect generated samples and compare them with corpus-level structural statistics.
4. Add one probing or representation-analysis component.
5. Write the final report around these results.
