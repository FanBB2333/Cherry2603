# Lab3 Executed Experiments

This file records the experiment suite that was actually run on April 9, 2026 with the local NVIDIA GeForce RTX 3090 Ti.

## Experiment Plan

1. Attention implementation benchmark on GPU.
2. Main baseline training run on `songci_main_5mb.txt`.
3. Extended main run to `20k` steps on the same configuration to inspect long-training behaviour.
4. Hyperparameter study with three controlled variations:
   - context length `128` vs baseline `256`
   - layers `4` vs baseline `6`
   - dropout `0.0` vs baseline `0.1`
5. Generated-poem structural comparison against the training corpus.
6. A probing experiment: can a linear probe predict whether the next character is a newline from the final hidden state?

## Attention Benchmark

Configuration:

- device: `cuda`
- batch size: `32`
- context length: `256`
- hidden dim: `256`
- heads: `8`

Results:

- `sdpa`: `2.166 ms` average step time, `114.02 MB` peak memory
- `classic`: `3.975 ms` average step time, `331.31 MB` peak memory

Interpretation:

- `scaled_dot_product_attention` is clearly faster.
- It also uses much less memory.
- This supports using `sdpa` as the main training backend.

## Main Baseline

Run directory: `lab3_runs/main_songci5mb_gpu_baseline_5k`

Configuration:

- corpus: `songci_main_5mb.txt`
- context length: `256`
- hidden dim: `256`
- layers: `6`
- heads: `8`
- dropout: `0.1`
- batch size: `32`
- learning rate: `3e-4`
- warmup steps: `500`
- max steps: `5000`

Key metrics:

- final train loss: `3.7247`
- final val loss: `4.0442`
- best val loss: `4.0411` at step `4500`
- final val BPC: `5.8345`

Qualitative result:

- By step `5000`, the model produces clearly poem-like lines with strong lexical recurrence and plausible ci-style imagery.
- The output is still repetitive, but it is no longer near-random or purely local.

## Extended Main Run

Run directory: `lab3_runs/main_songci5mb_gpu_baseline_20k`

Key metrics:

- final train loss: `2.8377`
- final val loss: `4.3550`
- best val loss: `4.0186` at step `6000`
- final val BPC: `6.2829`

Interpretation:

- The extended run is extremely useful because it reveals a clear overfitting pattern.
- Validation improves up to about `6000` steps, then gradually worsens while train loss keeps decreasing.
- This justifies using checkpoints and selecting the model by validation performance, not by final training step.

## Hyperparameter Study

### Baseline Reference

- baseline best val loss: `4.0411`

### Context Length = 128

Run directory: `lab3_runs/hparam_context128_gpu_2k`

- final / best val loss: `4.3677`
- final val BPC: `6.3013`

Interpretation:

- Shorter context hurts validation performance.
- This supports the report claim that `256` is more appropriate for capturing cross-line structure.

### Layers = 4

Run directory: `lab3_runs/hparam_layers4_gpu_2k`

- final / best val loss: `4.3967`
- final val BPC: `6.3432`

Interpretation:

- Reducing depth weakens the model relative to the 6-layer baseline.
- The smaller model still learns, but it converges to a worse validation loss.

### Dropout = 0.0

Run directory: `lab3_runs/hparam_dropout0_gpu_2k`

- final / best val loss: `4.2696`
- final val BPC: `6.1597`

Interpretation:

- At `2000` steps, removing dropout improved validation loss relative to the other short hyperparameter runs.
- This suggests that for this corpus and training horizon, `0.1` dropout may slightly slow early optimisation.
- This does not automatically mean `0.0` is best for longer training; it should be framed as an early-training trade-off.

## Generated Structure Analysis

Files:

- generated samples: `lab3_runs/main_songci5mb_gpu_baseline_5k/generated_samples.txt`
- comparison JSON: `lab3_runs/main_songci5mb_gpu_baseline_5k/structure_comparison.json`

Why the `5k` checkpoint was used:

- The `20k` run showed that validation was best much earlier than the final step.
- The saved `5k` checkpoint lies near the best-validation region and produced cleaner outputs than the overtrained final model.

Key comparison points:

- reference average raw line length: `10.7413`
- generated average raw line length: `10.5026`
- reference average stripped line length: `8.9089`
- generated average stripped line length: `8.5825`
- reference punctuation ratio: `0.1706`
- generated punctuation ratio: `0.1828`
- line-ending distribution L1 gap: `0.1065`

Interpretation:

- The model matches line length and punctuation density surprisingly well.
- The major mismatch is poem length:
  the generated samples average `15.4` lines, while the reference corpus averages `7.57`.
- In other words, local line structure is learned more successfully than global poem-length control.

## Probing Experiment

File: `lab3_runs/main_songci5mb_gpu_baseline_5k/probe_newline.json`

Task:

- train a linear probe on frozen final hidden states
- predict whether the next character is a newline

Results:

- trained-model accuracy: `0.9880`
- trained-model F1: `0.9343`
- random-init accuracy: `0.9517`
- random-init F1: `0.6882`
- F1 gain over random init: `+0.2461`

Interpretation:

- The trained model encodes line-boundary information much more cleanly than a random model with the same architecture.
- This is a valid and useful probing result for the lab's "emergence of structure" goal.

## Recommended Report Narrative

1. Use the attention benchmark to justify `sdpa`.
2. Present the `20k` run as the full main experiment and highlight that the best validation region appears around `6k`.
3. Use the `5k` checkpoint for qualitative analysis because it is near the best saved region and generates stronger samples than the overtrained final checkpoint.
4. Use the three hyperparameter runs as a controlled study of context, depth, and dropout.
5. Use the structure analysis to argue that line-level regularities emerge earlier and more reliably than poem-level length control.
6. Use the newline probe as direct evidence that structural information is present in internal representations.
