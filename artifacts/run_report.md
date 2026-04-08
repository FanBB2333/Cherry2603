# Lab2 Run & Test Report

- Generated: `2026-04-08 10:10:40 CST`
- Repo: `/home/l1ght/repos/Cherry2603`
- Git HEAD: `c2db7a8aa349b5267c9c3d86547bd31cf548cfcb`
- Python: `3.12.12`
- Platform: `Linux-6.8.0-106-generic-x86_64-with-glibc2.35`

## Pipeline
- (skipped)

## Checks
| item | ok | note |
|---|---:|---|
| `wikitext_word_freq.pkl` | OK |  |
| `morph_contexts.jsonl` | OK |  |
| `synonym_pairs.tsv` | OK |  |
| `antonym_pairs.tsv` | OK |  |
| `glove/index.json` | OK |  |
| `bert/layer_01/index.json` | OK |  |
| `bert/layer_06/index.json` | OK |  |
| `bert/layer_12/index.json` | OK |  |
| `anisotropy_metrics.json` | OK |  |
| `morphology_results.json` | OK |  |
| `synant_results.json` | OK |  |

## Pytest
- Exit code: `0`

```
......                                                                   [100%]
6 passed in 0.07s
```

## Result Summary (raw)

### Anisotropy
- GloVe/raw: {'n_vectors': 467, 'n_pairs': 108811, 'mean_pairwise_cosine': 0.023680906742811203, 'cos_to_mean_mean': 0.15758831799030304}
- BERT/layer_1/raw: {'n_vectors': 467, 'n_pairs': 108811, 'mean_pairwise_cosine': 0.22631590068340302, 'cos_to_mean_mean': 0.4771554172039032}
- BERT/layer_6/raw: {'n_vectors': 467, 'n_pairs': 108811, 'mean_pairwise_cosine': 0.15708404779434204, 'cos_to_mean_mean': 0.3984622061252594}
- BERT/layer_12/raw: {'n_vectors': 467, 'n_pairs': 108811, 'mean_pairwise_cosine': 0.5300953984260559, 'cos_to_mean_mean': 0.7287397980690002}

### Morphology
- GloVe intra-vs-inter (raw): cliffs_delta={'estimate': 0.8975765306122448, 'low': 0.8610459183673469, 'high': 0.9276275510204082} p=1.9644318165920838e-75
- BERT/layer_1 intra-vs-inter (raw): cliffs_delta={'estimate': 0.9889795918367347, 'low': 0.975, 'high': 0.9982908163265306} p=3.5728706962180567e-91
- BERT/layer_6 intra-vs-inter (raw): cliffs_delta={'estimate': 0.9966071428571428, 'low': 0.9921173469387755, 'high': 0.9996173469387755} p=1.4822093927627301e-92
- BERT/layer_12 intra-vs-inter (raw): cliffs_delta={'estimate': 0.9824234693877552, 'low': 0.9715816326530612, 'high': 0.9910969387755102} p=5.401786510160983e-90

### Synonyms/Antonyms
- GloVe neighbourhood (raw): {'n_targets': 50, 'rate_antonym_in_topk': 0.48}
- BERT/layer_1 neighbourhood (raw): {'n_targets': 50, 'rate_antonym_in_topk': 0.46}
- BERT/layer_6 neighbourhood (raw): {'n_targets': 50, 'rate_antonym_in_topk': 0.54}
- BERT/layer_12 neighbourhood (raw): {'n_targets': 50, 'rate_antonym_in_topk': 0.52}

## Artifacts
- Results dir: `/home/l1ght/repos/Cherry2603/artifacts/results`
- Figures dir: `/home/l1ght/repos/Cherry2603/artifacts/figures`
- Embeddings dir: `/home/l1ght/repos/Cherry2603/artifacts/embeddings`

