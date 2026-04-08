# Lab2 Run & Test Report

- Generated: `2026-04-08 00:07:48 CST`
- Repo: `/home/l1ght/repos/Cherry2603`
- Git HEAD: `cd4b31000b7d461503923cdc5dfa55ad9443a0a6`
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
6 passed in 0.12s
```

## Result Summary (raw)

### Anisotropy
- GloVe/raw: {'n_vectors': 467, 'n_pairs': 108811, 'mean_pairwise_cosine': 0.023680906742811203, 'cos_to_mean_mean': 0.15758831799030304}
- BERT/layer_1/raw: {'n_vectors': 467, 'n_pairs': 108811, 'mean_pairwise_cosine': 0.1256357729434967, 'cos_to_mean_mean': 0.3565421402454376}
- BERT/layer_6/raw: {'n_vectors': 467, 'n_pairs': 108811, 'mean_pairwise_cosine': 0.3337724804878235, 'cos_to_mean_mean': 0.5788667798042297}
- BERT/layer_12/raw: {'n_vectors': 467, 'n_pairs': 108811, 'mean_pairwise_cosine': 0.40671733021736145, 'cos_to_mean_mean': 0.6386417746543884}

### Morphology
- GloVe intra-vs-inter (raw): cliffs_delta={'estimate': 0.8975765306122448, 'low': 0.8610459183673469, 'high': 0.9276275510204082} p=1.9644318165920838e-75
- BERT/layer_1 intra-vs-inter (raw): cliffs_delta={'estimate': 0.9930867346938775, 'low': 0.9860204081632653, 'high': 0.9979591836734694} p=6.458306038043788e-92
- BERT/layer_6 intra-vs-inter (raw): cliffs_delta={'estimate': 0.9784438775510204, 'low': 0.9652551020408163, 'high': 0.9887755102040816} p=2.784215947620276e-89
- BERT/layer_12 intra-vs-inter (raw): cliffs_delta={'estimate': 0.9599744897959184, 'low': 0.9376020408163265, 'high': 0.9782908163265306} p=5.154994986879388e-86

### Synonyms/Antonyms
- GloVe neighbourhood (raw): {'n_targets': 50, 'rate_antonym_in_topk': 0.48}
- BERT/layer_1 neighbourhood (raw): {'n_targets': 50, 'rate_antonym_in_topk': 0.5}
- BERT/layer_6 neighbourhood (raw): {'n_targets': 50, 'rate_antonym_in_topk': 0.38}
- BERT/layer_12 neighbourhood (raw): {'n_targets': 50, 'rate_antonym_in_topk': 0.38}

## Artifacts
- Results dir: `/home/l1ght/repos/Cherry2603/artifacts/results`
- Figures dir: `/home/l1ght/repos/Cherry2603/artifacts/figures`
- Embeddings dir: `/home/l1ght/repos/Cherry2603/artifacts/embeddings`

