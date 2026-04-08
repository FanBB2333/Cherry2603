# Lab 2 Report

This report summarizes the results obtained from the pipeline in this repository for
`lab2-ddl0306.pdf`.

Experimental setup:
- Static baseline: `GloVe 6B 100d`
- Contextual model: local `bert-base-uncased`
- BERT layers analysed: 1, 6, 12
- Corpus: local `WikiText-103 train`
- Frequency threshold for WordNet filtering: `N = 10`
- De-anisotropisation method used in sections 3 and 4: removal of the top 2 principal components

Generated artifacts:
- Results: `artifacts/results/`
- Figures: `artifacts/figures/`
- Run summary: `artifacts/run_report.md`

## 2 — Anisotropy and the geometry of word embeddings

### 1) Intuitive interpretation of the three metrics

Mean pairwise cosine `A1` measures the average similarity between random vectors. If the space is
close to isotropic, random vectors should be close to orthogonal and `A1` should be near zero.
Large positive values indicate that many vectors share common directions.

Cosine to the mean direction `cos(v, μ)` measures how strongly vectors align with the global mean
direction. If many vectors point into the same cone, the mean vector becomes informative and
`cos(v, μ)` becomes large.

PCA dominance measures how much variance is captured by the first principal components. If a few
PCs explain a large share of the variance, the space is concentrated around a small number of
dominant directions.

### 2) Metrics for BERT vs GloVe

After intersecting GloVe, BERT and the random-word anisotropy sample, the analysis used `M = 467`
vectors. Therefore the number of distinct pairs used for `A1` is:

`M(M-1)/2 = 467*466/2 = 108811`

Raw results:

| Model | A1 | mean cos(v, μ) | PC1 EVR |
|---|---:|---:|---:|
| GloVe | 0.0237 | 0.1576 | 0.0679 |
| BERT layer 1 | 0.1256 | 0.3565 | 0.0253 |
| BERT layer 6 | 0.3338 | 0.5789 | 0.0494 |
| BERT layer 12 | 0.4067 | 0.6386 | 0.0500 |

The trend is clear: BERT is much more anisotropic than GloVe, and anisotropy increases with
depth. Layer 12 is the most anisotropic under all three criteria used here.

### 3) Metrics after de-anisotropisation

All methods reduce anisotropy strongly, but mean centering already removes most of it.

Selected comparisons:

| Model | Raw A1 | Centered A1 | Remove PC2 A1 |
|---|---:|---:|---:|
| GloVe | 0.0237 | -0.0009 | -0.0019 |
| BERT layer 1 | 0.1256 | -0.0015 | -0.0021 |
| BERT layer 6 | 0.3338 | -0.0018 | -0.0021 |
| BERT layer 12 | 0.4067 | -0.0017 | -0.0019 |

For `cos(v, μ)`, the same pattern appears: for example, BERT layer 12 drops from `0.6386` in the
raw space to `0.0001` after removing the top 2 PCs. The first PCA component also becomes much less
dominant after PC removal; for BERT layer 12 it falls from `0.0500` to `0.0290`.

### 4) Interpretation

Three conclusions emerge.

First, the raw BERT space is strongly anisotropic, much more than GloVe, and this effect becomes
stronger in higher layers. Second, simple centering is already enough to push pairwise cosine and
mean-direction cosine close to zero. Third, PC removal flattens the leading variance directions
more aggressively than centering alone, which makes it a reasonable choice for the later
experiments.

## 3 — Morphological encoding

All detailed outputs are in `artifacts/results/morphology_results.json` and
`artifacts/figures/morphology/`.

### 5) (i) Intra- vs inter-family distances

Raw intra-vs-inter effect sizes:

| Model | Cliff's δ | Cohen's d | Mann–Whitney p |
|---|---:|---:|---:|
| GloVe | 0.8976 | 2.3662 | 1.96e-75 |
| BERT layer 1 | 0.9931 | 4.4731 | 6.46e-92 |
| BERT layer 6 | 0.9784 | 3.5912 | 2.78e-89 |
| BERT layer 12 | 0.9600 | 3.2079 | 5.15e-86 |

The difference between intra-family and inter-family similarities is therefore extremely strong and
highly significant for all models. Morphologically related forms are consistently closer to each
other than words drawn from different families.

After removing the top 2 PCs, separation becomes even stronger:

| Model | Raw Cliff's δ | De-anisotropised Cliff's δ |
|---|---:|---:|
| GloVe | 0.8976 | 0.9622 |
| BERT layer 1 | 0.9931 | 0.9944 |
| BERT layer 6 | 0.9784 | 0.9976 |
| BERT layer 12 | 0.9600 | 0.9855 |

This suggests that anisotropy partly hides morphological structure rather than creating it.

### 5) (ii) Offset consistency

Mean cosine between offset vectors:

| Model | +ed raw / deaniso | +ing raw / deaniso | +s raw / deaniso |
|---|---:|---:|---:|
| GloVe | 0.3345 / 0.2899 | 0.3440 / 0.3430 | 0.3172 / 0.2765 |
| BERT layer 1 | 0.2239 / 0.2248 | 0.2195 / 0.2170 | 0.2543 / 0.2522 |
| BERT layer 6 | 0.3020 / 0.3009 | 0.3604 / 0.3628 | 0.2981 / 0.2957 |
| BERT layer 12 | 0.4433 / 0.4472 | 0.4362 / 0.4424 | 0.4588 / 0.4619 |

Offset consistency is weakest in BERT layer 1, stronger in layer 6, and strongest in layer 12.
For BERT, deeper layers therefore appear to encode more stable morphological transformations.
De-anisotropisation changes these offset statistics only slightly.

### 5) (iii) Probing

The probing task is past vs present tense with grouped 5-fold cross-validation over `+ed` pairs.

Raw results:
- GloVe: accuracy `1.0000`, macro-F1 `1.0000`
- BERT layer 1: accuracy `1.0000`, macro-F1 `1.0000`
- BERT layer 6: accuracy `1.0000`, macro-F1 `1.0000`
- BERT layer 12: accuracy `1.0000`, macro-F1 `1.0000`

After de-anisotropisation:
- GloVe drops slightly to accuracy `0.9417`, macro-F1 `0.9403`
- All three BERT layers remain at `1.0000`

This probing task is therefore very easy for all representations, especially for BERT. The result
is still informative: removing dominant PCs does not destroy the morphological signal in BERT.

## 4 — Synonyms vs antonyms

All detailed outputs are in `artifacts/results/synant_results.json`,
`artifacts/results/synant_sampling_meta.json` and `artifacts/figures/synant/`.

The revised WordNet sampling produced 200 synonym pairs, 200 antonym pairs, and 57 words that
appear in both relation sets. This makes the neighbourhood evaluation possible on the requested 50
targets.

### 6) Why antonyms can be close to synonyms

This is expected under distributional semantics because antonyms often occur in highly similar
contexts. Words such as `hot/cold` or `good/bad` fit the same syntactic frames and topical
environments, so context-based embeddings naturally place them near each other.

At the same time, it is counter-intuitive semantically because antonymy is a relation of
opposition, not similarity. Distributional spaces therefore capture contextual relatedness better
than strict semantic equivalence.

### 7) Similarity separation

Raw cosine means:

| Model | Synonyms | Antonyms | Random |
|---|---:|---:|---:|
| GloVe | 0.3544 | 0.4936 | 0.0538 |
| BERT layer 1 | 0.2341 | 0.3225 | 0.1053 |
| BERT layer 6 | 0.4953 | 0.5553 | 0.3286 |
| BERT layer 12 | 0.6203 | 0.6705 | 0.4416 |

The central result is that antonyms are not only close to synonyms; in this experiment they are
often even closer than synonyms. This is especially strong in deeper BERT layers, where even
random pairs are already fairly close because of anisotropy.

After de-anisotropisation, the random baseline collapses toward zero:

| Model | Synonyms | Antonyms | Random |
|---|---:|---:|---:|
| GloVe | 0.2763 | 0.4047 | -0.0047 |
| BERT layer 1 | 0.1395 | 0.2389 | 0.0023 |
| BERT layer 6 | 0.2380 | 0.3191 | -0.0052 |
| BERT layer 12 | 0.2760 | 0.3567 | -0.0013 |

This makes the interpretation cleaner: de-anisotropisation removes the global similarity bias, but
the antonym-vs-synonym ordering remains. The closeness of antonyms is therefore not only an
artifact of anisotropy.

### 8) Neighbourhood evaluation

Antonym-in-top10 rate on 50 targets:

| Model | Raw | De-anisotropised |
|---|---:|---:|
| GloVe | 0.48 | 0.54 |
| BERT layer 1 | 0.50 | 0.48 |
| BERT layer 6 | 0.38 | 0.48 |
| BERT layer 12 | 0.38 | 0.48 |

Antonyms therefore appear very frequently in the local neighbourhood of a word, even when the
space is de-anisotropised. The strongest change is in deeper BERT layers, where neighbourhood
antonym rates increase from `0.38` to `0.48` after PC removal.

## Overall conclusion

The experiment supports four main conclusions.

1. BERT representations are much more anisotropic than GloVe, and anisotropy increases with
depth.
2. Morphological information is very clearly encoded in both static and contextual embeddings.
3. De-anisotropisation generally helps reveal structure rather than destroy it, especially for
intra-vs-inter family comparisons.
4. Synonyms and antonyms remain close in embedding space, which is entirely compatible with
distributional semantics but remains semantically counter-intuitive.

One practical caveat is that this run used the locally available `bert-base-uncased` model rather
than a 1024-dimensional BERT variant. The qualitative trends are nevertheless strong and coherent
across all three parts of the lab.
