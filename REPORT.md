# Lab 2 Report (Template)

This file is a **ready-to-fill report template** that matches the numbered questions in
`lab2-ddl0306.pdf`. Run the pipeline to generate results/figures under `artifacts/`, then paste
key numbers/plots here and add your interpretations.

## 2 — Anisotropy and the geometry of word embeddings

### 1) Intuitive interpretation of the three metrics

**Mean pairwise cosine (A1).**\
Measures how similar *random* vectors are to each other on average. In an “isotropic” space,
random directions should be close to orthogonal, so the average cosine should be near 0. A
large positive A1 indicates the space is **anisotropic** (vectors share a common direction /
cone).

**Cosine to the mean direction (cos(v, μ)).**\
Computes the average vector μ and then measures how aligned each vector is with μ. If many
vectors point in a similar direction, μ has a large norm and cos(v, μ) concentrates at high
values. In a more isotropic space, μ is smaller and cos(v, μ) is closer to 0 (and more spread).

**PCA dominance.**\
If a few principal components explain a large fraction of the variance, the data vary mostly
along a small number of directions (again suggesting anisotropy / “common directions”). A more
uniformly spread (isotropic) space yields flatter EVR and slower cumulative growth.

### 2) Metrics for BERT (layers) vs GloVe

- **Number of distinct pairs** for A1 with `M` vectors is `M(M-1)/2`.
- Metrics and plots are written to:
  - `artifacts/results/anisotropy_metrics.json`
  - `artifacts/figures/anisotropy_cos2mean__*.png`
  - `artifacts/figures/anisotropy_pca_evr__*.png`

### 3) Metrics after de-anisotropisation

This repo recomputes all three metrics after:
- mean centering
- mean centering + L2 normalization
- removing the top principal components (m ∈ {1,2,5})

### 4) Interpretation

Paste the key numbers from `anisotropy_metrics.json` and explain:
- which models/layers are most anisotropic under each metric
- how much each de-anisotropisation method changes the geometry
- whether BERT layers differ in anisotropy (and in which direction)

## 3 — Morphological encoding

All results/plots are written to:
- `artifacts/results/morphology_results.json`
- `artifacts/figures/morphology/`

### 5) (i) Intra- vs inter-family distances

Summarise intra-family vs inter-family cosine similarity distributions, including:
- a plot (violin/box)
- effect size (Cohen’s d and Cliff’s δ are both computed)
- Mann–Whitney p-value
- comparison raw vs de-anisotropised

### 5) (ii) Offset consistency

This code computes offset consistency for simple suffix patterns found *inside each family*:
- `+ed` (incl. `+d` after trailing `e`)
- `+ing` (incl. dropping trailing `e`)
- plural `+s`

Report mean pairwise cosine between offset vectors and bootstrap confidence intervals.

### 5) (iii) Probing

This repo implements a lightweight probing task:
- label: **past vs present** using `+ed` pairs from the family list
- model: logistic regression
- evaluation: stratified 5-fold CV (grouped by base lemma when available)
- metrics: accuracy + macro-F1 with bootstrap CIs across folds

Compare raw vs de-anisotropised embeddings, and compare GloVe vs BERT layers.

## 4 — Synonyms vs antonyms

All results/plots are written to:
- `artifacts/results/synant_results.json`
- `artifacts/figures/synant/`

### 6) Why antonyms can be close to synonyms (expected yet counter-intuitive)

**Expected (distributional semantics).**\
Distributional embeddings are learned from **contexts**. Many antonym pairs (e.g. *hot/cold*,
*good/bad*) appear in very similar syntactic frames and topical contexts (“X is ___”, “too ___”,
comparatives, coordination, etc.). If “meaning” is approximated by context distributions, then
synonyms *and antonyms* can end up nearby because they are both highly substitutable in context.

**Counter-intuitive (lexical semantics).**\
Antonyms encode an explicit opposition relation; from a semantic viewpoint, we might prefer
representations where opposites are far apart. Distributional similarity is therefore not a
guarantee of semantic equivalence: it often captures *relatedness* rather than *sameness*.

### 7) Similarity separation

Compare cosine similarity distributions for:
- synonym pairs
- antonym pairs
- random baseline pairs

Do this for GloVe and BERT layers, and for raw vs de-anisotropised embeddings.

### 8) Neighbourhood evaluation

For 50 target words (sampled from those that have both a synonym and an antonym in the built
lists), compute top-10 nearest neighbours and estimate how often an antonym appears among the
top-10 neighbours when a synonym is present in the vocabulary.

Compare the rate before vs after de-anisotropisation and across models/layers.

