# Lab 2 — Static vs Contextual Embeddings (BERT vs GloVe)

This repo contains a small, readable Python codebase to complete the tasks described in
`lab2-ddl0306.pdf` (anisotropy, morphology, synonyms/antonyms).

Per the lab instructions, **all downloads and caches are stored inside this repository** (see
`data/` and `artifacts/`).

## Setup

1) Create a Python environment (recommended: Python 3.10–3.12).

2) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3) Put the provided morphological families file at:

```text
data/morph_families.tsv
```

If you already placed it in the repo root as `morph_families.tsv`, the code will also pick that up.

4) Download a GloVe text file (recommended default: `glove.6B.100d.txt`) into:

```text
data/glove/glove.6B.100d.txt
```

## Run

All commands are driven by a single CLI:

```bash
python -m lab2 --help
```

Typical end-to-end flow:

```bash
# 1) Build corpora resources (WikiText, frequencies, contexts)
python -m lab2 prepare-corpus

# 2) Build WordNet synonym/antonym pairs
python -m lab2 build-synant

# 3) Compute embeddings (GloVe + BERT) with caching
python -m lab2 compute-embeddings

# 4) Run analyses + plots (anisotropy, morphology, synonyms/antonyms)
python -m lab2 run-analyses
```

One-shot run + markdown report (includes `pytest` output):

```bash
python -m lab2 generate-report --out artifacts/run_report.md
```

Outputs:
- `artifacts/results/` — JSON/CSV summaries
- `artifacts/figures/` — plots (PNG)
- `artifacts/embeddings/` — cached embeddings (NPY)
- `artifacts/contexts/` — sampled context sentences (JSONL)

## Notes

- BERT token-to-word alignment is handled via **offset mappings** from Hugging Face “fast”
  tokenizers, and **subword pieces are mean-pooled**, as required by the lab.
- For contextual embeddings, this code uses **WikiText-103 (train split)** and implements the
  exact sampling rules in the PDF (≤40 tokens, de-dup, sample 50 with seed 42, etc.).

- Offline fallback: if you cannot install `datasets`, you can place the WikiText-103 train split
  as plain text at `data/wikitext103_train.txt` and the pipeline will read from it.
