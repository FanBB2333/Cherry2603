# Repository Structure

This repository is a lab project for word embeddings and BERT analysis.

## Main Structure

- [README.md](README.md): short project guide.
- [lab2-ddl0306.pdf](lab2-ddl0306.pdf): the lab instruction file.
- [REPORT.md](REPORT.md): the final experiment report. It is in `Markdown` format.
- [lab2](lab2): the main Python package.
  - `lab2/data`: load WikiText, morphology pairs, WordNet pairs, and contexts.
  - `lab2/embeddings`: build and save GloVe and BERT embeddings.
  - `lab2/analysis`: anisotropy, morphology, and synonym/antonym analysis.
  - `lab2/pipeline.py`: the main pipeline.
  - `lab2/reporting.py`: build the run report.
- [src](src): older step-by-step scripts for each experiment stage.
- [tests](tests): unit tests.
- [artifacts](artifacts): generated outputs.

## Output Files

- [artifacts/run_report.md](artifacts/run_report.md): auto-generated run summary, also in `Markdown`.
- [artifacts/figures](artifacts/figures): all experiment images. The image format is `PNG` (`.png`).
- [artifacts/results](artifacts/results): numeric results and metadata.
  - `JSON` files: metrics and summaries, like `anisotropy_metrics.json`.
  - `TSV` files: synonym and antonym word pairs.
  - `PKL` file: cached word frequency data.
- [artifacts/contexts](artifacts/contexts): sampled sentences in `JSONL` format.
- [artifacts/embeddings](artifacts/embeddings): saved vectors.
  - vectors are stored as `NumPy` files (`.npy`)
  - indexes and metadata are stored as `JSON`

## In Simple Words

- `lab2` = main code
- `tests` = checks
- `artifacts` = outputs
- `REPORT.md` = final written answer
- `artifacts/figures/*.png` = plots and images
