# ASR Robustness Evaluation

A reproducible DVC pipeline for evaluating phoneme-level ASR system robustness under different noise conditions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data (see docs/data.md)
mkdir -p data/raw/en/wav
# Place audio files in data/raw/en/wav/
# Create data/raw/en/transcripts.txt

# Run pipeline
dvc repro

# View results
dvc metrics show
open plots/per_vs_snr.png
```

## Documentation

- [Setup and Running Guide](docs/setup.md)
- [Data Preparation Guide](docs/data.md)
- [Model Download Guide](docs/model.md)
- [Experiment Report](REPORT.md)

## Project Structure

```
.
├── scripts/          # Pipeline scripts
├── data/            # Data directory
├── metrics/         # Evaluation metrics
├── plots/           # Visualization results
├── docs/            # Detailed documentation
├── params.yaml      # Parameter configuration
└── dvc.yaml         # Pipeline definition
```

## Key Features

- Fully reproducible experimental workflow
- Multi-language support
- Automatic GPU/CPU switching
- DVC version control
