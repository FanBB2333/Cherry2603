# DVC Pipeline for Phoneme ASR Robustness to Noise

This project implements a fully reproducible DVC pipeline to evaluate the robustness of phoneme-based Automatic Speech Recognition (ASR) to noise at different Signal-to-Noise Ratio (SNR) levels.

## Project Structure

```
.
├── scripts/
│   ├── create_manifest.py    # Create manifests from raw audio
│   ├── add_noise.py          # Add noise to audio files
│   ├── inference.py          # Run phoneme recognition
│   ├── evaluate.py           # Compute PER metrics
│   └── plot_results.py       # Plot PER vs SNR curves
├── data/
│   ├── raw/                  # Raw audio files (not tracked)
│   │   ├── en/
│   │   │   ├── wav/
│   │   │   └── transcripts.txt
│   │   └── fr/
│   │       ├── wav/
│   │       └── transcripts.txt
│   ├── manifests/            # JSONL manifests (DVC tracked)
│   ├── noisy/                # Noisy audio files (DVC tracked)
│   └── predictions/          # Prediction manifests (DVC tracked)
├── metrics/                  # Evaluation metrics (DVC tracked)
├── plots/                    # Result plots (DVC tracked)
├── params.yaml               # Pipeline parameters
├── dvc.yaml                  # DVC pipeline definition
└── requirements.txt          # Python dependencies

```

## Prerequisites

1. **Python 3.9+**
2. **espeak-ng** - For text-to-phoneme conversion
   ```bash
   # macOS
   brew install espeak-ng

   # Ubuntu/Debian
   sudo apt-get install espeak-ng
   ```

3. **DVC** - Data Version Control
   ```bash
   pip install dvc
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Cherry2603
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize DVC:
   ```bash
   dvc init
   ```

## Data Preparation

You need to prepare your raw audio data in the following structure:

```
data/raw/<lang>/
├── wav/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── transcripts.txt
```

The `transcripts.txt` file should be tab-separated with format:
```
<filename_without_extension>\t<transcript_text>
```

Example:
```
file1	This is the first sentence.
file2	This is the second sentence.
```

## Running the Pipeline

### Single Language

1. Edit `params.yaml` to specify one language:
   ```yaml
   languages:
     - en
   ```

2. Run the complete pipeline:
   ```bash
   dvc repro
   ```

3. View metrics:
   ```bash
   dvc metrics show
   ```

4. View plots:
   ```bash
   open plots/per_vs_snr.png
   ```

### Multiple Languages

1. Edit `params.yaml` to add more languages:
   ```yaml
   languages:
     - en
     - fr
   ```

2. Re-run the pipeline (only new stages will execute):
   ```bash
   dvc repro
   ```

## Pipeline Stages

1. **create_manifest**: Convert raw audio + transcripts to JSONL manifests with phoneme references
2. **add_noise**: Generate noisy versions at different SNR levels
3. **inference_clean**: Run phoneme recognition on clean audio
4. **inference_noisy**: Run phoneme recognition on noisy audio
5. **evaluate_clean**: Compute PER on clean audio
6. **evaluate_noisy**: Compute PER on noisy audio
7. **plot_results**: Generate PER vs SNR curves

## Configuration

Edit `params.yaml` to customize:
- `languages`: List of language codes
- `snr_levels`: SNR levels in dB (e.g., [20, 15, 10, 5, 0, -5, -10])
- `model_name`: HuggingFace model identifier
- `seed`: Random seed for reproducibility

## Key Features

- **Atomic manifest creation**: Manifests are written atomically to ensure data integrity
- **Reproducibility**: Fixed random seeds ensure identical results across runs
- **Language-agnostic**: Add new languages by only changing parameters
- **DVC tracking**: All intermediate artifacts and metrics are tracked by DVC

## Troubleshooting

### espeak-ng not found
Make sure espeak-ng is installed and in your PATH.

### CUDA out of memory
The inference script will automatically use CPU if GPU is not available. For large datasets, process in batches.

### Audio format issues
Ensure all audio files are:
- Mono (single channel)
- 16kHz sampling rate
- WAV format

## License

This project is for educational purposes.
