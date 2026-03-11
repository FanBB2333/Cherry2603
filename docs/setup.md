# Setup and Running Guide

## 1. Data Preparation

### Required Data Structure

| File Path | Description | Format Requirements |
|-----------|-------------|---------------------|
| `data/raw/en/wav/*.wav` | English audio files | Mono, 16kHz, WAV format |
| `data/raw/en/transcripts.txt` | English transcripts | Tab-separated: `filename\ttext` |
| `data/raw/fr/wav/*.wav` | French audio files (optional) | Mono, 16kHz, WAV format |
| `data/raw/fr/transcripts.txt` | French transcripts (optional) | Tab-separated: `filename\ttext` |

### transcripts.txt Format Example

```
commonvoice_000001	I can hardly believe it.
commonvoice_000002	Please call me tomorrow.
commonvoice_000003	The weather is nice today.
```

**Note**:
- Filename without extension (.wav)
- Use Tab character (not space) as separator
- One audio file per line

### Recommended Data Sources

1. **Common Voice** (https://commonvoice.mozilla.org/)
   - Multi-language open-source speech dataset
   - Includes transcripts
   - Requires format conversion

2. **LibriSpeech** (https://www.openslr.org/12/)
   - English speech dataset
   - High-quality recordings

## 2. Environment Setup

### Install System Dependencies

```bash
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt-get install espeak-ng

# Verify installation
espeak-ng --version
```

### Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install DVC
pip install dvc
```

### Initialize Git and DVC

```bash
# Initialize Git (if not already done)
git init

# Initialize DVC
dvc init

# Add remote storage (optional, for data sharing)
# dvc remote add -d myremote /path/to/remote/storage
# or use cloud storage:
# dvc remote add -d myremote s3://mybucket/path
```

## 3. Running the Pipeline

### Step 1: Prepare Data

Create data directory structure:

```bash
mkdir -p data/raw/en/wav
mkdir -p data/raw/fr/wav
```

Place your audio files in the appropriate directories and create transcripts.txt files.

### Step 2: Configure Parameters

Edit `params.yaml`:

```yaml
# Start with one language for testing
languages:
  - en

# Use fewer SNR levels for initial testing
snr_levels:
  - 10
  - 5
  - 0
  - -5
```

### Step 3: Run Pipeline

```bash
# Run complete pipeline
dvc repro

# View pipeline status
dvc dag

# View metrics
dvc metrics show

# View generated plots
open plots/per_vs_snr.png  # macOS
# or xdg-open plots/per_vs_snr.png  # Linux
```

### Step 4: Add New Languages

Edit `params.yaml` to add new languages:

```yaml
languages:
  - en
  - fr  # Add French
```

Run again:

```bash
# DVC automatically detects changes and runs only necessary stages
dvc repro

# View updated results
dvc metrics show
open plots/per_vs_snr.png
```

## 4. Pipeline Stages

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| create_manifest | Raw audio + text | clean.jsonl | Create clean audio manifest |
| add_noise | clean.jsonl | snr_X.jsonl + noisy audio | Add noise at different SNR levels |
| inference_clean | clean.jsonl | predictions/clean.jsonl | Run phoneme recognition on clean audio |
| inference_noisy | snr_X.jsonl | predictions/snr_X.jsonl | Run phoneme recognition on noisy audio |
| evaluate_clean | predictions/clean.jsonl | metrics/clean.json | Compute PER for clean audio |
| evaluate_noisy | predictions/snr_X.jsonl | metrics/snr_X.json | Compute PER for noisy audio |
| plot_results | All metrics | per_vs_snr.png | Generate PER vs SNR curves |

## 5. Troubleshooting

### espeak-ng not found

```bash
# Check if installed
which espeak-ng

# If not, reinstall
brew install espeak-ng  # macOS
```

### Out of memory

If dataset is large:
- Reduce number of audio files
- Use smaller batch size
- Add batch processing logic in inference.py

### Audio format issues

Ensure audio files are:
- Mono (single channel)
- 16kHz sampling rate
- WAV format

Conversion command (using ffmpeg):
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```

### DVC cache issues

```bash
# Clean DVC cache
dvc gc

# Force re-run a specific stage
dvc repro -f <stage_name>
```

## 6. Git Commit

```bash
# Add code and configuration files
git add scripts/ params.yaml dvc.yaml requirements.txt .gitignore README.md
git add .dvc/config .dvc/.gitignore

# Commit
git commit -m "Initial pipeline implementation"

# Push to GitHub
git remote add origin <your-github-repo-url>
git push -u origin main
```

## 7. Expected Output

After successful run, you should see:

1. **Manifests**: `data/manifests/<lang>/*.jsonl`
2. **Metrics**: `metrics/<lang>/*.json`
3. **Plot**: `plots/per_vs_snr.png` - showing PER vs SNR curves for each language and average

## 8. Performance Optimization

1. **Use GPU**: If GPU is available, inference will be much faster
2. **Parallel processing**: Modify scripts to support multiprocessing
3. **Model caching**: First run downloads model, subsequent runs use cache
4. **Reduce data**: Use fewer audio files for testing

## 9. Extension Suggestions

1. Add more languages (Spanish, German, etc.)
2. Try different noise types (not just white noise)
3. Test different ASR models
4. Add more evaluation metrics (WER, CER, etc.)
