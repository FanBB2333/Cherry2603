# HuggingFace Model Download Guide

## Required Models

| Model ID | Purpose | Size | Files |
|----------|---------|------|-------|
| `facebook/wav2vec2-lv-60-espeak-cv-ft` | Multi-language phoneme recognition (English, French, etc.) | ~1.2GB | Multiple files |

## Download Methods (Choose One)

### Method 1: Automatic Download (Simplest)

The model will be automatically downloaded when running the pipeline:

```bash
# Run directly, first time will auto-download
dvc repro
```

**Pros**: No extra steps required
**Cons**: Need to wait for download on first run

---

### Method 2: Using huggingface-cli (Recommended for Pre-download)

```bash
# 1. Install huggingface-cli
pip install -U "huggingface_hub[cli]"

# 2. Download model to default cache directory (recommended)
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# Or download to specific directory
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft \
  --local-dir ./models/wav2vec2-lv-60-espeak-cv-ft \
  --local-dir-use-symlinks False
```

**Pros**: Can pre-download, supports resume
**Cons**: Requires additional tool installation

---

### Method 3: Using Provided Python Script

```bash
# Run download script
python scripts/download_models.py
```

**Pros**: Simple and direct, shows download progress
**Cons**: Requires Python environment configured

---

## Accelerated Download (For Users in China)

### Using HuggingFace Mirror

```bash
# Method A: Set environment variable (temporary)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# Method B: Permanent setting (add to ~/.zshrc or ~/.bash_profile)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
source ~/.zshrc
```

### Using Proxy

```bash
# Set proxy
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890

# Then download
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft
```

---

## Model Storage Location

### Default Cache Location

```
~/.cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/
```

### View Downloaded Models

```bash
# View cache directory
ls -lh ~/.cache/huggingface/hub/

# View model size
du -sh ~/.cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/
```

---

## Verify Download Success

### Method 1: Using Python Verification

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# If already downloaded, will load directly; otherwise will start download
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

print("Model loaded successfully!")
```

### Method 2: Check File Existence

```bash
# Check model directory
ls ~/.cache/huggingface/hub/ | grep wav2vec2-lv-60-espeak-cv-ft
```

---

## Complete Download Workflow (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install huggingface-cli
pip install -U "huggingface_hub[cli]"

# 3. (Optional) Set mirror acceleration
export HF_ENDPOINT=https://hf-mirror.com

# 4. Download model
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# 5. Verify download
python scripts/download_models.py
```

---

## Common Issues

### Q1: Download speed is very slow?

**A**: Use mirror or proxy:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: Download interrupted?

**A**: `huggingface-cli` supports resume, just re-run the command to continue.

### Q3: How to delete downloaded models?

**A**: Delete cache directory:
```bash
rm -rf ~/.cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/
```

### Q4: Model takes too much space?

**A**: This model is required, ~1.2GB. If space is insufficient, clean up other unnecessary models:
```bash
# View all cached models
ls ~/.cache/huggingface/hub/

# Delete unnecessary models
rm -rf ~/.cache/huggingface/hub/models--<model-name>/
```

### Q5: Can I use other models?

**A**: Yes, but need to modify `model_name` parameter in `params.yaml`. Ensure new model:
- Supports phoneme output (not text)
- Supports your required languages
- Input format is 16kHz audio

---

## Quick Command Summary

```bash
# Simplest: Auto-download (when running pipeline)
dvc repro

# Recommended: Pre-download
pip install -U "huggingface_hub[cli]"
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# Accelerated (China)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# Using script
python scripts/download_models.py
```

---

## Download Time Estimates

| Network Speed | Estimated Download Time |
|---------------|-------------------------|
| 10 MB/s | ~2 minutes |
| 5 MB/s  | ~4 minutes |
| 1 MB/s  | ~20 minutes |
| 500 KB/s | ~40 minutes |

**Recommendation**: If network is slow, use mirror or pre-download when network conditions are good.
