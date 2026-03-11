# Data Download and Preparation Guide

## Recommended Data Sources

### 1. Mozilla Common Voice (Recommended)

**Website**: https://commonvoice.mozilla.org/

**Features**:
- Free and open-source
- 100+ languages supported
- Includes transcripts
- Good audio quality

#### Download Steps

1. **Visit and register**
   ```
   https://commonvoice.mozilla.org/datasets
   ```

2. **Select language and version**
   - Choose your language (e.g., English, French)
   - Select latest version (e.g., Common Voice Corpus 17.0)

3. **Download dataset**
   - Click download button
   - Format: Recommended `.tar.gz` format
   - Size: English ~30GB (full), can download smaller validated subset

4. **Extract data**
   ```bash
   tar -xzf cv-corpus-17.0-2024-03-15-en.tar.gz
   ```

#### Data Structure

After extraction:
```
cv-corpus-17.0-2024-03-15/en/
├── clips/              # Audio files directory
│   ├── common_voice_en_00001.mp3
│   ├── common_voice_en_00002.mp3
│   └── ...
├── validated.tsv       # Validated data (recommended)
├── train.tsv          # Training set
├── dev.tsv            # Validation set
└── test.tsv           # Test set
```

#### Convert to Project Format

Use provided conversion script:

```bash
# Install ffmpeg (for audio conversion)
brew install ffmpeg  # macOS

# Convert English data (process first 100 samples for testing)
python scripts/convert_common_voice.py \
  --cv-dir ~/Downloads/cv-corpus-17.0-2024-03-15/en \
  --output-dir data/raw \
  --lang en \
  --max-samples 100

# Convert French data
python scripts/convert_common_voice.py \
  --cv-dir ~/Downloads/cv-corpus-17.0-2024-03-15/fr \
  --output-dir data/raw \
  --lang fr \
  --max-samples 100
```

After conversion:
```
data/raw/
├── en/
│   ├── wav/
│   │   ├── en_000000.wav
│   │   ├── en_000001.wav
│   │   └── ...
│   └── transcripts.txt
└── fr/
    ├── wav/
    │   ├── fr_000000.wav
    │   ├── fr_000001.wav
    │   └── ...
    └── transcripts.txt
```

---

### 2. LibriSpeech (English Only)

**Website**: https://www.openslr.org/12/

**Features**:
- English speech dataset
- High-quality recordings
- ~1000 hours of audio

#### Download Steps

```bash
# Download test set (smaller, ~350MB)
wget https://www.openslr.org/resources/12/test-clean.tar.gz

# Extract
tar -xzf test-clean.tar.gz
```

---

### 3. Prepare Your Own Data

If you have your own audio data, prepare it in the following format:

#### Directory Structure

```
data/raw/<lang>/
├── wav/
│   ├── file001.wav
│   ├── file002.wav
│   └── ...
└── transcripts.txt
```

#### Audio Requirements

- **Format**: WAV
- **Sampling Rate**: 16000 Hz
- **Channels**: Mono (single channel)
- **Bit Depth**: 16-bit (recommended)

#### Conversion Commands

If your audio doesn't meet requirements, use ffmpeg:

```bash
# Convert single file
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Batch conversion (in audio directory)
for file in *.mp3; do
  ffmpeg -i "$file" -ar 16000 -ac 1 "${file%.mp3}.wav"
done
```

#### transcripts.txt Format

```
file001	This is the first sentence.
file002	This is the second sentence.
file003	Hello world.
```

**Note**:
- Filename without extension (.wav)
- Use Tab character (not space) as separator
- One audio file per line
- Use UTF-8 encoding

---

## Quick Start (Complete Workflow)

### Option A: Using Common Voice (Recommended)

```bash
# 1. Download Common Voice dataset
# Visit https://commonvoice.mozilla.org/datasets
# Download English and French datasets

# 2. Extract data
tar -xzf cv-corpus-17.0-2024-03-15-en.tar.gz
tar -xzf cv-corpus-17.0-2024-03-15-fr.tar.gz

# 3. Install ffmpeg
brew install ffmpeg

# 4. Convert data (use small sample for testing)
python scripts/convert_common_voice.py \
  --cv-dir ~/Downloads/cv-corpus-17.0-2024-03-15/en \
  --output-dir data/raw \
  --lang en \
  --max-samples 50

# 5. Verify data
ls data/raw/en/wav/
cat data/raw/en/transcripts.txt | head

# 6. Run pipeline
dvc repro
```

### Option B: Using Sample Data (Quick Test)

If you just want to test the pipeline:

```bash
# Create directory
mkdir -p data/raw/en/wav

# Generate sample audio using espeak-ng
echo "This is a test sentence." | espeak-ng -w data/raw/en/wav/test001.wav
echo "Hello world." | espeak-ng -w data/raw/en/wav/test002.wav
echo "How are you today?" | espeak-ng -w data/raw/en/wav/test003.wav

# Create transcripts.txt
cat > data/raw/en/transcripts.txt << EOF
test001	This is a test sentence.
test002	Hello world.
test003	How are you today?
EOF

# Convert to 16kHz mono
for file in data/raw/en/wav/*.wav; do
  ffmpeg -i "$file" -ar 16000 -ac 1 -y "${file%.wav}_16k.wav"
  mv "${file%.wav}_16k.wav" "$file"
done

# Run pipeline
dvc repro
```

---

## Data Volume Recommendations

| Purpose | Recommended Samples | Estimated Processing Time |
|---------|---------------------|---------------------------|
| Quick test | 10-20 | 5-10 minutes |
| Development | 50-100 | 20-30 minutes |
| Full experiment | 500-1000 | 2-4 hours |
| Publication-level | 5000+ | Hours to days |

**Recommendation**: Start with small data (50-100 samples) to test the entire pipeline, then process full dataset after confirming everything works.

---

## Common Issues

### Q1: Common Voice download is slow?

**A**:
- Use download tools (wget, aria2c) with resume support
- Download smaller subset (validated.tsv only)
- Consider using mirrors if available

### Q2: Audio conversion fails?

**A**: Check if ffmpeg is installed correctly:
```bash
ffmpeg -version
```

### Q3: transcripts.txt format error?

**A**: Ensure:
- Use Tab character (not space) as separator
- Filename without extension
- Save with UTF-8 encoding

Verification:
```bash
# View separator (should see \t)
cat -A data/raw/en/transcripts.txt | head
```

### Q4: Insufficient disk space?

**A**:
- Process partial data (use --max-samples parameter)
- Delete original Common Voice data after conversion
- Use external storage

---

## Data Preparation Checklist

Complete these checks before running pipeline:

- [ ] Audio files in `data/raw/<lang>/wav/` directory
- [ ] All audio files are 16kHz, mono, WAV format
- [ ] `transcripts.txt` file exists
- [ ] `transcripts.txt` uses Tab separator
- [ ] Filenames match audio files (without extension)
- [ ] At least 10 samples for testing

Verification commands:
```bash
# Check audio files
ls data/raw/en/wav/*.wav | wc -l

# Check transcripts.txt
wc -l data/raw/en/transcripts.txt

# Check audio format
file data/raw/en/wav/*.wav | head
soxi data/raw/en/wav/*.wav | head
```

---

## Recommended Workflow

1. **Download small dataset for testing** (50-100 samples)
2. **Run complete pipeline for verification**
3. **Check output results** (metrics and plots)
4. **Process full dataset after confirmation**
5. **Add new languages** (only modify params.yaml)

This approach helps quickly identify issues and avoids wasting time processing large datasets only to discover configuration errors.
