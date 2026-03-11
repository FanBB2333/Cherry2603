# 数据下载和准备指南

## 📦 推荐数据源

### 1. Mozilla Common Voice（推荐）⭐️

**官网**：https://commonvoice.mozilla.org/

**特点**：
- 免费开源
- 支持100+语言
- 已包含转录文本
- 音频质量好

#### 下载步骤

1. **访问网站并注册**
   ```
   https://commonvoice.mozilla.org/datasets
   ```

2. **选择语言和版本**
   - 选择你需要的语言（如：English, French）
   - 选择最新版本（如：Common Voice Corpus 17.0）

3. **下载数据集**
   - 点击下载按钮
   - 选择格式：推荐下载 `.tar.gz` 格式
   - 文件大小：英语约30GB（完整版），可以下载较小的validated子集

4. **解压数据**
   ```bash
   # 解压下载的文件
   tar -xzf cv-corpus-17.0-2024-03-15-en.tar.gz
   ```

#### 数据结构

解压后的目录结构：
```
cv-corpus-17.0-2024-03-15/en/
├── clips/              # 音频文件目录
│   ├── common_voice_en_00001.mp3
│   ├── common_voice_en_00002.mp3
│   └── ...
├── validated.tsv       # 验证过的数据（推荐使用）
├── train.tsv          # 训练集
├── dev.tsv            # 验证集
└── test.tsv           # 测试集
```

#### 转换为项目所需格式

<function_calls>
<invoke name="Write">
<parameter name="content">#!/usr/bin/env python3
"""
Convert Common Voice dataset to project format.
"""
import argparse
import csv
import shutil
from pathlib import Path
import subprocess


def convert_audio(input_path: Path, output_path: Path):
    """Convert audio to 16kHz mono WAV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to convert
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',      # mono
        '-y',            # overwrite
        str(output_path)
    ]

    subprocess.run(cmd, check=True, capture_output=True)


def process_common_voice(cv_dir: Path, output_dir: Path, lang: str, max_samples: int = None):
    """
    Process Common Voice dataset.

    Args:
        cv_dir: Path to Common Voice directory (e.g., cv-corpus-17.0-2024-03-15/en/)
        output_dir: Output directory (e.g., data/raw)
        lang: Language code (e.g., 'en', 'fr')
        max_samples: Maximum number of samples to process (None for all)
    """
    clips_dir = cv_dir / "clips"
    tsv_file = cv_dir / "validated.tsv"  # Use validated subset

    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    if not tsv_file.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")

    # Create output directories
    output_wav_dir = output_dir / lang / "wav"
    output_wav_dir.mkdir(parents=True, exist_ok=True)

    transcripts = []

    # Read TSV file
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for idx, row in enumerate(reader):
            if max_samples and idx >= max_samples:
                break

            # Get audio filename and transcript
            audio_filename = row['path']
            sentence = row['sentence']

            # Input audio path (usually .mp3)
            input_audio = clips_dir / audio_filename

            if not input_audio.exists():
                print(f"Warning: Audio file not found: {input_audio}")
                continue

            # Output filename (convert to .wav)
            output_filename = f"{lang}_{idx:06d}.wav"
            output_audio = output_wav_dir / output_filename

            # Convert audio
            try:
                convert_audio(input_audio, output_audio)

                # Add to transcripts
                transcripts.append((output_filename.replace('.wav', ''), sentence))

                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} files...")

            except Exception as e:
                print(f"Error processing {audio_filename}: {e}")
                continue

    # Write transcripts.txt
    transcript_file = output_dir / lang / "transcripts.txt"
    with open(transcript_file, 'w', encoding='utf-8') as f:
        for filename, text in transcripts:
            f.write(f"{filename}\t{text}\n")

    print(f"\n✅ Conversion complete!")
    print(f"   Processed: {len(transcripts)} files")
    print(f"   Audio files: {output_wav_dir}")
    print(f"   Transcripts: {transcript_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert Common Voice to project format')
    parser.add_argument('--cv-dir', type=Path, required=True,
                        help='Common Voice directory (e.g., cv-corpus-17.0-2024-03-15/en/)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw'),
                        help='Output directory (default: data/raw)')
    parser.add_argument('--lang', type=str, required=True,
                        help='Language code (e.g., en, fr)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (default: all)')

    args = parser.parse_args()

    process_common_voice(args.cv_dir, args.output_dir, args.lang, args.max_samples)


if __name__ == '__main__':
    main()
