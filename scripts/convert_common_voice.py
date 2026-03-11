#!/usr/bin/env python3
"""
Convert Common Voice dataset to project format.
"""
import argparse
import csv
from pathlib import Path
import subprocess


def convert_audio(input_path: Path, output_path: Path):
    """Convert audio to 16kHz mono WAV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-ar', '16000',
        '-ac', '1',
        '-y',
        str(output_path)
    ]

    subprocess.run(cmd, check=True, capture_output=True)


def process_common_voice(cv_dir: Path, output_dir: Path, lang: str, max_samples: int = None):
    """
    Process Common Voice dataset.

    Args:
        cv_dir: Path to Common Voice directory
        output_dir: Output directory
        lang: Language code
        max_samples: Maximum number of samples to process
    """
    clips_dir = cv_dir / "clips"
    tsv_file = cv_dir / "validated.tsv"

    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    if not tsv_file.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")

    output_wav_dir = output_dir / lang / "wav"
    output_wav_dir.mkdir(parents=True, exist_ok=True)

    transcripts = []

    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for idx, row in enumerate(reader):
            if max_samples and idx >= max_samples:
                break

            audio_filename = row['path']
            sentence = row['sentence']

            input_audio = clips_dir / audio_filename

            if not input_audio.exists():
                print(f"Warning: Audio file not found: {input_audio}")
                continue

            output_filename = f"{lang}_{idx:06d}.wav"
            output_audio = output_wav_dir / output_filename

            try:
                convert_audio(input_audio, output_audio)
                transcripts.append((output_filename.replace('.wav', ''), sentence))

                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} files...")

            except Exception as e:
                print(f"Error processing {audio_filename}: {e}")
                continue

    transcript_file = output_dir / lang / "transcripts.txt"
    with open(transcript_file, 'w', encoding='utf-8') as f:
        for filename, text in transcripts:
            f.write(f"{filename}\t{text}\n")

    print(f"\nConversion complete!")
    print(f"Processed: {len(transcripts)} files")
    print(f"Audio files: {output_wav_dir}")
    print(f"Transcripts: {transcript_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert Common Voice to project format')
    parser.add_argument('--cv-dir', type=Path, required=True,
                        help='Common Voice directory')
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw'),
                        help='Output directory')
    parser.add_argument('--lang', type=str, required=True,
                        help='Language code')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')

    args = parser.parse_args()

    process_common_voice(args.cv_dir, args.output_dir, args.lang, args.max_samples)


if __name__ == '__main__':
    main()
