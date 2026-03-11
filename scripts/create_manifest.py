#!/usr/bin/env python3
"""
Create manifest from raw audio files with phoneme transcription using espeak-ng.
"""
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
import tempfile
import shutil
import soundfile as sf


def text_to_phonemes(text: str, lang: str) -> str:
    """Convert text to phonemes using espeak-ng."""
    # Map language codes to espeak-ng language codes
    lang_map = {
        'en': 'en-us',
        'fr': 'fr-fr',
        'es': 'es',
        'de': 'de',
    }
    espeak_lang = lang_map.get(lang, lang)

    result = subprocess.run(
        ['espeak-ng', '-q', '-v', espeak_lang, '--ipa', text],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def create_manifest(raw_dir: Path, lang: str, output_path: Path):
    """Create manifest for a language."""
    wav_dir = raw_dir / lang / "wav"
    text_file = raw_dir / lang / "transcripts.txt"

    if not wav_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {wav_dir}")
    if not text_file.exists():
        raise FileNotFoundError(f"Transcript file not found: {text_file}")

    # Read transcripts
    transcripts = {}
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id] = text

    # Create manifest entries
    manifest_entries = []
    for wav_file in sorted(wav_dir.glob("*.wav")):
        stem = wav_file.stem
        utt_id = f"{lang}_{stem}"

        if stem not in transcripts:
            print(f"Warning: No transcript for {stem}, skipping")
            continue

        ref_text = transcripts[stem]

        # Get audio info
        info = sf.info(wav_file)
        sr = info.samplerate
        duration_s = info.duration

        # Convert text to phonemes
        try:
            ref_phon = text_to_phonemes(ref_text, lang)
        except Exception as e:
            print(f"Warning: Failed to convert text to phonemes for {utt_id}: {e}")
            continue

        # Compute MD5
        audio_md5 = compute_md5(wav_file)

        # Create relative path
        wav_path = str(wav_file)

        entry = {
            "utt_id": utt_id,
            "lang": lang,
            "wav_path": wav_path,
            "ref_text": ref_text,
            "ref_phon": ref_phon,
            "sr": sr,
            "duration_s": round(duration_s, 2),
            "snr_db": None,
            "audio_md5": audio_md5
        }
        manifest_entries.append(entry)

    # Write manifest atomically
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                     dir=output_path.parent,
                                     suffix='.tmp') as tmp_file:
        for entry in manifest_entries:
            json.dump(entry, tmp_file, ensure_ascii=False)
            tmp_file.write('\n')
        tmp_path = Path(tmp_file.name)

    # Atomic rename
    shutil.move(str(tmp_path), str(output_path))
    print(f"Created manifest with {len(manifest_entries)} entries: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create manifest from raw audio files')
    parser.add_argument('--raw-dir', type=Path, required=True, help='Raw data directory')
    parser.add_argument('--lang', type=str, required=True, help='Language code')
    parser.add_argument('--output', type=Path, required=True, help='Output manifest path')

    args = parser.parse_args()
    create_manifest(args.raw_dir, args.lang, args.output)


if __name__ == '__main__':
    main()
