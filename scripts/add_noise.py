#!/usr/bin/env python3
"""
Add noise to audio files at specified SNR levels.
"""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
import numpy as np
import soundfile as sf


def add_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add white Gaussian noise to signal at specified SNR."""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=signal.shape,
    )
    return signal + noise


def add_noise_to_file(
    input_wav: str,
    output_wav: str,
    snr_db: float,
    seed: int | None = None,
) -> None:
    """Add noise to a wav file and save the result."""
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")

    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)

    # Ensure output directory exists
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, noisy_signal, sr)


def process_manifest(input_manifest: Path, output_manifest: Path,
                     snr_db: float, noisy_dir: Path, seed: int):
    """Process all audio files in manifest and create noisy versions."""
    manifest_entries = []

    with open(input_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())

            # Generate noisy audio path
            original_path = Path(entry['wav_path'])
            noisy_path = noisy_dir / entry['lang'] / f"snr_{snr_db}" / original_path.name

            # Add noise to audio file
            # Use utt_id hash as part of seed for reproducibility
            file_seed = seed + hash(entry['utt_id']) % (2**31)
            add_noise_to_file(
                entry['wav_path'],
                str(noisy_path),
                snr_db,
                seed=file_seed
            )

            # Update manifest entry
            new_entry = entry.copy()
            new_entry['wav_path'] = str(noisy_path)
            new_entry['snr_db'] = snr_db

            manifest_entries.append(new_entry)

    # Write manifest atomically
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                     dir=output_manifest.parent,
                                     suffix='.tmp') as tmp_file:
        for entry in manifest_entries:
            json.dump(entry, tmp_file, ensure_ascii=False)
            tmp_file.write('\n')
        tmp_path = Path(tmp_file.name)

    shutil.move(str(tmp_path), str(output_manifest))
    print(f"Created noisy manifest with {len(manifest_entries)} entries at SNR={snr_db}dB: {output_manifest}")


def main():
    parser = argparse.ArgumentParser(description='Add noise to audio files')
    parser.add_argument('--input-manifest', type=Path, required=True,
                        help='Input manifest (clean audio)')
    parser.add_argument('--output-manifest', type=Path, required=True,
                        help='Output manifest (noisy audio)')
    parser.add_argument('--snr-db', type=float, required=True,
                        help='Signal-to-noise ratio in dB')
    parser.add_argument('--noisy-dir', type=Path, required=True,
                        help='Directory for noisy audio files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    process_manifest(args.input_manifest, args.output_manifest,
                     args.snr_db, args.noisy_dir, args.seed)


if __name__ == '__main__':
    main()
