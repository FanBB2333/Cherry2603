#!/usr/bin/env python3
"""
Plot PER vs SNR curves for each language and cross-language mean.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_dir: Path, languages: list, snr_levels: list):
    """Load metrics for all languages and SNR levels."""
    data = {}

    for lang in languages:
        data[lang] = {}

        # Load clean metrics
        clean_metrics_path = metrics_dir / lang / "clean.json"
        if clean_metrics_path.exists():
            with open(clean_metrics_path, 'r') as f:
                metrics = json.load(f)
                data[lang]['clean'] = metrics.get('avg_per', 0.0)

        # Load noisy metrics
        for snr in snr_levels:
            metrics_path = metrics_dir / lang / f"snr_{snr}.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    data[lang][snr] = metrics.get('avg_per', 0.0)

    return data


def plot_per_vs_snr(data: dict, snr_levels: list, output_path: Path):
    """Plot PER vs SNR for each language and cross-language mean."""
    plt.figure(figsize=(10, 6))

    # Prepare SNR axis (including clean as infinity)
    snr_axis = sorted(snr_levels)

    # Plot each language
    for lang, lang_data in data.items():
        per_values = []
        for snr in snr_axis:
            per = lang_data.get(snr, None)
            if per is not None:
                per_values.append(per)
            else:
                per_values.append(np.nan)

        plt.plot(snr_axis, per_values, marker='o', label=lang, linewidth=2)

    # Compute and plot cross-language mean
    mean_per_values = []
    for snr in snr_axis:
        snr_pers = [lang_data.get(snr) for lang_data in data.values()]
        snr_pers = [p for p in snr_pers if p is not None]
        if snr_pers:
            mean_per_values.append(np.mean(snr_pers))
        else:
            mean_per_values.append(np.nan)

    plt.plot(snr_axis, mean_per_values, marker='s', label='Mean',
             linewidth=3, linestyle='--', color='black')

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Phoneme Error Rate (PER)', fontsize=12)
    plt.title('ASR Performance vs Noise Level', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot PER vs SNR curves')
    parser.add_argument('--metrics-dir', type=Path, required=True,
                        help='Directory containing metrics files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output plot path')

    args = parser.parse_args()

    # Load parameters from params.yaml
    import yaml
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    languages = params['languages']
    snr_levels = params['snr_levels']

    data = load_metrics(args.metrics_dir, languages, snr_levels)
    plot_per_vs_snr(data, snr_levels, args.output)


if __name__ == '__main__':
    main()
