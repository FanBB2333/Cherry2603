#!/usr/bin/env python3
"""
Evaluate phoneme recognition performance using PER (Phoneme Error Rate).
"""
import argparse
import json
from pathlib import Path
import Levenshtein


def compute_per(reference: str, hypothesis: str) -> float:
    """
    Compute Phoneme Error Rate (PER).

    PER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference length
    """
    if not reference:
        return 0.0 if not hypothesis else float('inf')

    # Levenshtein distance gives us the edit distance (S + D + I)
    distance = Levenshtein.distance(reference, hypothesis)
    per = distance / len(reference)
    return per


def evaluate_manifest(input_manifest: Path, output_metrics: Path):
    """Evaluate all predictions in manifest and compute metrics."""
    total_per = 0.0
    count = 0
    per_by_snr = {}
    per_by_lang = {}

    with open(input_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())

            ref_phon = entry.get('ref_phon', '')
            pred_phon = entry.get('pred_phon', '')
            snr_db = entry.get('snr_db')
            lang = entry.get('lang')

            # Compute PER
            per = compute_per(ref_phon, pred_phon)

            total_per += per
            count += 1

            # Aggregate by SNR
            snr_key = str(snr_db) if snr_db is not None else 'clean'
            if snr_key not in per_by_snr:
                per_by_snr[snr_key] = []
            per_by_snr[snr_key].append(per)

            # Aggregate by language
            if lang not in per_by_lang:
                per_by_lang[lang] = []
            per_by_lang[lang].append(per)

    # Compute average metrics
    avg_per = total_per / count if count > 0 else 0.0

    avg_per_by_snr = {
        snr: sum(pers) / len(pers)
        for snr, pers in per_by_snr.items()
    }

    avg_per_by_lang = {
        lang: sum(pers) / len(pers)
        for lang, pers in per_by_lang.items()
    }

    # Prepare metrics
    metrics = {
        'avg_per': avg_per,
        'count': count,
        'per_by_snr': avg_per_by_snr,
        'per_by_lang': avg_per_by_lang
    }

    # Write metrics
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Average PER: {avg_per:.4f}")
    print(f"Evaluated {count} utterances")
    print(f"Metrics saved to: {output_metrics}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate phoneme recognition performance')
    parser.add_argument('--input-manifest', type=Path, required=True,
                        help='Input manifest with predictions')
    parser.add_argument('--output-metrics', type=Path, required=True,
                        help='Output metrics file (JSON)')

    args = parser.parse_args()
    evaluate_manifest(args.input_manifest, args.output_metrics)


if __name__ == '__main__':
    main()
