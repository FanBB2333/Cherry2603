from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PUNCT_TO_REMOVE = "，。！？；：、（）《》〈〉“”‘’〔〕【】—…·,.!?;:()[]\"' "
SAMPLE_SEPARATOR = "<<<LAB3_SAMPLE_SEPARATOR>>>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare corpus and generated-poem structure statistics.")
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--generated-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCT_TO_REMOVE)


def read_poems(file_path: Path) -> list[list[str]]:
    text = file_path.read_text(encoding="utf-8")
    if SAMPLE_SEPARATOR in text:
        poems = [poem for poem in text.split(SAMPLE_SEPARATOR) if poem.strip()]
    else:
        poems = [poem for poem in text.strip().split("\n\n") if poem.strip()]
    return [[line for line in poem.splitlines() if line.strip()] for poem in poems]


def normalized(counter: Counter[int | str]) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {str(key): value / total for key, value in counter.items()}


def l1_distance(counter_a: Counter[int | str], counter_b: Counter[int | str]) -> float:
    keys = set(counter_a) | set(counter_b)
    total_a = sum(counter_a.values()) or 1
    total_b = sum(counter_b.values()) or 1
    return sum(abs(counter_a.get(key, 0) / total_a - counter_b.get(key, 0) / total_b) for key in keys)


def compute_stats(file_path: Path) -> dict[str, object]:
    poems = read_poems(file_path)
    lines = [line for poem in poems for line in poem]

    raw_lengths = Counter(len(line) for line in lines)
    stripped_lengths = Counter(len(remove_punctuation(line)) for line in lines)
    lines_per_poem = Counter(len(poem) for poem in poems)
    line_endings = Counter(line[-1] for line in lines if line)
    punctuation_chars = sum(sum(1 for ch in line if ch in PUNCT_TO_REMOVE) for line in lines)
    total_chars = sum(len(line) for line in lines) or 1

    return {
        "file": str(file_path),
        "num_poems": len(poems),
        "num_lines": len(lines),
        "avg_lines_per_poem": len(lines) / len(poems) if poems else 0.0,
        "avg_raw_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0.0,
        "avg_stripped_line_length": sum(len(remove_punctuation(line)) for line in lines) / len(lines) if lines else 0.0,
        "punctuation_ratio": punctuation_chars / total_chars,
        "lines_per_poem_distribution": dict(lines_per_poem.most_common(10)),
        "raw_line_length_distribution": dict(raw_lengths.most_common(15)),
        "stripped_line_length_distribution": dict(stripped_lengths.most_common(15)),
        "line_ending_distribution": dict(line_endings.most_common(10)),
        "raw_lengths_counter": raw_lengths,
        "stripped_lengths_counter": stripped_lengths,
        "lines_per_poem_counter": lines_per_poem,
        "line_endings_counter": line_endings,
    }


def main() -> None:
    args = parse_args()

    reference = compute_stats(args.reference_file)
    generated = compute_stats(args.generated_file)

    comparison = {
        "line_count_l1": l1_distance(reference["lines_per_poem_counter"], generated["lines_per_poem_counter"]),
        "raw_line_length_l1": l1_distance(reference["raw_lengths_counter"], generated["raw_lengths_counter"]),
        "stripped_line_length_l1": l1_distance(reference["stripped_lengths_counter"], generated["stripped_lengths_counter"]),
        "line_ending_l1": l1_distance(reference["line_endings_counter"], generated["line_endings_counter"]),
        "avg_lines_per_poem_gap": generated["avg_lines_per_poem"] - reference["avg_lines_per_poem"],
        "avg_raw_line_length_gap": generated["avg_raw_line_length"] - reference["avg_raw_line_length"],
        "avg_stripped_line_length_gap": generated["avg_stripped_line_length"] - reference["avg_stripped_line_length"],
        "punctuation_ratio_gap": generated["punctuation_ratio"] - reference["punctuation_ratio"],
    }

    payload = {
        "reference": {k: v for k, v in reference.items() if not k.endswith("_counter")},
        "generated": {k: v for k, v in generated.items() if not k.endswith("_counter")},
        "comparison": comparison,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with args.output_md.open("w", encoding="utf-8") as handle:
        handle.write("# Structure Comparison\n\n")
        handle.write(f"Reference: `{args.reference_file}`\n\n")
        handle.write(f"Generated: `{args.generated_file}`\n\n")
        handle.write("## Reference\n")
        handle.write(json.dumps(payload["reference"], ensure_ascii=False, indent=2))
        handle.write("\n\n## Generated\n")
        handle.write(json.dumps(payload["generated"], ensure_ascii=False, indent=2))
        handle.write("\n\n## Comparison\n")
        handle.write(json.dumps(payload["comparison"], ensure_ascii=False, indent=2))
        handle.write("\n")


if __name__ == "__main__":
    main()
