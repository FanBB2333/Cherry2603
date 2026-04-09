from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(".")
ASSET_DIR = ROOT / "report_assets"


def load_metrics(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def extract_curve(rows: list[dict[str, str]], field: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for row in rows:
        if row[field]:
            xs.append(int(row["step"]))
            ys.append(float(row[field]))
    return xs, ys


def figure_training_curves() -> None:
    rows = load_metrics(ROOT / "lab3_runs/main_songci5mb_gpu_baseline_20k/metrics.csv")
    train_x, train_y = extract_curve(rows, "train_loss")
    val_x, val_y = extract_curve(rows, "val_loss")

    plt.figure(figsize=(8, 4.8))
    plt.plot(train_x, train_y, label="Train loss", linewidth=1.2, alpha=0.85)
    plt.plot(val_x, val_y, label="Validation loss", linewidth=2.0)
    best_idx = min(range(len(val_y)), key=lambda i: val_y[i])
    plt.scatter([val_x[best_idx]], [val_y[best_idx]], color="crimson", zorder=5, label=f"Best val @ {val_x[best_idx]}")
    plt.xlabel("Step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Main 20k Baseline Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "training_curve_20k.png", dpi=180)
    plt.close()


def figure_hparam_comparison() -> None:
    summary = json.loads((ROOT / "lab3_runs/experiment_summary.json").read_text(encoding="utf-8"))
    labels = [
        "Baseline 5k",
        "Baseline 20k\n(best checkpoint region)",
        "Context 128",
        "Layers 4",
        "Dropout 0.0",
    ]
    values = [
        summary[0]["best_val_loss"],
        summary[1]["best_val_loss"],
        summary[2]["best_val_loss"],
        summary[3]["best_val_loss"],
        summary[4]["best_val_loss"],
    ]

    plt.figure(figsize=(8, 4.8))
    colors = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b"]
    plt.bar(labels, values, color=colors)
    plt.ylabel("Best validation loss")
    plt.title("Hyperparameter Comparison")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "hyperparameter_comparison.png", dpi=180)
    plt.close()


def figure_attention_benchmark() -> None:
    payload = json.loads((ROOT / "lab3_runs/attention_benchmark_gpu.json").read_text(encoding="utf-8"))
    backends = [item["backend"] for item in payload["results"]]
    time_ms = [item["avg_step_time_ms"] for item in payload["results"]]
    memory_mb = [item["peak_memory_mb"] for item in payload["results"]]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    axes[0].bar(backends, time_ms, color=["#52796f", "#bc4749"])
    axes[0].set_title("Average step time")
    axes[0].set_ylabel("ms")

    axes[1].bar(backends, memory_mb, color=["#52796f", "#bc4749"])
    axes[1].set_title("Peak memory")
    axes[1].set_ylabel("MB")

    fig.suptitle("Attention Backend Benchmark")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "attention_benchmark.png", dpi=180)
    plt.close(fig)


def figure_structure_comparison() -> None:
    payload = json.loads((ROOT / "lab3_runs/main_songci5mb_gpu_baseline_5k/structure_comparison.json").read_text(encoding="utf-8"))
    ref = payload["reference"]["raw_line_length_distribution"]
    gen = payload["generated"]["raw_line_length_distribution"]
    keys = sorted({int(k) for k in ref} | {int(k) for k in gen})

    ref_vals = [ref.get(str(k), 0) for k in keys]
    gen_vals = [gen.get(str(k), 0) for k in keys]
    ref_total = sum(ref_vals) or 1
    gen_total = sum(gen_vals) or 1
    ref_vals = [v / ref_total for v in ref_vals]
    gen_vals = [v / gen_total for v in gen_vals]

    plt.figure(figsize=(8, 4.8))
    plt.plot(keys, ref_vals, marker="o", linewidth=1.5, label="Reference corpus")
    plt.plot(keys, gen_vals, marker="s", linewidth=1.5, label="Generated samples")
    plt.xlabel("Raw line length")
    plt.ylabel("Relative frequency")
    plt.title("Line-Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "structure_line_lengths.png", dpi=180)
    plt.close()


def figure_probe_results() -> None:
    payload = json.loads((ROOT / "lab3_runs/main_songci5mb_gpu_baseline_5k/probe_newline.json").read_text(encoding="utf-8"))
    labels = ["Accuracy", "F1"]
    trained = [payload["trained_model"]["accuracy"], payload["trained_model"]["f1"]]
    random = [payload["random_init_model"]["accuracy"], payload["random_init_model"]["f1"]]

    x = [0, 1]
    width = 0.35

    plt.figure(figsize=(6.4, 4.2))
    plt.bar([i - width / 2 for i in x], trained, width=width, label="Trained model", color="#355070")
    plt.bar([i + width / 2 for i in x], random, width=width, label="Random init", color="#b56576")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Newline Probe Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "probe_newline.png", dpi=180)
    plt.close()


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    figure_training_curves()
    figure_hparam_comparison()
    figure_attention_benchmark()
    figure_structure_comparison()
    figure_probe_results()


if __name__ == "__main__":
    main()
