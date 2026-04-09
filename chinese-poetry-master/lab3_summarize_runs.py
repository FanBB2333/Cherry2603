from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multiple lab3 run directories.")
    parser.add_argument("--run-dir", action="append", required=True, help="Path to a run directory containing metrics.csv")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def summarize_run(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open()))
    val_rows = [row for row in rows if row["val_loss"]]
    best_val = min(val_rows, key=lambda row: float(row["val_loss"])) if val_rows else None
    final = rows[-1]

    return {
        "run_dir": str(run_dir),
        "final_step": int(final["step"]),
        "final_train_loss": float(final["train_loss"]),
        "final_train_bpc": float(final["train_bpc"]),
        "final_val_loss": float(final["val_loss"]) if final["val_loss"] else None,
        "final_val_bpc": float(final["val_bpc"]) if final["val_bpc"] else None,
        "best_val_step": int(best_val["step"]) if best_val else None,
        "best_val_loss": float(best_val["val_loss"]) if best_val else None,
        "best_val_bpc": float(best_val["val_bpc"]) if best_val else None,
    }


def main() -> None:
    args = parse_args()
    summaries = [summarize_run(Path(run_dir)) for run_dir in args.run_dir]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    with args.output_md.open("w", encoding="utf-8") as handle:
        handle.write("# Run Summary\n\n")
        for summary in summaries:
            handle.write(json.dumps(summary, ensure_ascii=False, indent=2))
            handle.write("\n\n")


if __name__ == "__main__":
    main()
