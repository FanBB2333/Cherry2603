from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from lab3_transformer import CharacterGPT, get_batch, load_char_dataset, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether hidden states encode upcoming newline boundaries.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_features(
    model: CharacterGPT,
    split_ids: list[int],
    newline_id: int,
    batch_size: int,
    num_batches: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            x_batch, y_batch = get_batch(split_ids, batch_size, model.config.context_length, device)
            _, hidden = model(x_batch, return_hidden=True)
            feature_batch = hidden.reshape(-1, hidden.shape[-1]).cpu()
            label_batch = (y_batch.reshape(-1) == newline_id).float().cpu()
            features.append(feature_batch)
            labels.append(label_batch)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": labels.mean().item(),
    }


def train_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int,
    lr: float,
    device: str,
) -> dict[str, float]:
    probe = nn.Linear(train_features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x_train = train_features.to(device)
    y_train = train_labels.to(device).unsqueeze(1)
    x_val = val_features.to(device)
    y_val = val_labels.to(device).unsqueeze(1)

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(x_val).squeeze(1).cpu()
    return compute_binary_metrics(val_logits, val_labels)


def run_probe(model: CharacterGPT, train_ids: list[int], val_ids: list[int], newline_id: int, args: argparse.Namespace) -> dict[str, float]:
    train_features, train_labels = collect_features(
        model=model,
        split_ids=train_ids,
        newline_id=newline_id,
        batch_size=args.batch_size,
        num_batches=args.feature_batches,
        device=args.device,
    )
    val_features, val_labels = collect_features(
        model=model,
        split_ids=val_ids,
        newline_id=newline_id,
        batch_size=args.batch_size,
        num_batches=max(1, args.feature_batches // 2),
        device=args.device,
    )
    return train_probe(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    trained_model, payload = load_checkpoint(args.checkpoint, device=args.device)
    run_config = payload["run_config"]
    dataset = load_char_dataset(Path(run_config["input_file"]))

    if "\n" not in dataset.stoi:
        raise ValueError("Newline token not found in dataset vocabulary.")

    newline_id = dataset.stoi["\n"]

    random_model = CharacterGPT(trained_model.config).to(args.device)
    random_model.eval()

    trained_metrics = run_probe(trained_model, dataset.train_ids, dataset.val_ids, newline_id, args)
    random_metrics = run_probe(random_model, dataset.train_ids, dataset.val_ids, newline_id, args)

    payload = {
        "checkpoint": str(args.checkpoint),
        "feature_batches": args.feature_batches,
        "batch_size": args.batch_size,
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
        "trained_model": trained_metrics,
        "random_init_model": random_metrics,
        "f1_gain": trained_metrics["f1"] - random_metrics["f1"],
        "accuracy_gain": trained_metrics["accuracy"] - random_metrics["accuracy"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
