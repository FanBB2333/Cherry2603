from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lab3_transformer import MultiHeadCausalSelfAttention, choose_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SDPA vs classic masked attention for lab3.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def measure_backend(
    backend: str,
    device: str,
    batch_size: int,
    context_length: int,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    warmup_iters: int,
    benchmark_iters: int,
) -> tuple[float, float | None]:
    module = MultiHeadCausalSelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        attention_backend=backend,
    ).to(device)
    module.train()

    for _ in range(warmup_iters):
        x = torch.randn(batch_size, context_length, hidden_dim, device=device, requires_grad=True)
        out = module(x)
        loss = out.square().mean()
        loss.backward()
        module.zero_grad(set_to_none=True)

    synchronize_if_needed(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(benchmark_iters):
        x = torch.randn(batch_size, context_length, hidden_dim, device=device, requires_grad=True)
        out = module(x)
        loss = out.square().mean()
        loss.backward()
        module.zero_grad(set_to_none=True)
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start

    avg_time_ms = elapsed * 1000 / benchmark_iters
    peak_memory_mb = None
    if device.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return avg_time_ms, peak_memory_mb


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)

    print("=== Attention benchmark ===")
    print(
        f"device={device} batch_size={args.batch_size} context_length={args.context_length} "
        f"hidden_dim={args.hidden_dim} num_heads={args.num_heads}"
    )

    results = []
    for backend in ("sdpa", "classic"):
        avg_time_ms, peak_memory_mb = measure_backend(
            backend=backend,
            device=device,
            batch_size=args.batch_size,
            context_length=args.context_length,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
        )
        results.append(
            {
                "backend": backend,
                "avg_step_time_ms": avg_time_ms,
                "peak_memory_mb": peak_memory_mb,
            }
        )
        if peak_memory_mb is None:
            print(f"{backend:7s} avg_step_time_ms={avg_time_ms:.3f}")
        else:
            print(f"{backend:7s} avg_step_time_ms={avg_time_ms:.3f} peak_memory_mb={peak_memory_mb:.2f}")

    if args.output_json is not None:
        payload = {
            "device": device,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "warmup_iters": args.warmup_iters,
            "benchmark_iters": args.benchmark_iters,
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
