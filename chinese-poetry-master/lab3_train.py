from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path

import torch

from lab3_transformer import (
    CharacterGPT,
    ModelConfig,
    build_optimizer,
    choose_device,
    compute_loss,
    decode_ids,
    default_prompt,
    encode_text,
    evaluate_average_loss,
    get_batch,
    learning_rate_for_step,
    load_char_dataset,
    loss_to_bpc,
    save_checkpoint,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal lab3 training entrypoint for character-level GPT poetry generation.")
    parser.add_argument("--input-file", type=Path, default=Path("songci_main_5mb.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("lab3_runs/main_experiment"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-backend", choices=["sdpa", "classic"], default="sdpa")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--num-val-batches", type=int, default=8)
    parser.add_argument("--generation-every", type=int, default=1000)
    parser.add_argument("--generate-tokens", type=int, default=200)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-chars", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def append_sample(samples_path: Path, step: int, prompt_text: str, generated_text: str) -> None:
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("a", encoding="utf-8") as handle:
        handle.write(f"=== step {step} ===\n")
        handle.write("PROMPT:\n")
        handle.write(prompt_text)
        handle.write("\n\nGENERATED:\n")
        handle.write(generated_text)
        handle.write("\n\n")


def should_fire(interval: int, step: int, always_on_final_step: bool, max_steps: int) -> bool:
    if always_on_final_step and step == max_steps:
        return True
    return interval > 0 and step % interval == 0


def should_print(step: int, args: argparse.Namespace) -> bool:
    if step == 1 or step == args.max_steps:
        return True
    if args.print_every > 0 and step % args.print_every == 0:
        return True
    if should_fire(args.eval_every, step, always_on_final_step=False, max_steps=args.max_steps):
        return True
    if should_fire(args.generation_every, step, always_on_final_step=True, max_steps=args.max_steps):
        return True
    if should_fire(args.checkpoint_every, step, always_on_final_step=True, max_steps=args.max_steps):
        return True
    return False


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)

    dataset = load_char_dataset(args.input_file, val_ratio=args.val_ratio)
    model_config = ModelConfig(
        vocab_size=len(dataset.vocab),
        context_length=args.context_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_expansion=args.mlp_expansion,
        dropout=args.dropout,
        attention_backend=args.attention_backend,
    )

    model = CharacterGPT(model_config).to(device)
    optimizer = build_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    samples_path = args.output_dir / "samples.txt"
    checkpoints_dir = args.output_dir / "checkpoints"
    run_config_path = args.output_dir / "run_config.json"

    run_config = vars(args).copy()
    run_config["device"] = device
    run_config["model_config"] = asdict(model_config)
    run_config["vocab_size"] = len(dataset.vocab)
    run_config["train_tokens"] = len(dataset.train_ids)
    run_config["val_tokens"] = len(dataset.val_ids)
    run_config["input_file"] = str(args.input_file)
    run_config["output_dir"] = str(args.output_dir)
    write_json(run_config_path, run_config)

    prompt_text = args.prompt or default_prompt(dataset.text, prompt_chars=args.prompt_chars)
    prompt_ids = encode_text(prompt_text, dataset.stoi)

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "learning_rate",
                "train_loss",
                "train_bpc",
                "val_loss",
                "val_bpc",
                "step_time_sec",
            ],
        )
        writer.writeheader()

        print("=== Formal lab3 training run ===")
        print(f"input_file={args.input_file}")
        print(f"output_dir={args.output_dir}")
        print(f"device={device}")
        print(f"attention_backend={args.attention_backend}")
        print(f"vocab_size={len(dataset.vocab)}")
        print(f"train_tokens={len(dataset.train_ids)}")
        print(f"val_tokens={len(dataset.val_ids)}")
        print(
            "model="
            f"layers:{args.num_layers}, heads:{args.num_heads}, hidden_dim:{args.hidden_dim}, "
            f"context:{args.context_length}, dropout:{args.dropout}"
        )
        print(
            "training="
            f"batch_size:{args.batch_size}, lr:{args.learning_rate}, warmup:{args.warmup_steps}, "
            f"steps:{args.max_steps}, eval_every:{args.eval_every}, generation_every:{args.generation_every}"
        )

        for step in range(1, args.max_steps + 1):
            step_start = time.perf_counter()
            x_train, y_train = get_batch(dataset.train_ids, args.batch_size, args.context_length, device)
            model.train()

            current_lr = learning_rate_for_step(step, args.warmup_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            train_loss = compute_loss(model, x_train, y_train)
            train_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            step_time_sec = time.perf_counter() - step_start
            train_loss_value = train_loss.item()
            train_bpc_value = loss_to_bpc(train_loss_value)

            row = {
                "step": step,
                "learning_rate": current_lr,
                "train_loss": train_loss_value,
                "train_bpc": train_bpc_value,
                "val_loss": "",
                "val_bpc": "",
                "step_time_sec": step_time_sec,
            }

            do_eval = step == 1 or should_fire(args.eval_every, step, always_on_final_step=False, max_steps=args.max_steps)
            do_print = should_print(step, args)

            if do_eval:
                val_loss = evaluate_average_loss(
                    model=model,
                    split_ids=dataset.val_ids,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    num_batches=args.num_val_batches,
                    device=device,
                )
                val_bpc = loss_to_bpc(val_loss)
                row["val_loss"] = val_loss
                row["val_bpc"] = val_bpc
                if do_print:
                    print(
                        f"step={step:05d} "
                        f"train_loss={train_loss_value:.4f} train_bpc={train_bpc_value:.4f} "
                        f"val_loss={val_loss:.4f} val_bpc={val_bpc:.4f} "
                        f"lr={current_lr:.6f} step_time={step_time_sec:.3f}s"
                    )
            elif do_print:
                print(
                    f"step={step:05d} "
                    f"train_loss={train_loss_value:.4f} train_bpc={train_bpc_value:.4f} "
                    f"lr={current_lr:.6f} step_time={step_time_sec:.3f}s"
                )

            writer.writerow(row)
            handle.flush()

            if should_fire(args.generation_every, step, always_on_final_step=True, max_steps=args.max_steps):
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                generated_ids = model.generate(
                    prompt_tensor,
                    max_new_tokens=args.generate_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )[0].tolist()
                generated_text = decode_ids(generated_ids, dataset.itos)
                append_sample(samples_path, step, prompt_text, generated_text)
                print(f"[sample @ step {step}]\n{generated_text}\n")

            if should_fire(args.checkpoint_every, step, always_on_final_step=True, max_steps=args.max_steps):
                checkpoint_path = checkpoints_dir / f"model_step_{step:05d}.pt"
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=run_config,
                    dataset=dataset,
                )
                print(f"checkpoint_saved={checkpoint_path}")


if __name__ == "__main__":
    main()
