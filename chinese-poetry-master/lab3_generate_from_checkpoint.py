from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from lab3_transformer import DatasetBundle, decode_ids, default_prompt, encode_text, load_char_dataset, load_checkpoint, set_seed

SAMPLE_SEPARATOR = "\n<<<LAB3_SAMPLE_SEPARATOR>>>\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a corpus of samples from a saved lab3 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--prompt-chars", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_prompt(dataset: DatasetBundle, prompt_chars: int, rng: random.Random) -> str:
    poems = [poem for poem in dataset.text.split("\n\n") if poem.strip()]
    if not poems:
        return default_prompt(dataset.text, prompt_chars=prompt_chars)

    poem = rng.choice(poems)
    lines = [line for line in poem.splitlines() if line.strip()]
    if not lines:
        return default_prompt(dataset.text, prompt_chars=prompt_chars)

    prompt = ""
    for line in lines:
        candidate = f"{line}\n"
        if len(prompt) + len(candidate) > prompt_chars and prompt:
            break
        prompt += candidate
    return prompt or default_prompt(dataset.text, prompt_chars=prompt_chars)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    model, payload = load_checkpoint(args.checkpoint, device=args.device)
    run_config = payload["run_config"]
    input_file = Path(run_config["input_file"])
    dataset = load_char_dataset(input_file)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w", encoding="utf-8") as handle:
        for sample_idx in range(args.num_samples):
            prompt_text = sample_prompt(dataset, args.prompt_chars, rng)
            prompt_ids = encode_text(prompt_text, dataset.stoi)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
            generated_ids = model.generate(
                prompt_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )[0].tolist()
            continuation_ids = generated_ids[len(prompt_ids) :]
            continuation_text = decode_ids(continuation_ids, dataset.itos).strip()

            if continuation_text:
                handle.write(continuation_text)
            if sample_idx != args.num_samples - 1:
                handle.write(SAMPLE_SEPARATOR)


if __name__ == "__main__":
    main()
