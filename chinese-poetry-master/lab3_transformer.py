from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DatasetBundle:
    text: str
    vocab: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]
    encoded: list[int]
    train_ids: list[int]
    val_ids: list[int]


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_expansion: int = 4
    dropout: float = 0.1
    attention_backend: str = "sdpa"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_to_bpc(loss_value: float) -> float:
    return loss_value / math.log(2)


def load_char_dataset(input_file: Path, val_ratio: float = 0.1) -> DatasetBundle:
    text = input_file.read_text(encoding="utf-8")
    vocab = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(vocab)}
    itos = {idx: ch for idx, ch in enumerate(vocab)}
    encoded = [stoi[ch] for ch in text]

    split_idx = int(len(encoded) * (1 - val_ratio))
    train_ids = encoded[:split_idx]
    val_ids = encoded[split_idx:]

    return DatasetBundle(
        text=text,
        vocab=vocab,
        stoi=stoi,
        itos=itos,
        encoded=encoded,
        train_ids=train_ids,
        val_ids=val_ids,
    )


def get_batch(split_ids: list[int], batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(split_ids) - context_length - 1
    if max_start <= 0:
        raise ValueError("Sequence too short for the chosen context length.")

    starts = torch.randint(0, max_start + 1, (batch_size,))
    windows = [split_ids[start : start + context_length + 1] for start in starts.tolist()]
    batch = torch.tensor(windows, dtype=torch.long, device=device)
    return batch[:, :-1], batch[:, 1:]


def encode_text(text: str, stoi: dict[str, int]) -> list[int]:
    unknown = sorted({ch for ch in text if ch not in stoi})
    if unknown:
        preview = "".join(unknown[:10])
        raise ValueError(f"Prompt contains {len(unknown)} unseen characters: {preview!r}")
    return [stoi[ch] for ch in text]


def decode_ids(token_ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[token_id] for token_id in token_ids)


def default_prompt(text: str, prompt_chars: int = 48) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:prompt_chars]

    prompt = ""
    for line in lines:
        addition = f"{line}\n"
        if len(prompt) + len(addition) > prompt_chars and prompt:
            break
        prompt += addition
    return prompt or text[:prompt_chars]


def choose_device(requested_device: str) -> str:
    if requested_device == "auto":
        if torch.cuda.is_available():
            try:
                torch.empty(1, device="cuda")
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"
    if requested_device.startswith("cuda"):
        # In this environment, torch.cuda.is_available() can return a false
        # negative during script startup even though CUDA allocations succeed
        # later in the same process. For an explicit user request, trust it.
        return requested_device
    return requested_device


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, attention_backend: str) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if attention_backend not in {"sdpa", "classic"}:
            raise ValueError("attention_backend must be 'sdpa' or 'classic'.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.attention_backend = attention_backend

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        if self.attention_backend == "sdpa":
            dropout_p = self.dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            attn_out = weights @ v

        attn_out = attn_out.transpose(1, 2).contiguous().view(x.shape[0], seq_len, self.hidden_dim)
        return self.resid_dropout(self.out_proj(attn_out))


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_expansion: int, dropout: float) -> None:
        super().__init__()
        inner_dim = hidden_dim * mlp_expansion
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadCausalSelfAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attention_backend=config.attention_backend,
        )
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = FeedForward(
            hidden_dim=config.hidden_dim,
            mlp_expansion=config.mlp_expansion,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CharacterGPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.final_ln = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _, seq_len = x.shape
        if seq_len > self.config.context_length:
            raise ValueError("Sequence length exceeds model context length.")

        positions = torch.arange(seq_len, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions)
        h = self.dropout(h)
        for block in self.blocks:
            h = block(h)
        h = self.final_ln(h)
        logits = self.output_projection(h)
        if return_hidden:
            return logits, h
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        for _ in range(max_new_tokens):
            input_crop = input_ids[:, -self.config.context_length :]
            logits = self(input_crop)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                top_values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                cutoff = top_values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def compute_loss(model: CharacterGPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    batch_size, seq_len, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        y.reshape(batch_size * seq_len),
    )


@torch.no_grad()
def evaluate_average_loss(
    model: CharacterGPT,
    split_ids: list[int],
    batch_size: int,
    context_length: int,
    num_batches: int,
    device: str,
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x_batch, y_batch = get_batch(split_ids, batch_size, context_length, device)
        losses.append(compute_loss(model, x_batch, y_batch).item())
    return sum(losses) / len(losses)


def build_optimizer(model: CharacterGPT, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def learning_rate_for_step(step: int, warmup_steps: int, base_learning_rate: float) -> float:
    if warmup_steps <= 0:
        return base_learning_rate
    warmup_progress = min(step / warmup_steps, 1.0)
    return base_learning_rate * warmup_progress


def save_checkpoint(
    checkpoint_path: Path,
    model: CharacterGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    dataset: DatasetBundle,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_config": asdict(model.config),
        "run_config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": dataset.vocab,
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: str = "cpu") -> tuple[CharacterGPT, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**payload["model_config"])
    model = CharacterGPT(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
