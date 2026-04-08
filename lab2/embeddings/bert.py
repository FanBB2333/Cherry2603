from __future__ import annotations

import re
from typing import Iterable

from lab2.utils import normalize_token


def _find_full_word_spans(sentence: str, word: str) -> list[tuple[int, int]]:
    """
    Return (start, end) spans (character indices) of full-word matches (case-insensitive).

    Full-word is approximated with word-char boundaries: the match cannot be preceded/followed
    by `\\w` (letters/digits/_).
    """

    w = re.escape(word)
    pattern = re.compile(rf"(?i)(?<!\w){w}(?!\w)")
    return [(m.start(), m.end()) for m in pattern.finditer(sentence)]


class BertEmbedder:
    def __init__(self, *, model, tokenizer, device: str = "cpu"):
        import torch

        self._torch = torch
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # Make sure we can return offsets (requires a "fast" tokenizer).
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Tokenizer must be a fast tokenizer (use_fast=True) to get offset mappings.")

    @property
    def num_layers(self) -> int:
        return int(getattr(self.model.config, "num_hidden_layers", 0))

    def embed_word_in_sentence(self, *, sentence: str, target_word: str, layer: int) -> "numpy.ndarray":
        return self.embed_word_in_sentence_layers(sentence=sentence, target_word=target_word, layers=[layer])[layer]

    def embed_word_in_sentence_layers(
        self,
        *,
        sentence: str,
        target_word: str,
        layers: Iterable[int],
        occurrence_strategy: str = "mean",
    ) -> dict[int, "numpy.ndarray"]:
        """
        Compute the target word embedding for several BERT layers in a single forward pass.

        - `occurrence_strategy="mean"` averages embeddings across multiple occurrences in the sentence.
        - Subword pieces are mean-pooled, per the PDF.
        """

        layers = list(layers)
        if not layers:
            raise ValueError("layers must be non-empty")

        max_layer = max(layers)
        if max_layer < 0 or max_layer > self.num_layers:
            raise ValueError(f"layer out of range: {max_layer} (valid: 0..{self.num_layers})")

        spans = _find_full_word_spans(sentence, target_word)
        if not spans:
            raise ValueError(f"Target word not found as a full-word match: {target_word!r}")

        enc = self.tokenizer(
            sentence,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with self._torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)

        # Gather token indices for each occurrence span.
        occ_token_indices: list[list[int]] = []
        for span_start, span_end in spans:
            token_ids: list[int] = []
            for i, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue  # special token
                if e <= span_start or s >= span_end:
                    continue
                token_ids.append(i)
            if token_ids:
                occ_token_indices.append(token_ids)

        if not occ_token_indices:
            raise ValueError(
                f"Found word spans for {target_word!r} but could not align them to token offsets."
            )

        hidden_states = out.hidden_states  # tuple[layer_index] => (B, T, H)
        result: dict[int, "numpy.ndarray"] = {}

        for layer in layers:
            hs = hidden_states[layer][0]  # (T, H)
            per_occ: list["numpy.ndarray"] = []
            for token_ids in occ_token_indices:
                tok_vecs = hs[token_ids, :]  # (n_subtokens, H)
                per_occ.append(tok_vecs.mean(dim=0).detach().cpu().numpy())

            if occurrence_strategy == "mean":
                import numpy as np

                vec = np.mean(np.stack(per_occ, axis=0), axis=0)
            elif occurrence_strategy == "first":
                vec = per_occ[0]
            else:
                raise ValueError(f"Unknown occurrence_strategy: {occurrence_strategy}")

            result[layer] = vec

        return result


def normalize_word(word: str) -> str:
    return normalize_token(word)
