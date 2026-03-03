from __future__ import annotations

from pathlib import Path

import numpy as np

from lab2.embeddings.store import load_vector, read_index


def load_embeddings_from_index(index_path: Path) -> dict[str, np.ndarray]:
    items = read_index(index_path)
    return {it.word: load_vector(it.path) for it in items}


def restrict_to_words(embeddings: dict[str, np.ndarray], words: set[str]) -> dict[str, np.ndarray]:
    return {w: v for w, v in embeddings.items() if w in words}


def as_matrix(embeddings: dict[str, np.ndarray], *, words: list[str]) -> np.ndarray:
    return np.stack([embeddings[w] for w in words])

