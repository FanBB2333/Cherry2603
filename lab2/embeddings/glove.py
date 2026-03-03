from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lab2.utils import ensure_dir, normalize_token


def _default_vocab_cache_path(glove_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{glove_path.name}.vocab.txt"


def load_glove_vocab(glove_path: Path, *, cache_dir: Path | None = None) -> set[str]:
    """
    Load the vocabulary (word list) from a GloVe text file.

    To keep subsequent runs fast, we cache the vocab as a newline-separated text file next to
    the GloVe file (or in `cache_dir` if provided).
    """

    if cache_dir is None:
        cache_dir = glove_path.parent
    ensure_dir(cache_dir)

    cache_path = _default_vocab_cache_path(glove_path, cache_dir)
    if cache_path.exists():
        return {line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    vocab: set[str] = set()
    with glove_path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip().split(" ")
            if len(parts) < 2:
                continue
            vocab.add(normalize_token(parts[0]))

    cache_path.write_text("\n".join(sorted(vocab)) + "\n", encoding="utf-8")
    return vocab


@dataclass(frozen=True)
class GloveVectors:
    vectors: dict[str, "numpy.ndarray"]  # quoted to avoid importing numpy at import-time
    dim: int


def load_glove_vectors(glove_path: Path, *, words: set[str] | None = None) -> GloveVectors:
    """
    Load vectors from a GloVe text file.

    If `words` is provided, only vectors for those words are loaded (case-insensitive).
    """

    import numpy as np

    target = {normalize_token(w) for w in words} if words is not None else None

    vectors: dict[str, np.ndarray] = {}
    dim: int | None = None

    with glove_path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip().split()
            if len(parts) < 3:
                continue
            word = normalize_token(parts[0])
            if target is not None and word not in target:
                continue
            vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            if dim is None:
                dim = int(vec.shape[0])
            vectors[word] = vec

    if dim is None:
        raise ValueError(f"Failed to parse vectors from {glove_path}")

    return GloveVectors(vectors=vectors, dim=dim)

