from __future__ import annotations

from pathlib import Path

from lab2.utils import ensure_dir


def save_word_embeddings_npz(path: Path, *, vectors_by_word: dict[str, "numpy.ndarray"]) -> None:
    """
    Save embeddings as a single compressed NPZ file:
    - words: (N,) array of strings
    - vectors: (N, D) float32 matrix
    """

    import numpy as np

    ensure_dir(path.parent)
    words = sorted(vectors_by_word.keys())
    if not words:
        raise ValueError("No vectors to save.")

    matrix = np.stack([vectors_by_word[w] for w in words]).astype("float32", copy=False)
    np.savez_compressed(path, words=np.asarray(words), vectors=matrix)


def load_word_embeddings_npz(path: Path) -> dict[str, "numpy.ndarray"]:
    import numpy as np

    data = np.load(path, allow_pickle=False)
    words = [str(w) for w in data["words"].tolist()]
    vectors = data["vectors"]
    return {w: vectors[i] for i, w in enumerate(words)}

