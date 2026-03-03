from __future__ import annotations

import numpy as np


def l2_normalize_rows(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def cosine(u: np.ndarray, v: np.ndarray, *, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < eps:
        return 0.0
    return float(np.dot(u, v) / denom)


def cosine_to_mean(vectors: np.ndarray) -> np.ndarray:
    v = l2_normalize_rows(vectors)
    mu = np.mean(vectors, axis=0)
    mu_norm = mu / max(np.linalg.norm(mu), 1e-12)
    return v @ mu_norm

