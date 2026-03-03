from __future__ import annotations

import numpy as np


def mean_center(vectors: np.ndarray) -> np.ndarray:
    return vectors - np.mean(vectors, axis=0, keepdims=True)


def center_and_norm(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    centered = mean_center(vectors)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, eps)


def remove_top_pcs(vectors: np.ndarray, *, m: int) -> np.ndarray:
    """
    Remove the projection onto the top-m principal components (after mean-centering).
    """

    if m <= 0:
        raise ValueError("m must be positive")

    centered = mean_center(vectors)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=m, svd_solver="full")
    pca.fit(centered)
    comps = pca.components_  # (m, D), orthonormal
    proj = centered @ comps.T @ comps
    return centered - proj


def deaniso(vectors: np.ndarray, *, method: str, m: int = 2) -> np.ndarray:
    method = method.lower()
    if method == "center":
        return mean_center(vectors)
    if method == "center_norm":
        return center_and_norm(vectors)
    if method == "remove_pcs":
        return remove_top_pcs(vectors, m=m)
    raise ValueError(f"Unknown deaniso method: {method}")

