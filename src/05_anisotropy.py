from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import json

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
PLOTS_DIR = CACHE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Cached vectors you already built
GLOVE_NPZ = CACHE_DIR / "glove_6B_300d_targetwords.npz"
BERT_NPZ = {
    "bert_l1": CACHE_DIR / "bert_contextual_layer1.npz",
    "bert_l6": CACHE_DIR / "bert_contextual_layer6.npz",
    "bert_l12": CACHE_DIR / "bert_contextual_layer12.npz",
}

OUT_JSON = CACHE_DIR / "anisotropy_results.json"


def load_npz_matrix(path: Path) -> Tuple[np.ndarray, List[str]]:
    z = np.load(path)
    words = sorted(z.files)
    X = np.stack([z[w] for w in words], axis=0).astype(np.float64)  # (M, D)
    return X, words


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    # assumes rows are already normalized
    return X @ X.T


def metric_A1_mean_pairwise_cos(X: np.ndarray) -> float:
    # normalize first
    Xn = l2_normalize_rows(X)
    C = cosine_matrix(Xn)
    M = C.shape[0]
    # average over i<j
    upper = np.triu_indices(M, k=1)
    return float(C[upper].mean())


def metric_A2_cos_to_mean(X: np.ndarray) -> Dict[str, float]:
    # Compute mu from raw vectors, then cosine(v_i, mu)
    mu = X.mean(axis=0)
    mu_norm = np.linalg.norm(mu)
    if mu_norm == 0:
        cos_vals = np.zeros(X.shape[0], dtype=np.float64)
    else:
        Xn = l2_normalize_rows(X)
        mu_u = mu / mu_norm
        cos_vals = Xn @ mu_u
    return {
        "mean": float(cos_vals.mean()),
        "std": float(cos_vals.std(ddof=0)),
        "min": float(cos_vals.min()),
        "max": float(cos_vals.max()),
    }


def metric_A3_pca(X: np.ndarray, k: int = 10) -> Dict[str, object]:
    # PCA on centered data is standard; sklearn centers by default
    pca = PCA(n_components=min(k, X.shape[1], X.shape[0]))
    pca.fit(X)
    evr = pca.explained_variance_ratio_.astype(np.float64)
    cum = np.cumsum(evr)
    return {
        "k": int(len(evr)),
        "evr": [float(x) for x in evr],
        "cum_evr": [float(x) for x in cum],
    }


def remove_top_pcs(X: np.ndarray, m: int) -> np.ndarray:
    # Fit PCA directions u_j on X (sklearn centers internally)
    pca = PCA(n_components=m)
    pca.fit(X)
    U = pca.components_  # (m, D), rows are u_j
    # Remove projections: v - sum_j (u_j^T v) u_j
    # If U has shape (m, D), projection = (X @ U.T) @ U
    return X - (X @ U.T) @ U


def compute_all_metrics(X: np.ndarray) -> Dict[str, object]:
    M = X.shape[0]
    return {
        "M": int(M),
        "num_pairs_for_A1": int(M * (M - 1) // 2),
        "A1_mean_pairwise_cos": metric_A1_mean_pairwise_cos(X),
        "A2_cos_to_mean": metric_A2_cos_to_mean(X),
        "A3_pca": metric_A3_pca(X, k=10),
    }


def save_A2_histogram(X: np.ndarray, title: str, filename: Path) -> None:
    mu = X.mean(axis=0)
    mu_norm = np.linalg.norm(mu)
    Xn = l2_normalize_rows(X)
    if mu_norm == 0:
        cos_vals = np.zeros(X.shape[0], dtype=np.float64)
    else:
        cos_vals = Xn @ (mu / mu_norm)

    plt.figure()
    plt.hist(cos_vals, bins=50)
    plt.title(title)
    plt.xlabel("cos(v_i, mu)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Load all models
    models: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    Xg, words_g = load_npz_matrix(GLOVE_NPZ)
    models["glove"] = (Xg, words_g)

    for name, p in BERT_NPZ.items():
        Xb, words_b = load_npz_matrix(p)
        models[name] = (Xb, words_b)

    results: Dict[str, object] = {}

    for model_name, (X, words) in models.items():
        # Ensure same word set across models (you constructed all from same target list)
        results[model_name] = {}

        # RAW
        results[model_name]["raw"] = compute_all_metrics(X)
        save_A2_histogram(X, f"{model_name} RAW: cos(v, mu)", PLOTS_DIR / f"A2_{model_name}_raw.png")

        # Mean centering: v <- v - mu
        X_center = X - X.mean(axis=0, keepdims=True)
        results[model_name]["mean_center"] = compute_all_metrics(X_center)
        save_A2_histogram(X_center, f"{model_name} mean-center: cos(v, mu)", PLOTS_DIR / f"A2_{model_name}_meancenter.png")

        # Mean centering + normalization by Euclidean norm
        X_center_norm = l2_normalize_rows(X_center)
        results[model_name]["mean_center_norm"] = compute_all_metrics(X_center_norm)
        save_A2_histogram(X_center_norm, f"{model_name} mean-center+norm: cos(v, mu)", PLOTS_DIR / f"A2_{model_name}_meancenter_norm.png")

        # Remove top PCs for m in {1,2,5}
        for m in [1, 2, 5]:
            X_rmpc = remove_top_pcs(X, m=m)
            key = f"remove_top_pcs_m{m}"
            results[model_name][key] = compute_all_metrics(X_rmpc)
            save_A2_histogram(X_rmpc, f"{model_name} remove top PCs (m={m}): cos(v, mu)", PLOTS_DIR / f"A2_{model_name}_rmpc{m}.png")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print("Saved results JSON:", OUT_JSON)
    print("Saved plots in:", PLOTS_DIR)


if __name__ == "__main__":
    main()
