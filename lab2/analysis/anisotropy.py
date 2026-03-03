from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lab2.analysis.deanisotropy import center_and_norm, mean_center, remove_top_pcs
from lab2.analysis.similarity import cosine_to_mean, l2_normalize_rows
from lab2.utils import ensure_dir, safe_filename, write_json


@dataclass(frozen=True)
class AnisotropyMetrics:
    n_vectors: int
    n_pairs: int
    mean_pairwise_cosine: float
    cos_to_mean_mean: float
    cos_to_mean_std: float
    pca_evr: list[float]
    pca_evr_cum: list[float]


def compute_metrics(vectors: np.ndarray, *, pca_top_k: int) -> tuple[AnisotropyMetrics, np.ndarray]:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    m = int(vectors.shape[0])
    if m < 2:
        raise ValueError("Need at least 2 vectors")

    v_norm = l2_normalize_rows(vectors)
    sim = v_norm @ v_norm.T
    iu = np.triu_indices(m, k=1)
    a1 = float(sim[iu].mean())

    c2m = cosine_to_mean(vectors)

    from sklearn.decomposition import PCA

    k = min(int(pca_top_k), m, int(vectors.shape[1]))
    pca = PCA(n_components=k, svd_solver="full")
    pca.fit(vectors)
    evr = [float(x) for x in pca.explained_variance_ratio_.tolist()]
    evr_cum = [float(x) for x in np.cumsum(pca.explained_variance_ratio_).tolist()]

    metrics = AnisotropyMetrics(
        n_vectors=m,
        n_pairs=m * (m - 1) // 2,
        mean_pairwise_cosine=a1,
        cos_to_mean_mean=float(np.mean(c2m)),
        cos_to_mean_std=float(np.std(c2m)),
        pca_evr=evr,
        pca_evr_cum=evr_cum,
    )
    return metrics, c2m


def _plot_cos2mean(c2m: np.ndarray, *, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)
    plt.figure(figsize=(6, 4))
    plt.hist(c2m, bins=40, color="#4C78A8", alpha=0.9)
    plt.title(title)
    plt.xlabel("cos(v, μ)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pca(evr: list[float], *, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)
    xs = list(range(1, len(evr) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, evr, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_anisotropy(
    *,
    vectors: np.ndarray,
    model_label: str,
    layer_label: str,
    pca_top_k: int,
    out_dir: Path,
) -> dict[str, dict]:
    """
    Compute metrics for the raw vectors and for the de-anisotropisation variants required by the PDF.
    """

    transforms: dict[str, np.ndarray] = {
        "raw": vectors,
        "center": mean_center(vectors),
        "center_norm": center_and_norm(vectors),
        "remove_pcs_1": remove_top_pcs(vectors, m=1),
        "remove_pcs_2": remove_top_pcs(vectors, m=2),
        "remove_pcs_5": remove_top_pcs(vectors, m=5),
    }

    results: dict[str, dict] = {}
    for name, v in transforms.items():
        metrics, c2m = compute_metrics(v, pca_top_k=pca_top_k)
        results[name] = {
            "n_vectors": metrics.n_vectors,
            "n_pairs": metrics.n_pairs,
            "mean_pairwise_cosine": metrics.mean_pairwise_cosine,
            "cos_to_mean_mean": metrics.cos_to_mean_mean,
            "cos_to_mean_std": metrics.cos_to_mean_std,
            "pca_evr": metrics.pca_evr,
            "pca_evr_cum": metrics.pca_evr_cum,
        }

        stem = safe_filename(f"{model_label}_{layer_label}_{name}")
        _plot_cos2mean(
            c2m,
            title=f"{model_label} {layer_label} — {name}: cos(v, μ)",
            out_path=out_dir / f"anisotropy_cos2mean__{stem}.png",
        )
        _plot_pca(
            metrics.pca_evr,
            title=f"{model_label} {layer_label} — {name}: PCA EVR",
            out_path=out_dir / f"anisotropy_pca_evr__{stem}.png",
        )

    return results

