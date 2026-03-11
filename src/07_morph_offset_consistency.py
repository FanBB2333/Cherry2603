from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
PLOTS_DIR = CACHE_DIR / "plots_morph"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MORPH_TSV = PROJECT_ROOT / "morph_families.tsv"

VECTORS = {
    "glove": CACHE_DIR / "glove_6B_300d_targetwords.npz",
    "bert_l1": CACHE_DIR / "bert_contextual_layer1.npz",
    "bert_l6": CACHE_DIR / "bert_contextual_layer6.npz",
    "bert_l12": CACHE_DIR / "bert_contextual_layer12.npz",
}

SEED = 42
np.random.seed(SEED)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


def l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / max(np.linalg.norm(v), eps)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(l2norm(u), l2norm(v)))


def mean_center(vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    X = np.stack(list(vectors.values()), axis=0)
    mu = X.mean(axis=0, keepdims=True)
    return {w: (vec - mu[0]) for w, vec in vectors.items()}


def parse_families(path: Path) -> pd.DataFrame:
    """
    Returns dataframe with columns:
      lemma, forms(list), family_type, transform
    forms are stored as python list in a column.
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    df.columns = ["lemma", "forms", "family_type", "transform"]
    df["lemma"] = df["lemma"].str.strip().str.lower()
    df["transform"] = df["transform"].str.strip()
    df["forms_list"] = df["forms"].apply(lambda s: [x.strip().lower() for x in s.split(",") if x.strip()])
    return df


def bootstrap_ci(values: np.ndarray, B: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    n = len(values)
    stats = []
    for _ in range(B):
        samp = np.random.choice(values, size=n, replace=True)
        stats.append(float(np.mean(samp)))
    stats = np.sort(stats)
    lo = stats[int((alpha/2) * B)]
    hi = stats[int((1 - alpha/2) * B) - 1]
    return float(lo), float(hi)


def offset_consistency_for_transform(df: pd.DataFrame, vectors: Dict[str, np.ndarray], transform: str) -> Dict[str, object]:
    """
    For a transform label (e.g., 'verb_regular'), build offsets (form - lemma) for each family row
    and compute pairwise cosine similarities among offsets.
    """
    sub = df[df["transform"] == transform].copy()
    pairs = []  # list of (lemma, form)
    for _, row in sub.iterrows():
        lemma = row["lemma"]
        for form in row["forms_list"]:
            pairs.append((lemma, form))

    # Keep only pairs where both words exist in embedding vocab
    offsets = []
    kept_pairs = []
    for lemma, form in pairs:
        if lemma in vectors and form in vectors:
            offsets.append(vectors[form] - vectors[lemma])
            kept_pairs.append((lemma, form))

    if len(offsets) < 2:
        return {
            "transform": transform,
            "num_pairs": int(len(offsets)),
            "num_offset_pairs": 0,
            "mean_cos": None,
            "ci": None,
        }

    offsets = np.stack(offsets, axis=0)  # (N, D)

    # Pairwise cosine among offsets
    cos_vals = []
    for i, j in itertools.combinations(range(offsets.shape[0]), 2):
        cos_vals.append(cosine(offsets[i], offsets[j]))
    cos_vals = np.array(cos_vals, dtype=np.float64)

    ci = bootstrap_ci(cos_vals)

    return {
        "transform": transform,
        "num_pairs": int(len(offsets)),
        "num_offset_pairs": int(len(cos_vals)),
        "mean_cos": float(np.mean(cos_vals)),
        "ci": ci,
    }


def main():
    df = parse_families(MORPH_TSV)
    transforms = sorted(df["transform"].unique().tolist())
    print("Transforms in TSV:", transforms)

    results = {}

    for model_name, path in VECTORS.items():
        vecs_raw = load_npz(path)
        vecs_mc = mean_center(vecs_raw)

        results[model_name] = {"raw": {}, "mean_center": {}}

        for t in transforms:
            results[model_name]["raw"][t] = offset_consistency_for_transform(df, vecs_raw, t)
            results[model_name]["mean_center"][t] = offset_consistency_for_transform(df, vecs_mc, t)

    out = CACHE_DIR / "morph_offset_consistency.json"
    out.write_text(json.dumps(results, indent=2))
    print("Saved:", out)

    # Simple plot: mean offset cosine per model for each transform (raw)
    for t in transforms:
        plt.figure()
        xs, ys = [], []
        for model_name in VECTORS.keys():
            r = results[model_name]["raw"][t]
            if r["mean_cos"] is None:
                continue
            xs.append(model_name)
            ys.append(r["mean_cos"])
        plt.bar(xs, ys)
        plt.title(f"Offset consistency (RAW) for transform={t}")
        plt.ylabel("mean cosine between offsets")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"offset_consistency_raw_{t}.png")
        plt.close()

    print("Plots saved in:", PLOTS_DIR)


if __name__ == "__main__":
    main()
