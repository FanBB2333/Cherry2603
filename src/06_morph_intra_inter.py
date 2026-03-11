from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import itertools
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


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
rng = random.Random(SEED)


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


def parse_families(path: Path) -> List[List[str]]:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    families = []
    for _, row in df.iterrows():
        lemma = row[0].strip().lower()
        forms = [x.strip().lower() for x in row[1].split(",") if x.strip()]
        fam = []
        for w in [lemma] + forms:
            if w and w not in fam:
                fam.append(w)
        families.append(fam)
    return families


def all_intra_pairs(family: List[str]) -> List[Tuple[str, str]]:
    return list(itertools.combinations(family, 2))


def build_inter_pairs_matched(
    families: List[List[str]],
    n_pairs: int,
) -> List[Tuple[str, str]]:
    """
    Sample random pairs from different families.
    We create pairs by:
      - sampling two distinct families
      - sampling one word from each
    """
    pairs = []
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 50:
        tries += 1
        f1, f2 = rng.sample(families, 2)
        w1 = rng.choice(f1)
        w2 = rng.choice(f2)
        if w1 == w2:
            continue
        pairs.append((w1, w2))
    return pairs


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # O(n*m), but our sizes are small; fine.
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


def bootstrap_ci(stat_fn, x: np.ndarray, y: np.ndarray, B: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    vals = []
    n = len(x)
    m = len(y)
    for _ in range(B):
        xb = np.random.choice(x, size=n, replace=True)
        yb = np.random.choice(y, size=m, replace=True)
        vals.append(stat_fn(xb, yb))
    vals = np.sort(vals)
    lo = vals[int((alpha/2) * B)]
    hi = vals[int((1 - alpha/2) * B) - 1]
    return float(lo), float(hi)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if sp == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / sp)


def run_one(vectors: Dict[str, np.ndarray], label: str, out_prefix: str) -> Dict[str, object]:
    families = parse_families(MORPH_TSV)

    # keep only words that exist in this embedding vocabulary
    families = [[w for w in fam if w in vectors] for fam in families]
    families = [fam for fam in families if len(fam) >= 2]

    intra_pairs = []
    for fam in families:
        intra_pairs.extend(all_intra_pairs(fam))

    intra_sims = np.array([cosine(vectors[a], vectors[b]) for a, b in intra_pairs], dtype=np.float64)

    inter_pairs = build_inter_pairs_matched(families, n_pairs=len(intra_pairs))
    inter_sims = np.array([cosine(vectors[a], vectors[b]) for a, b in inter_pairs], dtype=np.float64)

    # Effect size + CI (choose one; we compute both and you can decide)
    d = cohens_d(intra_sims, inter_sims)
    d_ci = bootstrap_ci(lambda x,y: cohens_d(x,y), intra_sims, inter_sims)

    delta = cliffs_delta(intra_sims, inter_sims)
    delta_ci = bootstrap_ci(lambda x,y: cliffs_delta(x,y), intra_sims, inter_sims)

    # Mann–Whitney (non-parametric)
    # alternative="greater": intra expected > inter
    mw = mannwhitneyu(intra_sims, inter_sims, alternative="greater")
    pval = float(mw.pvalue)

    # Plot
    plt.figure()
    plt.violinplot([intra_sims, inter_sims], showmeans=True)
    plt.xticks([1, 2], ["intra", "inter"])
    plt.title(f"{label}: intra vs inter cosine")
    plt.ylabel("cosine similarity")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{out_prefix}_violin.png")
    plt.close()

    return {
        "label": label,
        "num_families_used": int(len(families)),
        "num_intra_pairs": int(len(intra_pairs)),
        "num_inter_pairs": int(len(inter_pairs)),
        "intra_mean": float(np.mean(intra_sims)),
        "inter_mean": float(np.mean(inter_sims)),
        "cohens_d": d,
        "cohens_d_ci": d_ci,
        "cliffs_delta": delta,
        "cliffs_delta_ci": delta_ci,
        "mannwhitney_p_greater": pval,
    }


def main():
    results = {}

    for name, path in VECTORS.items():
        vecs_raw = load_npz(path)
        vecs_mc = mean_center(vecs_raw)

        results[name] = {
            "raw": run_one(vecs_raw, f"{name} RAW", f"{name}_raw"),
            "mean_center": run_one(vecs_mc, f"{name} mean_center", f"{name}_mc"),
        }

    out = CACHE_DIR / "morph_intra_inter_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("Saved:", out)
    print("Plots saved in:", PLOTS_DIR)


if __name__ == "__main__":
    main()
