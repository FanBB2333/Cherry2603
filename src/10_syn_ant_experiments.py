from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
import random

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
PLOTS_DIR = CACHE_DIR / "plots_syn_ant"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PAIRS_JSON = CACHE_DIR / "wordnet_pairs.json"

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


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def mean_center_matrix(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = u / max(np.linalg.norm(u), 1e-12)
    v = v / max(np.linalg.norm(v), 1e-12)
    return float(np.dot(u, v))


def pair_cosines(vectors: Dict[str, np.ndarray], pairs: List[Tuple[str, str]]) -> np.ndarray:
    vals = []
    for a, b in pairs:
        vals.append(cosine(vectors[a], vectors[b]))
    return np.array(vals, dtype=np.float64)


def save_violin(vals_dict: Dict[str, np.ndarray], title: str, outpath: Path) -> None:
    plt.figure()
    labels = list(vals_dict.keys())
    data = [vals_dict[k] for k in labels]
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.xticks(range(1, len(labels)+1), labels)
    plt.title(title)
    plt.ylabel("cosine similarity")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def build_nn_index(vectors: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    words = sorted(vectors.keys())
    X = np.stack([vectors[w] for w in words], axis=0)
    X = l2_normalize_rows(X)
    return words, X


def topk_neighbors(words: List[str], X: np.ndarray, w: str, k: int = 10) -> List[str]:
    idx = words.index(w)
    sims = X @ X[idx]
    # exclude itself
    sims[idx] = -1.0
    nn_idx = np.argpartition(-sims, kth=k)[:k]
    nn_idx = nn_idx[np.argsort(-sims[nn_idx])]
    return [words[i] for i in nn_idx]


def invert_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    m: Dict[str, Set[str]] = {}
    for a, b in pairs:
        m.setdefault(a, set()).add(b)
        m.setdefault(b, set()).add(a)
    return m


def neighbourhood_eval(vectors: Dict[str, np.ndarray], syn_map: Dict[str, Set[str]], ant_map: Dict[str, Set[str]],
                       targets: List[str], k: int = 10) -> Dict[str, float]:
    words, X = build_nn_index(vectors)

    eligible = 0
    hit = 0
    for w in targets:
        if w not in words:
            continue
        # "when a synonym of the target is present in the vocabulary"
        # => at least one synonym exists AND is in vocab
        syns = syn_map.get(w, set())
        if not any(s in vectors for s in syns):
            continue
        eligible += 1

        nns = topk_neighbors(words, X, w, k=k)
        ants = ant_map.get(w, set())
        if any(a in nns for a in ants):
            hit += 1

    rate = hit / eligible if eligible > 0 else float("nan")
    return {"eligible": eligible, "hit": hit, "rate": rate}


def main():
    pairs = json.loads(Path(PAIRS_JSON).read_text())
    syn_pairs = [tuple(p) for p in pairs["syn_200"]]
    ant_pairs = [tuple(p) for p in pairs["ant_200"]]
    rand_pairs = [tuple(p) for p in pairs["rand_200"]]

    syn_map = invert_pairs(syn_pairs)
    ant_map = invert_pairs(ant_pairs)

    results = {}

    for name, path in VECTORS.items():
        vecs = load_npz(path)

        # (7) similarity separation - RAW
        syn_vals = pair_cosines(vecs, syn_pairs)
        ant_vals = pair_cosines(vecs, ant_pairs)
        rnd_vals = pair_cosines(vecs, rand_pairs)
        results.setdefault(name, {})["similarity_separation_raw"] = {
            "syn_mean": float(syn_vals.mean()),
            "ant_mean": float(ant_vals.mean()),
            "rnd_mean": float(rnd_vals.mean()),
        }
        save_violin(
            {"syn": syn_vals, "ant": ant_vals, "random": rnd_vals},
            f"{name} RAW: cosine distributions",
            PLOTS_DIR / f"simsep_{name}_raw_violin.png"
        )

        # choose 50 targets from vocab (fixed seed)
        vocab_list = sorted(vecs.keys())
        targets = rng.sample(vocab_list, 50)

        # (8) neighbourhood evaluation - RAW
        nn_raw = neighbourhood_eval(vecs, syn_map, ant_map, targets, k=10)
        results[name]["neighbourhood_raw"] = nn_raw

        # mean-centering de-anisotropisation
        # For neighbourhood we need vectors dict again
        X = np.stack([vecs[w] for w in vecs.keys()], axis=0)
        mu = X.mean(axis=0)
        vecs_mc = {w: (v - mu) for w, v in vecs.items()}

        # (7) similarity separation - mean_center
        syn_vals_mc = pair_cosines(vecs_mc, syn_pairs)
        ant_vals_mc = pair_cosines(vecs_mc, ant_pairs)
        rnd_vals_mc = pair_cosines(vecs_mc, rand_pairs)
        results[name]["similarity_separation_mean_center"] = {
            "syn_mean": float(syn_vals_mc.mean()),
            "ant_mean": float(ant_vals_mc.mean()),
            "rnd_mean": float(rnd_vals_mc.mean()),
        }
        save_violin(
            {"syn": syn_vals_mc, "ant": ant_vals_mc, "random": rnd_vals_mc},
            f"{name} mean_center: cosine distributions",
            PLOTS_DIR / f"simsep_{name}_meancenter_violin.png"
        )

        # (8) neighbourhood evaluation - mean_center
        nn_mc = neighbourhood_eval(vecs_mc, syn_map, ant_map, targets, k=10)
        results[name]["neighbourhood_mean_center"] = nn_mc

    out = CACHE_DIR / "syn_ant_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("Saved:", out)
    print("Plots in:", PLOTS_DIR)

    # print compact summary
    for name, r in results.items():
        sraw = r["similarity_separation_raw"]
        smc = r["similarity_separation_mean_center"]
        nraw = r["neighbourhood_raw"]
        nmc = r["neighbourhood_mean_center"]
        print(f"\n== {name} ==")
        print("simsep raw     syn/ant/rnd means:", round(sraw["syn_mean"],4), round(sraw["ant_mean"],4), round(sraw["rnd_mean"],4))
        print("simsep meanctr syn/ant/rnd means:", round(smc["syn_mean"],4), round(smc["ant_mean"],4), round(smc["rnd_mean"],4))
        print("neigh raw      eligible/hit/rate:", nraw["eligible"], nraw["hit"], round(nraw["rate"],4))
        print("neigh meanctr  eligible/hit/rate:", nmc["eligible"], nmc["hit"], round(nmc["rate"],4))

if __name__ == "__main__":
    main()
