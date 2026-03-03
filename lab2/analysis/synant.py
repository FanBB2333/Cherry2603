from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lab2.analysis.deanisotropy import deaniso
from lab2.analysis.similarity import cosine, l2_normalize_rows
from lab2.stats import BootstrapCI, bootstrap_ci
from lab2.utils import ensure_dir, safe_filename


def read_pairs_tsv(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("word1"):
            continue
        a, b = line.split("\t")
        a = a.strip().lower()
        b = b.strip().lower()
        if a and b and a != b:
            pairs.append((a, b) if a <= b else (b, a))
    # de-dup while keeping deterministic order
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def cosine_for_pairs(pairs: list[tuple[str, str]], embeddings: dict[str, np.ndarray]) -> list[float]:
    vals: list[float] = []
    for a, b in pairs:
        if a not in embeddings or b not in embeddings:
            continue
        vals.append(cosine(embeddings[a], embeddings[b]))
    return vals


def sample_random_pairs(
    *,
    words: list[str],
    n_pairs: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    attempts = 0
    max_attempts = n_pairs * 50
    while len(out) < n_pairs and attempts < max_attempts:
        attempts += 1
        a = rng.choice(words)
        b = rng.choice(words)
        if a == b:
            continue
        p = (a, b) if a <= b else (b, a)
        if p in pairs:
            continue
        pairs.add(p)
        out.append(p)
    return out


def _plot_similarity_distributions(
    *,
    syn: list[float],
    ant: list[float],
    rnd: list[float],
    title: str,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            [{"type": "synonym", "cos": v} for v in syn]
            + [{"type": "antonym", "cos": v} for v in ant]
            + [{"type": "random", "cos": v} for v in rnd]
        )
        plt.figure(figsize=(7, 4))
        sns.violinplot(data=df, x="type", y="cos", inner="quartile", cut=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except ModuleNotFoundError:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.boxplot([syn, ant, rnd], labels=["synonym", "antonym", "random"])
        plt.title(title)
        plt.ylabel("Cosine similarity")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()


@dataclass(frozen=True)
class NeighbourhoodResult:
    n_targets: int
    rate_antonym_in_topk: float


def neighbourhood_eval(
    *,
    embeddings: dict[str, np.ndarray],
    synonym_pairs: list[tuple[str, str]],
    antonym_pairs: list[tuple[str, str]],
    k: int,
    seed: int,
    n_targets: int = 50,
) -> NeighbourhoodResult:
    # Build maps
    syn_map: dict[str, set[str]] = {}
    ant_map: dict[str, set[str]] = {}
    for a, b in synonym_pairs:
        syn_map.setdefault(a, set()).add(b)
        syn_map.setdefault(b, set()).add(a)
    for a, b in antonym_pairs:
        ant_map.setdefault(a, set()).add(b)
        ant_map.setdefault(b, set()).add(a)

    vocab = sorted(embeddings.keys())
    vocab_set = set(vocab)

    candidates = [w for w in vocab if (w in syn_map) and (w in ant_map)]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    targets = candidates[:n_targets]

    if not targets:
        return NeighbourhoodResult(n_targets=0, rate_antonym_in_topk=float("nan"))

    mat = np.stack([embeddings[w] for w in vocab])
    mat_norm = l2_normalize_rows(mat)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    hits = 0
    denom = 0
    for t in targets:
        syns = {s for s in syn_map.get(t, set()) if s in vocab_set}
        ants = {a for a in ant_map.get(t, set()) if a in vocab_set}
        if not syns or not ants:
            continue

        denom += 1
        i = word_to_idx[t]
        sims = mat_norm @ mat_norm[i]
        sims[i] = -1.0  # exclude self
        nn_idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        nn_words = {vocab[j] for j in nn_idx}
        if nn_words.intersection(ants):
            hits += 1

    rate = hits / denom if denom else float("nan")
    return NeighbourhoodResult(n_targets=int(denom), rate_antonym_in_topk=float(rate))


def run_synant_suite(
    *,
    embeddings: dict[str, np.ndarray],
    synonym_pairs: list[tuple[str, str]],
    antonym_pairs: list[tuple[str, str]],
    k: int,
    seed: int,
    deaniso_method: str,
    deaniso_pcs: int,
    figures_dir: Path,
    label: str,
) -> dict:
    """
    Implements PDF §4:
    (i) similarity separation (syn vs ant vs random)
    (ii) neighbourhood evaluation (antonym among top-k neighbours)
    Both before and after de-anisotropisation.
    """

    ensure_dir(figures_dir)

    vocab_words = sorted(embeddings.keys())
    # Filter pairs to those in vocab
    syn_pairs = [(a, b) for a, b in synonym_pairs if a in embeddings and b in embeddings]
    ant_pairs = [(a, b) for a, b in antonym_pairs if a in embeddings and b in embeddings]
    rnd_pairs = sample_random_pairs(words=vocab_words, n_pairs=min(len(syn_pairs), len(ant_pairs)), seed=seed)

    out: dict = {"label": label, "raw": {}, "deaniso": {}}

    # Raw similarity distributions
    syn_sims = cosine_for_pairs(syn_pairs, embeddings)
    ant_sims = cosine_for_pairs(ant_pairs, embeddings)
    rnd_sims = cosine_for_pairs(rnd_pairs, embeddings)
    _plot_similarity_distributions(
        syn=syn_sims,
        ant=ant_sims,
        rnd=rnd_sims,
        title=f"{label} — syn/ant/random (raw)",
        out_path=figures_dir / f"synant_similarity__{safe_filename(label + '_raw')}.png",
    )
    out["raw"]["similarities"] = {
        "n_syn_pairs": len(syn_sims),
        "n_ant_pairs": len(ant_sims),
        "n_random_pairs": len(rnd_sims),
        "syn_mean_ci": bootstrap_ci(syn_sims, seed=seed).__dict__ if syn_sims else None,
        "ant_mean_ci": bootstrap_ci(ant_sims, seed=seed + 1).__dict__ if ant_sims else None,
        "rnd_mean_ci": bootstrap_ci(rnd_sims, seed=seed + 2).__dict__ if rnd_sims else None,
    }

    # Raw neighbourhood
    neigh_raw = neighbourhood_eval(
        embeddings=embeddings,
        synonym_pairs=syn_pairs,
        antonym_pairs=ant_pairs,
        k=k,
        seed=seed,
    )
    out["raw"]["neighbourhood"] = neigh_raw.__dict__

    # De-anisotropised
    words = sorted(embeddings.keys())
    mat = np.stack([embeddings[w] for w in words])
    mat_d = deaniso(mat, method=deaniso_method, m=deaniso_pcs)
    emb_d = {w: mat_d[i] for i, w in enumerate(words)}

    syn_sims_d = cosine_for_pairs(syn_pairs, emb_d)
    ant_sims_d = cosine_for_pairs(ant_pairs, emb_d)
    rnd_sims_d = cosine_for_pairs(rnd_pairs, emb_d)
    _plot_similarity_distributions(
        syn=syn_sims_d,
        ant=ant_sims_d,
        rnd=rnd_sims_d,
        title=f"{label} — syn/ant/random ({deaniso_method})",
        out_path=figures_dir / f"synant_similarity__{safe_filename(label + '_deaniso')}.png",
    )
    out["deaniso"]["similarities"] = {
        "method": deaniso_method,
        "pcs": deaniso_pcs,
        "n_syn_pairs": len(syn_sims_d),
        "n_ant_pairs": len(ant_sims_d),
        "n_random_pairs": len(rnd_sims_d),
        "syn_mean_ci": bootstrap_ci(syn_sims_d, seed=seed).__dict__ if syn_sims_d else None,
        "ant_mean_ci": bootstrap_ci(ant_sims_d, seed=seed + 1).__dict__ if ant_sims_d else None,
        "rnd_mean_ci": bootstrap_ci(rnd_sims_d, seed=seed + 2).__dict__ if rnd_sims_d else None,
    }

    neigh_d = neighbourhood_eval(
        embeddings=emb_d,
        synonym_pairs=syn_pairs,
        antonym_pairs=ant_pairs,
        k=k,
        seed=seed,
    )
    out["deaniso"]["neighbourhood"] = {
        "method": deaniso_method,
        "pcs": deaniso_pcs,
        **neigh_d.__dict__,
    }

    return out

