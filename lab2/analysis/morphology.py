from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lab2.analysis.deanisotropy import deaniso
from lab2.analysis.similarity import cosine, l2_normalize_rows
from lab2.data.morph_families import MorphFamily
from lab2.stats import BootstrapCI, bootstrap_ci, bootstrap_ci_2samp, cliffs_delta, cohen_d, mann_whitney_u
from lab2.utils import ensure_dir, safe_filename


def _filter_families(families: list[MorphFamily], *, vocab: set[str]) -> list[list[str]]:
    filtered: list[list[str]] = []
    for fam in families:
        forms = [f.lower() for f in fam.forms]
        forms = [f for f in forms if f in vocab]
        forms = sorted(set(forms))
        if len(forms) >= 2:
            filtered.append(forms)
    return filtered


def intra_family_similarities(families_forms: list[list[str]], embeddings: dict[str, np.ndarray]) -> list[float]:
    sims: list[float] = []
    for forms in families_forms:
        for a, b in itertools.combinations(forms, 2):
            sims.append(cosine(embeddings[a], embeddings[b]))
    return sims


def inter_family_similarities(
    families_forms: list[list[str]],
    embeddings: dict[str, np.ndarray],
    *,
    n_pairs: int,
    seed: int,
) -> list[float]:
    """
    Sample random pairs of words drawn from different families (matched count).
    """

    rng = random.Random(seed)
    word_to_family: dict[str, int] = {}
    words: list[str] = []
    for i, forms in enumerate(families_forms):
        for w in forms:
            if w in word_to_family:
                continue
            word_to_family[w] = i
            words.append(w)

    sims: list[float] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = n_pairs * 50
    while len(sims) < n_pairs and attempts < max_attempts:
        attempts += 1
        a = rng.choice(words)
        b = rng.choice(words)
        if a == b:
            continue
        if word_to_family.get(a) == word_to_family.get(b):
            continue
        pair = (a, b) if a <= b else (b, a)
        if pair in seen:
            continue
        seen.add(pair)
        sims.append(cosine(embeddings[a], embeddings[b]))

    return sims


@dataclass(frozen=True)
class IntraInterResult:
    n_intra: int
    n_inter: int
    cliffs_delta: BootstrapCI
    cohen_d: BootstrapCI
    p_mannwhitney: float


def run_intra_vs_inter(
    *,
    families: list[MorphFamily],
    embeddings: dict[str, np.ndarray],
    seed: int,
    n_boot: int = 2000,
) -> tuple[IntraInterResult, dict[str, list[float]]]:
    vocab = set(embeddings.keys())
    fam_forms = _filter_families(families, vocab=vocab)
    intra = intra_family_similarities(fam_forms, embeddings)
    inter = inter_family_similarities(fam_forms, embeddings, n_pairs=len(intra), seed=seed)

    cd = bootstrap_ci_2samp(intra, inter, statistic=cliffs_delta, n_boot=n_boot, seed=seed)
    d = bootstrap_ci_2samp(intra, inter, statistic=cohen_d, n_boot=n_boot, seed=seed + 1)
    p = mann_whitney_u(intra, inter)

    return (
        IntraInterResult(
            n_intra=len(intra),
            n_inter=len(inter),
            cliffs_delta=cd,
            cohen_d=d,
            p_mannwhitney=p,
        ),
        {"intra": intra, "inter": inter},
    )


def _plot_intra_inter(dist: dict[str, list[float]], *, title: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            [{"group": "intra", "cos": v} for v in dist["intra"]]
            + [{"group": "inter", "cos": v} for v in dist["inter"]]
        )
        plt.figure(figsize=(6, 4))
        sns.violinplot(data=df, x="group", y="cos", inner="quartile", cut=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except ModuleNotFoundError:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.boxplot([dist["intra"], dist["inter"]], labels=["intra", "inter"])
        plt.title(title)
        plt.ylabel("Cosine similarity")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()


def extract_suffix_pairs(families: list[MorphFamily], *, suffix: str) -> list[tuple[str, str]]:
    """
    Extract (base, derived) pairs found *within the same morphological family* via simple suffix rules.
    """

    pairs: set[tuple[str, str]] = set()
    for fam in families:
        forms = {f.lower() for f in fam.forms}
        for base in list(forms):
            base = base.lower()

            cand = None
            if suffix == "ed":
                cand = base + "ed"
                if cand not in forms and base.endswith("e"):
                    cand = base + "d"
            elif suffix == "ing":
                cand = base + "ing"
                if cand not in forms and base.endswith("e") and len(base) > 1:
                    cand = base[:-1] + "ing"
            elif suffix == "s":
                cand = base + "s"
            else:
                cand = base + suffix

            if cand in forms:
                a, b = (base, cand)
                pairs.add((a, b))

    return sorted(pairs)


def offset_consistency(offsets: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity between offset vectors.
    """

    if offsets.shape[0] < 2:
        return float("nan")
    o_norm = l2_normalize_rows(offsets)
    sim = o_norm @ o_norm.T
    iu = np.triu_indices(sim.shape[0], k=1)
    return float(sim[iu].mean())


def bootstrap_offset_consistency(
    offsets: np.ndarray, *, n_boot: int = 2000, ci: float = 0.95, seed: int = 42
) -> BootstrapCI:
    vals = offsets.tolist()  # makes bootstrap sampling cheaper to implement

    def stat(sample: list[list[float]]) -> float:
        arr = np.asarray(sample, dtype="float32")
        return offset_consistency(arr)

    return bootstrap_ci(vals, statistic=stat, n_boot=n_boot, ci=ci, seed=seed)


@dataclass(frozen=True)
class OffsetResult:
    n_offsets: int
    mean_cos: BootstrapCI


def run_offset_consistency(
    *,
    families: list[MorphFamily],
    embeddings: dict[str, np.ndarray],
    suffix: str,
    seed: int,
    n_boot: int = 2000,
) -> OffsetResult | None:
    pairs = extract_suffix_pairs(families, suffix=suffix)
    usable = [(a, b) for a, b in pairs if a in embeddings and b in embeddings]
    if len(usable) < 3:
        return None

    offsets = np.stack([embeddings[b] - embeddings[a] for a, b in usable])
    ci = bootstrap_offset_consistency(offsets, n_boot=n_boot, seed=seed)
    return OffsetResult(n_offsets=int(offsets.shape[0]), mean_cos=ci)


@dataclass(frozen=True)
class ProbingResult:
    n_examples: int
    accuracy: BootstrapCI
    macro_f1: BootstrapCI


def _tense_dataset_from_ed(families: list[MorphFamily], embeddings: dict[str, np.ndarray]):
    pairs = extract_suffix_pairs(families, suffix="ed")
    x: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []

    for base, past in pairs:
        if base not in embeddings or past not in embeddings:
            continue
        x.append(embeddings[base])
        y.append(0)
        groups.append(base)
        x.append(embeddings[past])
        y.append(1)
        groups.append(base)

    if not x:
        return None

    return np.stack(x), np.asarray(y), np.asarray(groups)


def run_tense_probing(
    *,
    families: list[MorphFamily],
    embeddings: dict[str, np.ndarray],
    seed: int,
    n_boot: int = 2000,
) -> ProbingResult | None:
    data = _tense_dataset_from_ed(families, embeddings)
    if data is None:
        return None
    x, y, groups = data

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = cv.split(x, y, groups=groups)
    except Exception:
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = cv.split(x, y)

    accs: list[float] = []
    f1s: list[float] = []
    for tr, te in splits:
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
        clf.fit(x[tr], y[tr])
        pred = clf.predict(x[te])
        accs.append(float(accuracy_score(y[te], pred)))
        f1s.append(float(f1_score(y[te], pred, average="macro")))

    acc_ci = bootstrap_ci(accs, statistic=lambda a: sum(a) / len(a), n_boot=n_boot, seed=seed)
    f1_ci = bootstrap_ci(f1s, statistic=lambda a: sum(a) / len(a), n_boot=n_boot, seed=seed + 1)
    return ProbingResult(n_examples=int(x.shape[0]), accuracy=acc_ci, macro_f1=f1_ci)


def run_morphology_suite(
    *,
    families: list[MorphFamily],
    embeddings: dict[str, np.ndarray],
    seed: int,
    deaniso_method: str,
    deaniso_pcs: int,
    figures_dir: Path,
    label: str,
) -> dict:
    """
    Implements PDF §3:
    (i) intra vs inter family similarities
    (ii) offset consistency (+ed, +ing, plural +s)
    (iii) probing (tense: past vs present, using +ed)
    All reported for raw and de-anisotropised embeddings.
    """

    ensure_dir(figures_dir)
    out: dict = {"label": label, "raw": {}, "deaniso": {}}

    # (i) intra vs inter
    res_raw, dist_raw = run_intra_vs_inter(families=families, embeddings=embeddings, seed=seed)
    _plot_intra_inter(
        dist_raw,
        title=f"{label} — intra vs inter (raw)",
        out_path=figures_dir / f"morph_intra_inter__{safe_filename(label + '_raw')}.png",
    )
    out["raw"]["intra_vs_inter"] = {
        "n_intra": res_raw.n_intra,
        "n_inter": res_raw.n_inter,
        "cliffs_delta": res_raw.cliffs_delta.__dict__,
        "cohen_d": res_raw.cohen_d.__dict__,
        "p_mannwhitney": res_raw.p_mannwhitney,
    }

    # De-anisotropise
    words = sorted(embeddings.keys())
    mat = np.stack([embeddings[w] for w in words])
    mat_d = deaniso(mat, method=deaniso_method, m=deaniso_pcs)
    emb_d = {w: mat_d[i] for i, w in enumerate(words)}

    res_d, dist_d = run_intra_vs_inter(families=families, embeddings=emb_d, seed=seed + 10)
    _plot_intra_inter(
        dist_d,
        title=f"{label} — intra vs inter ({deaniso_method})",
        out_path=figures_dir / f"morph_intra_inter__{safe_filename(label + '_deaniso')}.png",
    )
    out["deaniso"]["intra_vs_inter"] = {
        "method": deaniso_method,
        "pcs": deaniso_pcs,
        "n_intra": res_d.n_intra,
        "n_inter": res_d.n_inter,
        "cliffs_delta": res_d.cliffs_delta.__dict__,
        "cohen_d": res_d.cohen_d.__dict__,
        "p_mannwhitney": res_d.p_mannwhitney,
    }

    # (ii) offsets
    for suffix in ["ed", "ing", "s"]:
        r_raw = run_offset_consistency(families=families, embeddings=embeddings, suffix=suffix, seed=seed)
        r_d = run_offset_consistency(families=families, embeddings=emb_d, suffix=suffix, seed=seed + 1)
        out["raw"].setdefault("offsets", {})[suffix] = None if r_raw is None else {
            "n_offsets": r_raw.n_offsets,
            "mean_cos": r_raw.mean_cos.__dict__,
        }
        out["deaniso"].setdefault("offsets", {})[suffix] = None if r_d is None else {
            "n_offsets": r_d.n_offsets,
            "mean_cos": r_d.mean_cos.__dict__,
        }

    # (iii) probing
    p_raw = run_tense_probing(families=families, embeddings=embeddings, seed=seed)
    p_d = run_tense_probing(families=families, embeddings=emb_d, seed=seed + 1)
    out["raw"]["probing_tense_ed"] = None if p_raw is None else {
        "n_examples": p_raw.n_examples,
        "accuracy": p_raw.accuracy.__dict__,
        "macro_f1": p_raw.macro_f1.__dict__,
    }
    out["deaniso"]["probing_tense_ed"] = None if p_d is None else {
        "n_examples": p_d.n_examples,
        "accuracy": p_d.accuracy.__dict__,
        "macro_f1": p_d.macro_f1.__dict__,
    }

    return out

