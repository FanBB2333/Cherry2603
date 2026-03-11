from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"

MORPH_TSV = PROJECT_ROOT / "morph_families.tsv"

VECTORS = {
    "glove": CACHE_DIR / "glove_6B_300d_targetwords.npz",
    "bert_l1": CACHE_DIR / "bert_contextual_layer1.npz",
    "bert_l6": CACHE_DIR / "bert_contextual_layer6.npz",
    "bert_l12": CACHE_DIR / "bert_contextual_layer12.npz",
}

SEED = 42


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


def mean_center(vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    X = np.stack(list(vectors.values()), axis=0)
    mu = X.mean(axis=0, keepdims=True)
    return {w: (vec - mu[0]) for w, vec in vectors.items()}


def parse_word_labels(path: Path) -> Tuple[List[str], List[str]]:
    """
    Build word-level labels from morph_families.tsv:
    For each row: label is family_type, applied to lemma and all forms in that family.
    Returns (words, labels) lists (with duplicates possible; we'll dedupe consistently later).
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    df.columns = ["lemma", "forms", "family_type", "transform"]
    words = []
    labels = []
    for _, r in df.iterrows():
        y = r["family_type"].strip()
        lemma = r["lemma"].strip().lower()
        forms = [x.strip().lower() for x in r["forms"].split(",") if x.strip()]
        fam_words = [lemma] + forms
        for w in fam_words:
            words.append(w)
            labels.append(y)
    return words, labels


def build_dataset(vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create X, y using only words present in vectors.
    Deduplicate words: if a word appears multiple times, keep the first label (should be consistent here).
    """
    words, labels = parse_word_labels(MORPH_TSV)

    seen = set()
    X_list = []
    y_list = []
    kept_words = []

    for w, y in zip(words, labels):
        if w in seen:
            continue
        seen.add(w)
        if w not in vectors:
            continue
        X_list.append(vectors[w])
        y_list.append(y)
        kept_words.append(w)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y, kept_words


def bootstrap_ci(values: np.ndarray, B: int = 5000, alpha: float = 0.05) -> Tuple[float, float]:
    n = len(values)
    stats = []
    rng = np.random.default_rng(SEED)
    for _ in range(B):
        samp = rng.choice(values, size=n, replace=True)
        stats.append(float(np.mean(samp)))
    stats = np.sort(np.array(stats))
    lo = stats[int((alpha/2) * B)]
    hi = stats[int((1 - alpha/2) * B) - 1]
    return float(lo), float(hi)


def eval_logreg(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    accs = []
    f1s = []

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="macro"))

    accs = np.array(accs, dtype=np.float64)
    f1s = np.array(f1s, dtype=np.float64)

    return {
        "n_samples": int(len(y)),
        "class_counts": {c: int(np.sum(y == c)) for c in sorted(set(y))},
        "acc_mean": float(accs.mean()),
        "acc_ci": bootstrap_ci(accs),
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_ci": bootstrap_ci(f1s),
        "fold_accs": [float(x) for x in accs],
        "fold_macro_f1s": [float(x) for x in f1s],
    }


def main():
    results = {}

    for model_name, path in VECTORS.items():
        vecs_raw = load_npz(path)
        vecs_mc = mean_center(vecs_raw)

        X_raw, y_raw, words_raw = build_dataset(vecs_raw)
        X_mc, y_mc, words_mc = build_dataset(vecs_mc)

        # sanity: same labels/order is expected
        assert list(y_raw) == list(y_mc)

        results[model_name] = {
            "raw": eval_logreg(X_raw, y_raw),
            "mean_center": eval_logreg(X_mc, y_mc),
        }

    out = CACHE_DIR / "probe_family_type_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("Saved:", out)

    # Print a compact summary
    for model_name, block in results.items():
        for setting in ["raw", "mean_center"]:
            r = block[setting]
            print(
                f"{model_name}\t{setting}\tN={r['n_samples']}\t"
                f"acc={r['acc_mean']:.3f} CI{tuple(round(x,3) for x in r['acc_ci'])}\t"
                f"macroF1={r['macro_f1_mean']:.3f} CI{tuple(round(x,3) for x in r['macro_f1_ci'])}"
            )


if __name__ == "__main__":
    main()
