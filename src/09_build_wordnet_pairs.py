from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
import random
import itertools

import numpy as np
from nltk.corpus import wordnet as wn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

GLOVE_NPZ = CACHE_DIR / "glove_6B_300d_targetwords.npz"  # defines our working vocab

SEED = 42
rng = random.Random(SEED)

OUT = CACHE_DIR / "wordnet_pairs.json"


def load_vocab_from_npz(path: Path) -> Set[str]:
    z = np.load(path)
    return set(z.files)


def normalize_lemma(s: str) -> str:
    # WordNet lemmas can contain '_' for multiword. We drop multiwords for simplicity.
    s = s.lower()
    if "_" in s:
        return ""
    if not s.isalpha():
        # keep only alphabetic to match your corpus vocab style
        return ""
    return s


def build_syn_pairs(vocab: Set[str]) -> Set[Tuple[str, str]]:
    pairs = set()
    for syn in wn.all_synsets():
        lemmas = [normalize_lemma(l.name()) for l in syn.lemmas()]
        lemmas = [l for l in lemmas if l and l in vocab]
        lemmas = sorted(set(lemmas))
        for a, b in itertools.combinations(lemmas, 2):
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def build_ant_pairs(vocab: Set[str]) -> Set[Tuple[str, str]]:
    pairs = set()
    for syn in wn.all_synsets():
        for l in syn.lemmas():
            a = normalize_lemma(l.name())
            if not a or a not in vocab:
                continue
            for ant in l.antonyms():
                b = normalize_lemma(ant.name())
                if not b or b not in vocab:
                    continue
                pairs.add(tuple(sorted((a, b))))
    return pairs


def sample_pairs(pairs: List[Tuple[str, str]], n: int) -> List[Tuple[str, str]]:
    if len(pairs) <= n:
        return pairs
    return rng.sample(pairs, n)


def main():
    vocab = load_vocab_from_npz(GLOVE_NPZ)
    print("Working vocab size:", len(vocab))

    syn_pairs = sorted(build_syn_pairs(vocab))
    ant_pairs = sorted(build_ant_pairs(vocab))

    # target ~200 each (as in assignment expectation)
    syn_200 = sample_pairs(syn_pairs, 200)
    ant_200 = sample_pairs(ant_pairs, 200)

    # random baseline pairs (same size as synonyms/antonyms)
    vocab_list = sorted(vocab)
    rand_pairs = set()
    while len(rand_pairs) < 200:
        a, b = rng.sample(vocab_list, 2)
        rand_pairs.add(tuple(sorted((a, b))))
    rand_200 = sorted(rand_pairs)

    OUT.write_text(json.dumps({
        "vocab_size": len(vocab),
        "syn_total": len(syn_pairs),
        "ant_total": len(ant_pairs),
        "syn_200": syn_200,
        "ant_200": ant_200,
        "rand_200": rand_200
    }, indent=2))
    print("Saved:", OUT)
    print("syn_total:", len(syn_pairs), "ant_total:", len(ant_pairs))

if __name__ == "__main__":
    main()
