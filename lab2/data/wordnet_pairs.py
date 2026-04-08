from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from lab2.paths import ProjectPaths
from lab2.utils import ensure_dir, normalize_token


def _normalize_lemma(text: str) -> str:
    lemma = normalize_token(text)
    if "_" in lemma or "-" in lemma:
        return ""
    if not lemma.isalpha():
        return ""
    return lemma


def _load_wordnet(paths: ProjectPaths):
    import nltk

    ensure_dir(paths.nltk_dir)
    nltk.data.path.insert(0, str(paths.nltk_dir))

    from nltk.corpus import wordnet as wn

    try:
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", download_dir=str(paths.nltk_dir))
        nltk.download("omw-1.4", download_dir=str(paths.nltk_dir))
    return wn


def iter_synonym_pairs(paths: ProjectPaths) -> Iterator[tuple[str, str]]:
    import itertools

    wn = _load_wordnet(paths)
    seen: set[tuple[str, str]] = set()
    for synset in wn.all_synsets():
        lemmas = sorted(
            {
                lemma
                for lemma in (_normalize_lemma(item.name()) for item in synset.lemmas())
                if lemma
            }
        )
        for a, b in itertools.combinations(lemmas, 2):
            pair = (a, b) if a <= b else (b, a)
            if pair in seen:
                continue
            seen.add(pair)
            yield pair


def iter_antonym_pairs(paths: ProjectPaths) -> Iterator[tuple[str, str]]:
    wn = _load_wordnet(paths)
    seen: set[tuple[str, str]] = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            a = _normalize_lemma(lemma.name())
            if not a:
                continue
            for antonym in lemma.antonyms():
                b = _normalize_lemma(antonym.name())
                if not b or a == b:
                    continue
                pair = (a, b) if a <= b else (b, a)
                if pair in seen:
                    continue
                seen.add(pair)
                yield pair


@dataclass(frozen=True)
class PairSamplingConfig:
    n_pairs: int
    min_wikitext_count: int
    seed: int


def filter_pairs(
    pairs: Iterable[tuple[str, str]],
    *,
    freqs,
    glove_vocab: set[str],
    min_wikitext_count: int,
) -> list[tuple[str, str]]:
    filtered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for a, b in pairs:
        if a == b:
            continue
        pair = (a, b) if a <= b else (b, a)
        if pair in seen:
            continue
        if pair[0] not in glove_vocab or pair[1] not in glove_vocab:
            continue
        if freqs.get(pair[0], 0) < min_wikitext_count:
            continue
        if freqs.get(pair[1], 0) < min_wikitext_count:
            continue
        seen.add(pair)
        filtered.append(pair)

    return filtered


def sample_pairs(
    pairs: Iterable[tuple[str, str]],
    *,
    freqs,
    glove_vocab: set[str],
    cfg: PairSamplingConfig,
) -> list[tuple[str, str]]:
    filtered = filter_pairs(
        pairs,
        freqs=freqs,
        glove_vocab=glove_vocab,
        min_wikitext_count=cfg.min_wikitext_count,
    )
    if len(filtered) <= cfg.n_pairs:
        return filtered

    rng = random.Random(cfg.seed)
    return sorted(rng.sample(filtered, k=cfg.n_pairs))


def write_pairs_tsv(path: Path, pairs: list[tuple[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("word1\tword2\n")
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")
