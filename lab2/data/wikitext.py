from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable, Iterator

from lab2.config import LabConfig
from lab2.data.contexts import WordContexts
from lab2.utils import normalize_token, tokenize_simple, word_rng


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _spacy_sentencizer():
    try:
        import spacy
    except ModuleNotFoundError:
        return None

    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _split_sentences(text: str) -> list[str]:
    nlp = _spacy_sentencizer()
    if nlp is not None:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return [part.strip() for part in _SENT_SPLIT_RE.split(text) if part.strip()]


def iter_preprocessed_sentences(
    lines: Iterable[str], *, config: LabConfig
) -> Iterator[str]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("="):
            continue
        if len(tokenize_simple(line)) < config.min_tokens_per_line:
            continue
        for sentence in _split_sentences(line):
            if sentence:
                yield sentence


def count_word_frequencies(sentences: Iterable[str]) -> Counter[str]:
    freqs: Counter[str] = Counter()
    for sentence in sentences:
        freqs.update(normalize_token(tok) for tok in tokenize_simple(sentence))
    return freqs


def sample_contexts_for_words(
    sentences: Iterable[str],
    *,
    target_words: set[str],
    config: LabConfig,
    include_excluded: bool = True,
) -> tuple[dict[str, WordContexts], list[str]]:
    targets = {normalize_token(word) for word in target_words if word.strip()}
    collected: dict[str, list[str]] = {word: [] for word in targets}
    total_matches: Counter[str] = Counter()

    for sentence in sentences:
        tokens = tokenize_simple(sentence)
        if not tokens or len(tokens) > config.max_tokens_per_sentence:
            continue

        present = set(normalize_token(tok) for tok in tokens).intersection(targets)
        for word in present:
            collected[word].append(sentence)
            total_matches[word] += 1

    contexts: dict[str, WordContexts] = {}
    excluded: list[str] = []

    for word in sorted(targets):
        raw_sentences = collected.get(word, [])
        if not raw_sentences:
            excluded.append(word)
            continue

        deduped = list(dict.fromkeys(raw_sentences))
        n_unique = len(deduped)
        if n_unique >= config.max_sentences_per_word:
            rng = word_rng(config.random_seed, word)
            kept = rng.sample(deduped, k=config.max_sentences_per_word)
        else:
            kept = deduped

        if n_unique < config.min_sentences_keep:
            excluded.append(word)
            if not include_excluded:
                continue

        contexts[word] = WordContexts(
            word=word,
            sentences=kept,
            low_frequency=config.min_sentences_keep <= n_unique < config.max_sentences_per_word,
            n_unique_sentences=n_unique,
            n_total_matches=int(total_matches[word]),
        )

    return contexts, excluded
