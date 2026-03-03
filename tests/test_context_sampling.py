from __future__ import annotations

from lab2.config import LabConfig
from lab2.data.wikitext import sample_contexts_for_words


def test_sample_contexts_policy_excludes_low_counts() -> None:
    cfg = LabConfig(min_sentences_keep=3, max_sentences_per_word=5, max_tokens_per_sentence=40)
    sents = [
        "alpha beta gamma delta epsilon",
        "alpha beta gamma delta epsilon",
        "alpha zeta eta theta iota",
        "kappa lambda mu nu xi",
    ]
    contexts, excluded = sample_contexts_for_words(sents, target_words={"alpha", "kappa"}, config=cfg)
    assert "kappa" in contexts
    assert "alpha" in excluded  # only 2 unique sentences for alpha -> < min_sentences_keep

