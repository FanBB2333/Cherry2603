from __future__ import annotations

from lab2.embeddings.bert import _find_full_word_spans


def test_find_full_word_spans_full_word_only() -> None:
    s = "He walked, then walking; sidewalk is different."
    # "walk" should not match inside "walked"/"walking"/"sidewalk"
    assert _find_full_word_spans(s, "walk") == []
    assert _find_full_word_spans(s, "walked") == [(3, 9)]
    assert _find_full_word_spans(s, "walking") == [(16, 23)]


def test_find_full_word_spans_case_insensitive() -> None:
    s = "LEGAL illegal IlLeGaL."
    assert len(_find_full_word_spans(s, "illegal")) == 2

