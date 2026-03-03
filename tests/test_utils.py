from __future__ import annotations

from lab2.utils import safe_filename, tokenize_simple


def test_tokenize_simple_basic() -> None:
    assert tokenize_simple("Hello, world!") == ["Hello", "world"]
    assert tokenize_simple("can't won't") == ["can't", "won't"]
    assert tokenize_simple("Version 2.0") == ["Version", "2", "0"]


def test_safe_filename_is_stable() -> None:
    a = safe_filename("walked")
    b = safe_filename("walked")
    assert a == b
    assert a.endswith(a.split("__")[-1])

