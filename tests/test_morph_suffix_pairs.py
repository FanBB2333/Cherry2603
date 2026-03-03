from __future__ import annotations

from lab2.analysis.morphology import extract_suffix_pairs
from lab2.data.morph_families import MorphFamily


def test_extract_suffix_pairs_ed_and_ing() -> None:
    fam = MorphFamily(lemma="walk", forms=["walk", "walked", "walking"], family_type="inflection", transformation="x")
    pairs_ed = extract_suffix_pairs([fam], suffix="ed")
    pairs_ing = extract_suffix_pairs([fam], suffix="ing")
    assert ("walk", "walked") in pairs_ed
    assert ("walk", "walking") in pairs_ing

