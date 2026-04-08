from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from lab2.utils import normalize_token


@dataclass(frozen=True)
class MorphFamily:
    lemma: str
    forms: list[str]
    family_type: str
    transformation: str

    def all_words(self) -> list[str]:
        return list(self.forms)


def _dedupe_words(words: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for word in words:
        norm = normalize_token(word.strip())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def read_morph_families_tsv(path: Path) -> list[MorphFamily]:
    families: list[MorphFamily] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != 4:
                raise ValueError(f"Expected 4 columns in {path}, got {len(row)}: {row!r}")

            lemma = normalize_token(row[0].strip())
            forms = [item.strip() for item in row[1].split(",") if item.strip()]
            words = _dedupe_words([lemma, *forms])
            families.append(
                MorphFamily(
                    lemma=lemma,
                    forms=words,
                    family_type=row[2].strip(),
                    transformation=row[3].strip(),
                )
            )
    return families
