from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from lab2.utils import ensure_dir


@dataclass(frozen=True)
class WordContexts:
    word: str
    sentences: list[str]
    low_frequency: bool
    n_unique_sentences: int
    n_total_matches: int


def read_contexts_jsonl(path: Path) -> dict[str, WordContexts]:
    rows: dict[str, WordContexts] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            item = WordContexts(
                word=row["word"],
                sentences=list(row["sentences"]),
                low_frequency=bool(row["low_frequency"]),
                n_unique_sentences=int(row["n_unique_sentences"]),
                n_total_matches=int(row["n_total_matches"]),
            )
            rows[item.word] = item
    return rows


def write_contexts_jsonl(path: Path, contexts: Mapping[str, WordContexts]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for word in sorted(contexts):
            item = contexts[word]
            f.write(
                json.dumps(
                    {
                        "word": item.word,
                        "sentences": item.sentences,
                        "low_frequency": item.low_frequency,
                        "n_unique_sentences": item.n_unique_sentences,
                        "n_total_matches": item.n_total_matches,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
