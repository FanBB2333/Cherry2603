from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lab2.utils import ensure_dir, safe_filename


@dataclass(frozen=True)
class StoredVector:
    word: str
    path: Path


def vector_path(dir_path: Path, *, word: str) -> Path:
    return dir_path / f"{safe_filename(word)}.npy"


def save_vector(dir_path: Path, *, word: str, vector: "numpy.ndarray") -> Path:
    import numpy as np

    ensure_dir(dir_path)
    path = vector_path(dir_path, word=word)
    np.save(path, vector.astype("float32", copy=False))
    return path


def load_vector(path: Path) -> "numpy.ndarray":
    import numpy as np

    return np.load(path)


def write_index(path: Path, items: Iterable[StoredVector]) -> None:
    ensure_dir(path.parent)
    payload = [{"word": it.word, "path": str(it.path)} for it in items]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_index(path: Path) -> list[StoredVector]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [StoredVector(word=row["word"], path=Path(row["path"])) for row in payload]

