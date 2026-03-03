from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def tokenize_simple(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def normalize_token(token: str) -> str:
    return token.lower()


def word_rng(seed: int, word: str) -> random.Random:
    digest = hashlib.md5(f"{seed}:{word}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "little", signed=False))


def safe_filename(text: str) -> str:
    # Stable, filesystem-safe, reversible enough for debugging
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._-") or "item"
    return f"{cleaned}__{h}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pickle(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def read_pickle(path: Path) -> Any:
    return pickle.loads(path.read_bytes())


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
