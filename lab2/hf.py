from __future__ import annotations

import os
from pathlib import Path

from lab2.paths import ProjectPaths


def configure_hf_cache(paths: ProjectPaths) -> Path:
    """
    Configure Hugging Face caches to live inside this repository.

    The lab explicitly asks for all downloads to be kept in the working directory.
    """

    cache_dir = paths.hf_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # `HF_HOME` controls most HF caches (datasets + transformers).
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    return cache_dir


class _LocalTextDataset:
    """
    Minimal dataset-like wrapper that yields dict rows with a `text` field.

    This exists as an offline fallback when `datasets` is not available, but the user has
    already placed the WikiText train split into the repo.
    """

    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                yield {"text": line.rstrip("\n")}


class _LocalParquetDataset:
    """
    Minimal dataset-like wrapper over local parquet shards with a `text` column.
    """

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __iter__(self):
        import pandas as pd

        for path in self.paths:
            df = pd.read_parquet(path, columns=["text"])
            for text in df["text"].tolist():
                yield {"text": text}


def _local_wikitext_parquet_files(paths: ProjectPaths) -> list[Path]:
    local_root = paths.data_dir / "Salesforce" / "wikitext" / "wikitext-103-v1"
    return sorted(local_root.glob("train-*.parquet")) if local_root.exists() else []


def load_wikitext103_train(paths: ProjectPaths):
    configure_hf_cache(paths)

    local_parquet_files = _local_wikitext_parquet_files(paths)

    try:
        from datasets import load_dataset  # lazy import

        if local_parquet_files:
            return load_dataset(
                "parquet",
                data_files={"train": [str(p) for p in local_parquet_files]},
                split="train",
                cache_dir=str(paths.hf_cache_dir),
            )
        return load_dataset("wikitext", "wikitext-103-v1", split="train", cache_dir=str(paths.hf_cache_dir))
    except ModuleNotFoundError:
        if local_parquet_files:
            return _LocalParquetDataset(local_parquet_files)

        # Offline fallback: user-provided text file in repo.
        candidates = [
            paths.data_dir / "wikitext103_train.txt",
            paths.data_dir / "wikitext-103-v1.train.txt",
            paths.data_dir / "wikitext-103" / "train.txt",
        ]
        for p in candidates:
            if p.exists():
                return _LocalTextDataset(p)
        raise ModuleNotFoundError(
            "Missing dependency `datasets`, and no local WikiText file found.\n"
            "Install datasets (`pip install datasets`) OR place the WikiText-103 train split at one of:\n"
            f"- {candidates[0]}\n- {candidates[1]}\n- {candidates[2]}"
        )


def load_bert_model_and_tokenizer(model_name: str, paths: ProjectPaths):
    configure_hf_cache(paths)

    from transformers import AutoModel, AutoTokenizer  # lazy import

    local_model_dir = paths.repo_root / "models" / model_name
    source = str(local_model_dir) if local_model_dir.exists() else model_name

    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, cache_dir=str(paths.hf_cache_dir))
    model = AutoModel.from_pretrained(source, cache_dir=str(paths.hf_cache_dir))
    model.eval()
    return model, tokenizer
