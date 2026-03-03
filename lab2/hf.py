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


def load_wikitext103_train(paths: ProjectPaths):
    configure_hf_cache(paths)

    from datasets import load_dataset  # lazy import

    return load_dataset("wikitext", "wikitext-103-v1", split="train", cache_dir=str(paths.hf_cache_dir))


def load_bert_model_and_tokenizer(model_name: str, paths: ProjectPaths):
    configure_hf_cache(paths)

    from transformers import AutoModel, AutoTokenizer  # lazy import

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=str(paths.hf_cache_dir))
    model = AutoModel.from_pretrained(model_name, cache_dir=str(paths.hf_cache_dir))
    model.eval()
    return model, tokenizer

