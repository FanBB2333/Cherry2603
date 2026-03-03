from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabConfig:
    random_seed: int = 42

    # WikiText preprocessing / matching
    min_tokens_per_line: int = 5
    max_tokens_per_sentence: int = 40

    # Context sampling policy (per PDF)
    max_sentences_per_word: int = 50
    min_sentences_keep: int = 10

    # Frequency threshold used when filtering synonym/antonym pairs (PDF asks to define/justify)
    min_word_occurrences: int = 10

    # BERT
    bert_model_name: str = "bert-base-uncased"
    # Layers to analyse (PDF suggests 1, 6, 12 as a manageable subset)
    bert_layers: tuple[int, ...] = (1, 6, 12)

    # GloVe
    glove_filename: str = "glove.6B.100d.txt"

    # Analysis knobs (kept small for reasonable runtime on CPU)
    anisotropy_sample_size: int = 500
    pca_top_k: int = 10
    nearest_k: int = 10

    # De-anisotropisation method used in later parts of the PDF where "a method of your choice"
    # is required for comparisons (raw vs de-anisotropised).
    deaniso_method: str = "remove_pcs"  # one of: center, center_norm, remove_pcs
    deaniso_pcs: int = 2
