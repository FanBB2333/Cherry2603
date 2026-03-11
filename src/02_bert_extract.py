from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import json
import re

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
WORD2SENTS_PATH = CACHE_DIR / "word2sents.json"

MODEL_NAME = "bert-base-uncased"


def find_word_char_span(sentence: str, target_word: str) -> Optional[Tuple[int, int]]:
    """
    Find the first full-word match span (start_char, end_char), case-insensitive.
    Full-word match prevents 'cat' matching 'catch'.
    """
    pat = re.compile(rf"\b{re.escape(target_word)}\b", flags=re.IGNORECASE)
    m = pat.search(sentence)
    if not m:
        return None
    return (m.start(), m.end())


@torch.no_grad()
def get_bert_word_embedding(
    sentence: str,
    target_word: str,
    layer: int,
    tokenizer,
    model,
    device: str,
) -> np.ndarray:
    """
    Return embedding for target_word in sentence at the given layer.
    If tokenized into multiple wordpieces, mean-pool them.
    layer indexing:
      0 = embedding output
      1..12 = transformer layers for bert-base-uncased
    """
    span = find_word_char_span(sentence, target_word)
    if span is None:
        raise ValueError(f"Target word '{target_word}' not found as a full word in sentence.")

    enc = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()  # (seq_len, 2)

    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = out.hidden_states  # length 13

    if layer < 0 or layer >= len(hidden_states):
        raise ValueError(f"layer must be in [0, {len(hidden_states)-1}], got {layer}")

    hs = hidden_states[layer][0]  # (seq_len, hidden_dim)

    start_char, end_char = span

    token_indices: List[int] = []
    for i, (s, e) in enumerate(offsets):
        # special tokens typically have (0,0)
        if s == 0 and e == 0:
            continue
        # overlap between token span and target word span
        if s < end_char and e > start_char:
            token_indices.append(i)

    if not token_indices:
        raise RuntimeError(
            "No tokens aligned to target span. This can happen with unusual characters. "
            "Try a different sentence or inspect offsets."
        )

    vec = hs[token_indices].mean(dim=0)  # mean pooling across wordpieces
    return vec.detach().cpu().numpy()


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    word2sents = json.loads(Path(WORD2SENTS_PATH).read_text())
    sample_word = sorted(word2sents.keys())[0]
    sample_sent = word2sents[sample_word][0]

    print("Sample word:", sample_word)
    print("Sample sentence:", sample_sent)

    v12 = get_bert_word_embedding(sample_sent, sample_word, layer=12, tokenizer=tokenizer, model=model, device=device)
    v1 = get_bert_word_embedding(sample_sent, sample_word, layer=1, tokenizer=tokenizer, model=model, device=device)

    print("Vector shapes:", v1.shape, v12.shape)  # should be (768,)
    print("First 5 dims (layer12):", np.round(v12[:5], 6))


if __name__ == "__main__":
    main()
