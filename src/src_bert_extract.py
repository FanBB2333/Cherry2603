from __future__ import annotations
from typing import List, Tuple, Optional
import re
import numpy as np
import torch

def find_word_char_span(sentence: str, target_word: str) -> Optional[Tuple[int, int]]:
    pat = re.compile(rf"\b{re.escape(target_word)}\b", flags=re.IGNORECASE)
    m = pat.search(sentence)
    if not m:
        return None
    return (m.start(), m.end())

@torch.no_grad()
def get_bert_word_embedding(sentence: str, target_word: str, layer: int, tokenizer, model, device: str) -> np.ndarray:
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
    offsets = enc["offset_mapping"][0].tolist()

    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = out.hidden_states

    if layer < 0 or layer >= len(hidden_states):
        raise ValueError(f"layer must be in [0, {len(hidden_states)-1}], got {layer}")

    hs = hidden_states[layer][0]

    start_char, end_char = span
    token_indices: List[int] = []
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if s < end_char and e > start_char:
            token_indices.append(i)

    if not token_indices:
        raise RuntimeError("No tokens aligned to target span.")

    vec = hs[token_indices].mean(dim=0)
    return vec.detach().cpu().numpy()
