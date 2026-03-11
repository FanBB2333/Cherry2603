from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src_bert_extract import get_bert_word_embedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
WORD2SENTS_PATH = CACHE_DIR / "word2sents.json"

MODEL_NAME = "bert-base-uncased"
LAYERS = [1, 6, 12]  # we start with these three layers


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    word2sents: Dict[str, List[str]] = json.loads(Path(WORD2SENTS_PATH).read_text())

    for layer in LAYERS:
        out_path = CACHE_DIR / f"bert_contextual_layer{layer}.npz"
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue

        vectors = {}
        failed = 0

        for word, sents in tqdm(word2sents.items(), desc=f"Layer {layer} words"):
            sent_vecs = []
            for sent in sents:
                try:
                    v = get_bert_word_embedding(
                        sent, word, layer=layer,
                        tokenizer=tokenizer, model=model, device=device
                    )
                    sent_vecs.append(v)
                except Exception:
                    failed += 1
                    continue

            if sent_vecs:
                vectors[word] = np.mean(np.stack(sent_vecs, axis=0), axis=0)

        np.savez_compressed(out_path, **vectors)
        print(f"Saved {len(vectors)} word vectors to {out_path.name}. Failed sentence extractions: {failed}")


if __name__ == "__main__":
    main()
