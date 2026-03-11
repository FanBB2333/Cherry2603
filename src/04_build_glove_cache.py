from __future__ import annotations
from pathlib import Path
import json
import zipfile
import urllib.request

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

TARGET_WORDS_PATH = CACHE_DIR / "target_words.json"

# Standard GloVe package
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP = CACHE_DIR / "glove.6B.zip"
GLOVE_TXT_NAME = "glove.6B.300d.txt"  # 300-dim
OUT_NPZ = CACHE_DIR / "glove_6B_300d_targetwords.npz"


def download_if_needed(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] already downloaded: {dest.name}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    target_words = json.loads(TARGET_WORDS_PATH.read_text())
    target_set = set(target_words)
    print("Target words:", len(target_words))

    if OUT_NPZ.exists():
        print(f"[skip] {OUT_NPZ.name} already exists")
        return

    download_if_needed(GLOVE_URL, GLOVE_ZIP)

    # Read vectors line-by-line from the zip
    vectors = {}
    found = 0

    with zipfile.ZipFile(GLOVE_ZIP, "r") as zf:
        if GLOVE_TXT_NAME not in zf.namelist():
            raise FileNotFoundError(f"{GLOVE_TXT_NAME} not found in {GLOVE_ZIP.name}")
        with zf.open(GLOVE_TXT_NAME) as f:
            for raw in tqdm(f, desc=f"Scanning {GLOVE_TXT_NAME}"):
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split()
                word = parts[0]
                if word in target_set:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    vectors[word] = vec
                    found += 1
                    if found == len(target_set):
                        break

    print("Found GloVe vectors:", found)
    if found < len(target_set):
        missing = sorted(list(target_set - set(vectors.keys())))[:20]
        print("Missing examples (first 20):", missing)

    np.savez_compressed(OUT_NPZ, **vectors)
    print("Saved:", OUT_NPZ)


if __name__ == "__main__":
    main()
