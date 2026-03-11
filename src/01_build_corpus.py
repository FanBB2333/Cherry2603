"""
methods modified & summarised by gpt
Build resources required before section 1.1:
1) Parse morph_families.tsv
2) Load WikiText-103 train split, clean, sentence-split
3) Build word frequency table
4) For each target word (from morph families), collect sentences containing it
   with PDF rules:
   - case-insensitive, full-word match
   - drop sentences > 40 tokens
   - remove duplicates
   - if >=50, sample 50 with seed=42
   - if 10-49, keep all (mark low-frequency)
   - if <10, exclude from contextual analyses and record
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set
import json
import re
import random

import pandas as pd
from datasets import load_dataset
import spacy
from tqdm import tqdm


# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MORPH_TSV = PROJECT_ROOT / "morph_families.tsv"

# PDF sampling rule
MAX_SENT_PER_WORD = 50
MIN_SENT_PER_WORD = 10
SEED = 42

# PDF sentence length rule (after the tokenization)
MAX_TOKENS_PER_SENT = 40

# Clean rules 
MIN_TOKENS_PER_LINE = 5  # discard lines with < 5 tokens
HEADING_PREFIX = "="     # remove lines starting with "="

# Choose spaCy model
SPACY_MODEL = "en_core_web_sm"


@dataclass
class MorphFamily:
    lemma: str
    forms: List[str]
    family_type: str
    transform: str

    def all_words(self) -> List[str]:
        # include lemma + forms; normalize to lowercase
        words = [self.lemma] + list(self.forms)
        norm = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            norm.append(w.lower())
        # de-dup preserving order
        seen = set()
        out = []
        for w in norm:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out


def parse_morph_families(path: Path) -> List[MorphFamily]:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if df.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, got {df.shape[1]}")

    families: List[MorphFamily] = []
    for _, row in df.iterrows():
        lemma = row[0].strip()
        forms_raw = row[1].strip()
        family_type = row[2].strip()
        transform = row[3].strip()

        forms = [x.strip() for x in forms_raw.split(",") if x.strip()]
        families.append(MorphFamily(lemma=lemma, forms=forms, family_type=family_type, transform=transform))
    return families


def clean_wikitext_lines(lines: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        if line.startswith(HEADING_PREFIX):
            continue
        # quick token count on whitespace (PDF says drop <5 tokens lines)
        if len(line.split()) < MIN_TOKENS_PER_LINE:
            continue
        cleaned.append(line)
    return cleaned


def sent_tokenize_spacy(nlp, text: str) -> List[str]:
    doc = nlp(text)
    sents = []
    for s in doc.sents:
        st = s.text.strip()
        if st:
            sents.append(st)
    return sents


def tokenize_simple(sent: str) -> List[str]:
    # simple tokenization for the "<=40 tokens" constraint.
    # We'll count word-like tokens. This isn't BERT tokenization; it's just for filtering.
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", sent)


def build_word_frequency(sentences: Iterable[str]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for sent in sentences:
        toks = tokenize_simple(sent)
        for t in toks:
            w = t.lower()
            freq[w] = freq.get(w, 0) + 1
    return freq


def compile_word_regex(word: str) -> re.Pattern:
    # full-word match, case-insensitive
    # \b works reasonably for latin words; we also escape the word.
    return re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)


def build_word2sentences(
    sentences: List[str],
    target_words: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, int]]:
    """
    ai
    Returns:
      word2sents: word -> list of sampled sentences (10-50 or exactly 50)
      kept_counts: word -> number of sentences kept (after sampling)
      excluded_counts: word -> number of matched sentences found (<10 after filtering)
    """
    # Precompile regex patterns for speed
    patterns = {w: compile_word_regex(w) for w in target_words}

    # Collect raw matches
    word2raw: Dict[str, List[str]] = {w: [] for w in target_words}

    for sent in tqdm(sentences, desc="Scanning sentences for target words"):
        toks = tokenize_simple(sent)
        if len(toks) > MAX_TOKENS_PER_SENT:
            continue

        # check matches for each word. Since target_words is small (TSV only 42 families),
        # If it gets big later - optimize.
        for w, pat in patterns.items():
            if pat.search(sent):
                word2raw[w].append(sent)

    rng = random.Random(SEED)

    word2sents: Dict[str, List[str]] = {}
    kept_counts: Dict[str, int] = {}
    excluded_counts: Dict[str, int] = {}

    for w, sents in word2raw.items():
        # de-duplicate while preserving order
        seen: Set[str] = set()
        dedup = []
        for s in sents:
            if s not in seen:
                seen.add(s)
                dedup.append(s)

        if len(dedup) < MIN_SENT_PER_WORD:
            excluded_counts[w] = len(dedup)
            continue

        if len(dedup) >= MAX_SENT_PER_WORD:
            sampled = rng.sample(dedup, k=MAX_SENT_PER_WORD)
        else:
            sampled = dedup

        word2sents[w] = sampled
        kept_counts[w] = len(sampled)

    return word2sents, kept_counts, excluded_counts


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Reading morph families from:", MORPH_TSV)
    families = parse_morph_families(MORPH_TSV)
    print(f"Loaded {len(families)} families")

    # build target word list from TSV (lemma + forms)
    target_words_set: Set[str] = set()
    for fam in families:
        for w in fam.all_words():
            target_words_set.add(w)
    target_words = sorted(target_words_set)
    print(f"Total unique candidate words from TSV: {len(target_words)}")

    # load WikiText train split
    print("Loading WikiText-103 train split...")
    ds = load_dataset("wikitext", "wikitext-103-v1")
    train_lines = ds["train"]["text"]

    # clean lines, then sentence-split line by line (safer memory-wise)
    print("Loading spaCy:", SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL)
    nlp.disable_pipes("ner", "tagger", "lemmatizer")  # faster; sentence boundaries still work

    cleaned_lines = clean_wikitext_lines(train_lines)
    print("Cleaned lines:", len(cleaned_lines))

    print("Sentence splitting (this can take a while)...")
    sentences: List[str] = []
    for line in tqdm(cleaned_lines, desc="Sentence splitting"):
        sentences.extend(sent_tokenize_spacy(nlp, line))

    print("Total sentences after splitting:", len(sentences))

    # build word frequency table over sentences
    print("Building word frequency table...")
    word_freq = build_word_frequency(sentences)
    print("Unique word types in freq table:", len(word_freq))

    # build word->sentences mapping for TSV candidate words
    word2sents, kept_counts, excluded_counts = build_word2sentences(sentences, target_words)

    print("\nSummary:")
    print("Words with >=10 sentences kept:", len(word2sents))
    print("Words excluded (<10 sentences):", len(excluded_counts))

    # save caches
    (CACHE_DIR / "target_words.json").write_text(json.dumps(target_words, ensure_ascii=False, indent=2))
    (CACHE_DIR / "word_freq.json").write_text(json.dumps(word_freq, ensure_ascii=False))
    (CACHE_DIR / "word2sents.json").write_text(json.dumps(word2sents, ensure_ascii=False))
    (CACHE_DIR / "kept_counts.json").write_text(json.dumps(kept_counts, ensure_ascii=False, indent=2))
    (CACHE_DIR / "excluded_counts.json").write_text(json.dumps(excluded_counts, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(" - data_cache/target_words.json")
    print(" - data_cache/word_freq.json")
    print(" - data_cache/word2sents.json")
    print(" - data_cache/kept_counts.json")
    print(" - data_cache/excluded_counts.json")

    # quick preview of top excluded words (lowest sentence counts)
    if excluded_counts:
        worst = sorted(excluded_counts.items(), key=lambda x: x[1])[:10]
        print("\nExample excluded words (lowest counts):")
        for w, c in worst:
            print(f"  {w}: {c}")

if __name__ == "__main__":
    main()
