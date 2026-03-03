from __future__ import annotations

from collections import Counter
from pathlib import Path

from lab2.config import LabConfig
from lab2.hf import load_wikitext103_train
from lab2.paths import ProjectPaths
from lab2.utils import ensure_dir, write_json, write_jsonl, write_pickle


def prepare_corpus(*, config: LabConfig, paths: ProjectPaths) -> None:
    """
    Build the WikiText-103 resources required by the PDF:
    - clean lines, segment into sentences
    - compute word frequencies (case-insensitive)
    - optionally sample contexts for all forms listed in `data/morph_families.tsv`
    """

    ensure_dir(paths.results_dir)
    ensure_dir(paths.contexts_dir)

    print("Loading WikiText-103 (train split)…")
    dataset = load_wikitext103_train(paths)

    from lab2.data.wikitext import count_word_frequencies, iter_preprocessed_sentences, sample_contexts_for_words

    # 1) Frequency counts (used later to filter words/pairs by corpus frequency)
    print("Counting word frequencies… (this can take a while)")
    freqs = count_word_frequencies(
        iter_preprocessed_sentences((row["text"] for row in dataset), config=config)
    )
    write_pickle(paths.results_dir / "wikitext_word_freq.pkl", freqs)
    write_json(paths.results_dir / "wikitext_word_freq.meta.json", {"note": "Counts on preprocessed sentences."})

    # 2) Context sampling for morphological families (if dataset is provided)
    if not paths.morph_families_path.exists():
        print(f"Skipping morph contexts: missing {paths.morph_families_path}")
        return

    print("Reading morphological families…")
    from lab2.data.morph_families import read_morph_families_tsv

    families = read_morph_families_tsv(paths.morph_families_path)
    all_forms = {f for fam in families for f in fam.forms}

    print(f"Sampling contexts for {len(all_forms)} morphological forms…")
    contexts, excluded = sample_contexts_for_words(
        iter_preprocessed_sentences((row["text"] for row in dataset), config=config),
        target_words=all_forms,
        config=config,
    )

    out_path = paths.contexts_dir / "morph_contexts.jsonl"
    write_jsonl(
        out_path,
        (
            {
                "word": cs.word,
                "low_frequency": cs.low_frequency,
                "n_unique_sentences": cs.n_unique_sentences,
                "n_total_matches": cs.n_total_matches,
                "sentences": cs.sentences,
            }
            for cs in contexts.values()
        ),
    )
    write_json(paths.results_dir / "morph_contexts_excluded.json", {"excluded_words": excluded})
    print(f"Wrote: {out_path}")


def build_synant(*, config: LabConfig, paths: ProjectPaths) -> None:
    """
    Build ~200 synonym pairs and ~200 antonym pairs from WordNet, filtered by:
    - GloVe vocabulary membership
    - WikiText-103 frequency threshold (config.min_word_occurrences)
    """

    ensure_dir(paths.results_dir)
    freq_path = paths.results_dir / "wikitext_word_freq.pkl"
    if not freq_path.exists():
        raise FileNotFoundError(f"Missing {freq_path}. Run `python -m lab2 prepare-corpus` first.")

    freqs: Counter[str] = Counter()
    from lab2.utils import read_pickle

    freqs = read_pickle(freq_path)

    glove_path = paths.glove_dir / config.glove_filename
    if not glove_path.exists():
        raise FileNotFoundError(
            f"Missing {glove_path}. Put a GloVe text file there (see README.md)."
        )

    from lab2.embeddings.glove import load_glove_vocab

    print("Loading GloVe vocabulary…")
    glove_vocab = load_glove_vocab(glove_path, cache_dir=paths.glove_dir)

    from lab2.data.wordnet_pairs import (
        PairSamplingConfig,
        iter_antonym_pairs,
        iter_synonym_pairs,
        sample_pairs,
        write_pairs_tsv,
    )

    pair_cfg = PairSamplingConfig(
        n_pairs=200,
        min_wikitext_count=config.min_word_occurrences,
        seed=config.random_seed,
    )

    print("Sampling synonym pairs from WordNet…")
    syn_pairs = sample_pairs(
        iter_synonym_pairs(paths),
        freqs=freqs,
        glove_vocab=glove_vocab,
        cfg=pair_cfg,
    )
    print("Sampling antonym pairs from WordNet…")
    ant_pairs = sample_pairs(
        iter_antonym_pairs(paths),
        freqs=freqs,
        glove_vocab=glove_vocab,
        cfg=PairSamplingConfig(
            n_pairs=200,
            min_wikitext_count=config.min_word_occurrences,
            seed=config.random_seed + 1,
        ),
    )

    syn_out = paths.results_dir / "synonym_pairs.tsv"
    ant_out = paths.results_dir / "antonym_pairs.tsv"
    write_pairs_tsv(syn_out, syn_pairs)
    write_pairs_tsv(ant_out, ant_pairs)
    print(f"Wrote: {syn_out}")
    print(f"Wrote: {ant_out}")


def compute_embeddings(*, config: LabConfig, paths: ProjectPaths) -> None:  # pragma: no cover
    ensure_dir(paths.embeddings_dir)
    ensure_dir(paths.results_dir)

    from lab2.utils import read_pickle

    freq_path = paths.results_dir / "wikitext_word_freq.pkl"
    if not freq_path.exists():
        raise FileNotFoundError(f"Missing {freq_path}. Run `python -m lab2 prepare-corpus` first.")

    freqs = read_pickle(freq_path)

    morph_ctx_path = paths.contexts_dir / "morph_contexts.jsonl"
    if not morph_ctx_path.exists():
        raise FileNotFoundError(
            f"Missing {morph_ctx_path}. Run `python -m lab2 prepare-corpus` first."
        )

    from lab2.data.contexts import read_contexts_jsonl, write_contexts_jsonl, WordContexts

    morph_ctx = read_contexts_jsonl(morph_ctx_path)

    # Syn/ant contexts (built from WordNet pairs)
    syn_pairs_path = paths.results_dir / "synonym_pairs.tsv"
    ant_pairs_path = paths.results_dir / "antonym_pairs.tsv"
    synant_words: set[str] = set()
    for p in [syn_pairs_path, ant_pairs_path]:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("word1"):
                continue
            a, b = line.split("\t")
            synant_words.add(a.strip().lower())
            synant_words.add(b.strip().lower())

    synant_ctx: dict[str, WordContexts] = {}
    synant_ctx_path = paths.contexts_dir / "synant_contexts.jsonl"
    if synant_words:
        if synant_ctx_path.exists():
            synant_ctx = read_contexts_jsonl(synant_ctx_path)
        else:
            print(f"Sampling contexts for {len(synant_words)} synonym/antonym words…")
            dataset = load_wikitext103_train(paths)
            from lab2.data.wikitext import iter_preprocessed_sentences, sample_contexts_for_words

            contexts, excluded = sample_contexts_for_words(
                iter_preprocessed_sentences((row["text"] for row in dataset), config=config),
                target_words=synant_words,
                config=config,
            )
            synant_ctx = {
                w: WordContexts(
                    word=cs.word,
                    sentences=cs.sentences,
                    low_frequency=cs.low_frequency,
                    n_unique_sentences=cs.n_unique_sentences,
                    n_total_matches=cs.n_total_matches,
                )
                for w, cs in contexts.items()
            }
            write_contexts_jsonl(synant_ctx_path, synant_ctx)
            write_json(paths.results_dir / "synant_contexts_excluded.json", {"excluded_words": excluded})

    # Merge contexts; if a word appears in both, keep the one with more sentences.
    merged_ctx: dict[str, WordContexts] = dict(morph_ctx)
    for w, wc in synant_ctx.items():
        prev = merged_ctx.get(w)
        if prev is None or wc.n_unique_sentences > prev.n_unique_sentences:
            merged_ctx[w] = wc

    # Random words for anisotropy probing (PDF: similarities between randomly selected words).
    if config.anisotropy_sample_size > 0:
        from lab2.embeddings.glove import load_glove_vocab

        glove_path_for_vocab = paths.glove_dir / config.glove_filename
        glove_vocab = load_glove_vocab(glove_path_for_vocab, cache_dir=paths.glove_dir)

        import random

        rng = random.Random(config.random_seed)
        candidates = [
            w
            for w, c in freqs.items()
            if c >= config.min_word_occurrences and w in glove_vocab and w not in merged_ctx
        ]
        if len(candidates) <= config.anisotropy_sample_size:
            random_words = set(candidates)
        else:
            random_words = set(rng.sample(candidates, k=config.anisotropy_sample_size))

        random_ctx_path = paths.contexts_dir / "random_contexts.jsonl"
        random_ctx: dict[str, WordContexts] = {}
        if random_ctx_path.exists():
            random_ctx = read_contexts_jsonl(random_ctx_path)
        else:
            print(f"Sampling contexts for {len(random_words)} random words (anisotropy)…")
            dataset = load_wikitext103_train(paths)
            from lab2.data.wikitext import iter_preprocessed_sentences, sample_contexts_for_words

            contexts, excluded = sample_contexts_for_words(
                iter_preprocessed_sentences((row["text"] for row in dataset), config=config),
                target_words=random_words,
                config=config,
            )
            random_ctx = {
                w: WordContexts(
                    word=cs.word,
                    sentences=cs.sentences,
                    low_frequency=cs.low_frequency,
                    n_unique_sentences=cs.n_unique_sentences,
                    n_total_matches=cs.n_total_matches,
                )
                for w, cs in contexts.items()
            }
            write_contexts_jsonl(random_ctx_path, random_ctx)
            write_json(paths.results_dir / "random_contexts_excluded.json", {"excluded_words": excluded})

        for w, wc in random_ctx.items():
            merged_ctx.setdefault(w, wc)

    target_words = set(merged_ctx.keys())
    print(f"Total target words with contexts: {len(target_words)}")

    # 1) GloVe vectors (static baseline)
    glove_path = paths.glove_dir / config.glove_filename
    if not glove_path.exists():
        raise FileNotFoundError(
            f"Missing {glove_path}. Put a GloVe text file there (see README.md)."
        )

    from lab2.embeddings.glove import load_glove_vectors
    from lab2.embeddings.store import StoredVector, save_vector, write_index
    from lab2.utils import safe_filename

    print("Loading GloVe vectors for target words…")
    glove = load_glove_vectors(glove_path, words=target_words)

    glove_root = paths.embeddings_dir / "glove" / safe_filename(config.glove_filename)
    glove_vec_dir = glove_root / "vectors"
    stored_glove: list[StoredVector] = []
    for w, vec in glove.vectors.items():
        stored_glove.append(StoredVector(word=w, path=save_vector(glove_vec_dir, word=w, vector=vec)))
    write_index(glove_root / "index.json", stored_glove)
    write_json(
        glove_root / "meta.json",
        {
            "glove_file": str(glove_path),
            "dim": glove.dim,
            "n_words": len(stored_glove),
        },
    )

    missing_glove = sorted(target_words - set(glove.vectors.keys()))
    if missing_glove:
        write_json(paths.results_dir / "glove_missing_words.json", {"missing_words": missing_glove})
        print(f"Warning: {len(missing_glove)} target words not in GloVe; written to results.")

    # 2) BERT contextual word embeddings (mean over contexts)
    print("Loading BERT model + tokenizer…")
    from lab2.hf import load_bert_model_and_tokenizer
    from lab2.embeddings.bert import BertEmbedder

    model, tokenizer = load_bert_model_and_tokenizer(config.bert_model_name, paths)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BertEmbedder(model=model, tokenizer=tokenizer, device=device)

    bert_root = paths.embeddings_dir / "bert" / safe_filename(config.bert_model_name)
    layer_dirs = {layer: (bert_root / f"layer_{layer:02d}") for layer in config.bert_layers}
    for d in layer_dirs.values():
        ensure_dir(d)

    from lab2.embeddings.store import read_index

    bert_index_by_layer: dict[int, dict[str, StoredVector]] = {layer: {} for layer in config.bert_layers}
    for layer, d in layer_dirs.items():
        index_path = d / "index.json"
        if index_path.exists():
            for it in read_index(index_path):
                bert_index_by_layer[layer][it.word] = it
    bert_stats: dict[str, dict[str, int]] = {}

    words_for_bert = sorted(set(glove.vectors.keys()).intersection(target_words))
    print(f"Computing BERT embeddings on device={device} for {len(words_for_bert)} words…")

    import numpy as np

    for w in words_for_bert:
        contexts = merged_ctx[w].sentences
        # Skip if all layers already cached.
        cached_all = all(
            (layer_dirs[layer] / f"{safe_filename(w)}.npy").exists() and (w in bert_index_by_layer[layer])
            for layer in config.bert_layers
        )
        if cached_all:
            continue

        layer_sums: dict[int, np.ndarray] = {}
        n_ok = 0
        for sent in contexts:
            try:
                vecs = embedder.embed_word_in_sentence_layers(
                    sentence=sent, target_word=w, layers=config.bert_layers
                )
            except Exception:
                continue

            for layer, vec in vecs.items():
                if layer in layer_sums:
                    layer_sums[layer] = layer_sums[layer] + vec
                else:
                    layer_sums[layer] = vec.astype("float64", copy=False)
            n_ok += 1

        if n_ok == 0:
            continue

        bert_stats[w] = {"n_context_sentences": len(contexts), "n_successful": n_ok}

        for layer in config.bert_layers:
            mean_vec = (layer_sums[layer] / float(n_ok)).astype("float32", copy=False)
            out_path = save_vector(layer_dirs[layer], word=w, vector=mean_vec)
            bert_index_by_layer[layer][w] = StoredVector(word=w, path=out_path)

    for layer, items_by_word in bert_index_by_layer.items():
        write_index(layer_dirs[layer] / "index.json", items_by_word.values())
    write_json(
        bert_root / "meta.json",
        {
            "model": config.bert_model_name,
            "layers": list(config.bert_layers),
            "device": device,
        },
    )
    write_json(paths.results_dir / "bert_embedding_stats.json", bert_stats)
    print(f"Wrote embeddings under: {paths.embeddings_dir}")


def run_analyses(*, config: LabConfig, paths: ProjectPaths) -> None:  # pragma: no cover
    ensure_dir(paths.results_dir)
    ensure_dir(paths.figures_dir)

    from lab2.utils import safe_filename
    from lab2.analysis.load_embeddings import load_embeddings_from_index

    glove_root = paths.embeddings_dir / "glove" / safe_filename(config.glove_filename)
    glove_index = glove_root / "index.json"
    if not glove_index.exists():
        raise FileNotFoundError(f"Missing {glove_index}. Run `python -m lab2 compute-embeddings` first.")

    print("Loading GloVe embeddings…")
    glove_emb = load_embeddings_from_index(glove_index)

    bert_root = paths.embeddings_dir / "bert" / safe_filename(config.bert_model_name)
    bert_emb_by_layer: dict[int, dict[str, "numpy.ndarray"]] = {}
    for layer in config.bert_layers:
        idx = bert_root / f"layer_{layer:02d}" / "index.json"
        if not idx.exists():
            raise FileNotFoundError(f"Missing {idx}. Run `python -m lab2 compute-embeddings` first.")
        bert_emb_by_layer[layer] = load_embeddings_from_index(idx)

    # -----------------------
    # 2) Anisotropy (PDF §2)
    # -----------------------
    print("Running anisotropy analysis…")
    from lab2.analysis.anisotropy import run_anisotropy
    import numpy as np

    # Prefer the explicit random-word list if available.
    random_ctx_path = paths.contexts_dir / "random_contexts.jsonl"
    random_words: set[str] | None = None
    if random_ctx_path.exists():
        from lab2.data.contexts import read_contexts_jsonl

        random_words = set(read_contexts_jsonl(random_ctx_path).keys())

    common_words = set(glove_emb.keys())
    for layer in config.bert_layers:
        common_words &= set(bert_emb_by_layer[layer].keys())
    if random_words is not None:
        common_words &= random_words

    words = sorted(common_words)
    if len(words) < 10:
        print("Warning: not enough common words for anisotropy; skipping.")
        anisotropy_out = {}
    else:
        glove_mat = np.stack([glove_emb[w] for w in words])
        anisotropy_out = {
            "glove": {
                "static": run_anisotropy(
                    vectors=glove_mat,
                    model_label="GloVe",
                    layer_label="static",
                    pca_top_k=config.pca_top_k,
                    out_dir=paths.figures_dir,
                )
            },
            "bert": {},
        }
        for layer in config.bert_layers:
            bert_mat = np.stack([bert_emb_by_layer[layer][w] for w in words])
            anisotropy_out["bert"][f"layer_{layer}"] = run_anisotropy(
                vectors=bert_mat,
                model_label="BERT",
                layer_label=f"layer {layer}",
                pca_top_k=config.pca_top_k,
                out_dir=paths.figures_dir,
            )

    write_json(paths.results_dir / "anisotropy_metrics.json", anisotropy_out)

    # --------------------------
    # 3) Morphology (PDF §3)
    # --------------------------
    morph_out: dict = {}
    if paths.morph_families_path.exists():
        print("Running morphology analyses…")
        from lab2.data.morph_families import read_morph_families_tsv
        from lab2.analysis.morphology import run_morphology_suite

        families = read_morph_families_tsv(paths.morph_families_path)
        morph_fig_dir = paths.figures_dir / "morphology"
        ensure_dir(morph_fig_dir)

        morph_out["glove"] = run_morphology_suite(
            families=families,
            embeddings=glove_emb,
            seed=config.random_seed,
            deaniso_method=config.deaniso_method,
            deaniso_pcs=config.deaniso_pcs,
            figures_dir=morph_fig_dir,
            label="GloVe",
        )
        morph_out["bert"] = {}
        for layer in config.bert_layers:
            morph_out["bert"][f"layer_{layer}"] = run_morphology_suite(
                families=families,
                embeddings=bert_emb_by_layer[layer],
                seed=config.random_seed + layer,
                deaniso_method=config.deaniso_method,
                deaniso_pcs=config.deaniso_pcs,
                figures_dir=morph_fig_dir,
                label=f"BERT layer {layer}",
            )
    else:
        print(f"Skipping morphology: missing {paths.morph_families_path}")

    write_json(paths.results_dir / "morphology_results.json", morph_out)

    # ----------------------------
    # 4) Synonyms vs Antonyms (§4)
    # ----------------------------
    synant_out: dict = {}
    syn_pairs_path = paths.results_dir / "synonym_pairs.tsv"
    ant_pairs_path = paths.results_dir / "antonym_pairs.tsv"
    if syn_pairs_path.exists() and ant_pairs_path.exists():
        print("Running synonyms/antonyms analyses…")
        from lab2.analysis.synant import read_pairs_tsv, run_synant_suite

        syn_pairs = read_pairs_tsv(syn_pairs_path)
        ant_pairs = read_pairs_tsv(ant_pairs_path)
        synant_fig_dir = paths.figures_dir / "synant"
        ensure_dir(synant_fig_dir)

        synant_out["glove"] = run_synant_suite(
            embeddings=glove_emb,
            synonym_pairs=syn_pairs,
            antonym_pairs=ant_pairs,
            k=config.nearest_k,
            seed=config.random_seed,
            deaniso_method=config.deaniso_method,
            deaniso_pcs=config.deaniso_pcs,
            figures_dir=synant_fig_dir,
            label="GloVe",
        )
        synant_out["bert"] = {}
        for layer in config.bert_layers:
            synant_out["bert"][f"layer_{layer}"] = run_synant_suite(
                embeddings=bert_emb_by_layer[layer],
                synonym_pairs=syn_pairs,
                antonym_pairs=ant_pairs,
                k=config.nearest_k,
                seed=config.random_seed + layer,
                deaniso_method=config.deaniso_method,
                deaniso_pcs=config.deaniso_pcs,
                figures_dir=synant_fig_dir,
                label=f"BERT layer {layer}",
            )
    else:
        print("Skipping syn/ant: missing synonym_pairs.tsv or antonym_pairs.tsv (run build-synant).")

    write_json(paths.results_dir / "synant_results.json", synant_out)
    print(f"Done. Results in: {paths.results_dir}")
