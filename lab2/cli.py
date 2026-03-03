from __future__ import annotations

import argparse
from pathlib import Path

from lab2.config import LabConfig
from lab2.paths import ProjectPaths
from lab2.utils import ensure_dir, set_global_seed


def _repo_root_from_cwd() -> Path:
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab2", description="Lab 2 runner (BERT vs GloVe).")
    p.add_argument("--seed", type=int, default=LabConfig.random_seed, help="Random seed (default: 42).")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root_from_cwd(),
        help="Repo root (default: current working directory).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare-corpus", help="Download + clean WikiText and sample contexts.")
    sub.add_parser("build-synant", help="Build synonym/antonym pairs from WordNet.")
    sub.add_parser("compute-embeddings", help="Compute and cache GloVe + BERT embeddings.")
    sub.add_parser("run-analyses", help="Run all analyses (anisotropy/morphology/syn-ant).")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = LabConfig(random_seed=args.seed)
    paths = ProjectPaths(repo_root=args.repo_root)

    set_global_seed(config.random_seed)
    ensure_dir(paths.data_dir)
    ensure_dir(paths.artifacts_dir)

    # Lazy imports so `--help` stays fast even without ML deps installed.
    if args.cmd == "prepare-corpus":
        from lab2.pipeline import prepare_corpus

        prepare_corpus(config=config, paths=paths)
        return 0

    if args.cmd == "build-synant":
        from lab2.pipeline import build_synant

        build_synant(config=config, paths=paths)
        return 0

    if args.cmd == "compute-embeddings":
        from lab2.pipeline import compute_embeddings

        compute_embeddings(config=config, paths=paths)
        return 0

    if args.cmd == "run-analyses":
        from lab2.pipeline import run_analyses

        run_analyses(config=config, paths=paths)
        return 0

    raise AssertionError(f"Unhandled cmd: {args.cmd}")

