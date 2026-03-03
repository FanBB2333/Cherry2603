from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def glove_dir(self) -> Path:
        return self.data_dir / "glove"

    @property
    def nltk_dir(self) -> Path:
        return self.data_dir / "nltk_data"

    @property
    def hf_cache_dir(self) -> Path:
        return self.data_dir / "huggingface"

    @property
    def artifacts_dir(self) -> Path:
        return self.repo_root / "artifacts"

    @property
    def contexts_dir(self) -> Path:
        return self.artifacts_dir / "contexts"

    @property
    def embeddings_dir(self) -> Path:
        return self.artifacts_dir / "embeddings"

    @property
    def figures_dir(self) -> Path:
        return self.artifacts_dir / "figures"

    @property
    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"

    @property
    def morph_families_path(self) -> Path:
        return self.data_dir / "morph_families.tsv"

