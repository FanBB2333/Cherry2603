from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lab2.config import LabConfig
from lab2.paths import ProjectPaths
from lab2.pipeline import build_synant, compute_embeddings, prepare_corpus, run_analyses
from lab2.utils import ensure_dir, write_json


@dataclass(frozen=True)
class StepResult:
    name: str
    ok: bool
    seconds: float
    error: str | None = None


def _git_head(repo_root: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_anisotropy(obj: dict[str, Any] | None) -> str:
    if not obj:
        return "- (missing `anisotropy_metrics.json`)\n"

    lines: list[str] = []

    def pick(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        return {k: d.get(k) for k in keys}

    glove = obj.get("glove", {}).get("static", {}).get("raw", {})
    if glove:
        lines.append(f"- GloVe/raw: {pick(glove, ['n_vectors','n_pairs','mean_pairwise_cosine','cos_to_mean_mean'])}")

    bert = obj.get("bert", {})
    for layer_key, by_transform in bert.items():
        raw = by_transform.get("raw", {})
        if raw:
            lines.append(f"- BERT/{layer_key}/raw: {pick(raw, ['n_vectors','n_pairs','mean_pairwise_cosine','cos_to_mean_mean'])}")

    return "\n".join(lines) + "\n"


def _summarize_morphology(obj: dict[str, Any] | None) -> str:
    if not obj:
        return "- (missing `morphology_results.json`)\n"

    def grab_intra(o: dict[str, Any]) -> dict[str, Any] | None:
        return o.get("raw", {}).get("intra_vs_inter")

    lines: list[str] = []
    g = obj.get("glove")
    if g and grab_intra(g):
        intra = grab_intra(g)
        lines.append(
            f"- GloVe intra-vs-inter (raw): cliffs_delta={intra['cliffs_delta']} p={intra['p_mannwhitney']}"
        )

    for layer_key, o in (obj.get("bert") or {}).items():
        intra = grab_intra(o)
        if intra:
            lines.append(
                f"- BERT/{layer_key} intra-vs-inter (raw): cliffs_delta={intra['cliffs_delta']} p={intra['p_mannwhitney']}"
            )

    return "\n".join(lines) + "\n"


def _summarize_synant(obj: dict[str, Any] | None) -> str:
    if not obj:
        return "- (missing `synant_results.json`)\n"

    lines: list[str] = []
    g = obj.get("glove", {}).get("raw", {})
    if g.get("neighbourhood"):
        lines.append(f"- GloVe neighbourhood (raw): {g['neighbourhood']}")
    for layer_key, o in (obj.get("bert") or {}).items():
        raw = o.get("raw", {})
        if raw.get("neighbourhood"):
            lines.append(f"- BERT/{layer_key} neighbourhood (raw): {raw['neighbourhood']}")
    return "\n".join(lines) + "\n"


def _run_pytest(repo_root: Path) -> tuple[int, str]:
    try:
        res = subprocess.run(
            ["python", "-m", "pytest", "-q"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        return int(res.returncode), out.strip()
    except Exception as e:
        return 1, f"Failed to run pytest: {e}"


def _artifact_checks(paths: ProjectPaths, config: LabConfig) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, note: str = "") -> None:
        checks.append((label, ok, note))

    check("wikitext_word_freq.pkl", (paths.results_dir / "wikitext_word_freq.pkl").exists())
    check("morph_contexts.jsonl", (paths.contexts_dir / "morph_contexts.jsonl").exists())
    check("synonym_pairs.tsv", (paths.results_dir / "synonym_pairs.tsv").exists())
    check("antonym_pairs.tsv", (paths.results_dir / "antonym_pairs.tsv").exists())

    from lab2.utils import safe_filename

    glove_root = paths.embeddings_dir / "glove" / safe_filename(config.glove_filename)
    check("glove/index.json", (glove_root / "index.json").exists())

    bert_root = paths.embeddings_dir / "bert" / safe_filename(config.bert_model_name)
    for layer in config.bert_layers:
        check(f"bert/layer_{layer:02d}/index.json", (bert_root / f"layer_{layer:02d}" / "index.json").exists())

    check("anisotropy_metrics.json", (paths.results_dir / "anisotropy_metrics.json").exists())
    check("morphology_results.json", (paths.results_dir / "morphology_results.json").exists())
    check("synant_results.json", (paths.results_dir / "synant_results.json").exists())

    return checks


def generate_report(
    *,
    config: LabConfig,
    paths: ProjectPaths,
    out_path: Path,
    run_pipeline: bool = True,
    run_pytest: bool = True,
) -> None:
    ensure_dir(out_path.parent)

    steps: list[StepResult] = []
    if run_pipeline:
        for name, fn in [
            ("prepare-corpus", lambda: prepare_corpus(config=config, paths=paths)),
            ("build-synant", lambda: build_synant(config=config, paths=paths)),
            ("compute-embeddings", lambda: compute_embeddings(config=config, paths=paths)),
            ("run-analyses", lambda: run_analyses(config=config, paths=paths)),
        ]:
            t0 = time.time()
            try:
                fn()
                steps.append(StepResult(name=name, ok=True, seconds=time.time() - t0))
            except Exception as e:
                steps.append(StepResult(name=name, ok=False, seconds=time.time() - t0, error=str(e)))
                break

    pytest_rc = None
    pytest_out = None
    if run_pytest:
        pytest_rc, pytest_out = _run_pytest(paths.repo_root)

    # Summaries
    anis = _read_json_if_exists(paths.results_dir / "anisotropy_metrics.json")
    morph = _read_json_if_exists(paths.results_dir / "morphology_results.json")
    synant = _read_json_if_exists(paths.results_dir / "synant_results.json")

    checks = _artifact_checks(paths, config)
    head = _git_head(paths.repo_root)

    now = datetime.now().astimezone()
    md: list[str] = []
    md.append("# Lab2 Run & Test Report")
    md.append("")
    md.append(f"- Generated: `{now.strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    md.append(f"- Repo: `{paths.repo_root}`")
    if head:
        md.append(f"- Git HEAD: `{head}`")
    md.append(f"- Python: `{platform.python_version()}`")
    md.append(f"- Platform: `{platform.platform()}`")
    md.append("")

    md.append("## Pipeline")
    if steps:
        md.append("| step | status | seconds | error |")
        md.append("|---|---:|---:|---|")
        for s in steps:
            md.append(
                f"| `{s.name}` | {'OK' if s.ok else 'FAIL'} | {s.seconds:.1f} | {'' if s.ok else (s.error or '')} |"
            )
    else:
        md.append("- (skipped)")
    md.append("")

    md.append("## Checks")
    md.append("| item | ok | note |")
    md.append("|---|---:|---|")
    for label, ok, note in checks:
        md.append(f"| `{label}` | {'OK' if ok else 'FAIL'} | {note} |")
    md.append("")

    md.append("## Pytest")
    if pytest_rc is None:
        md.append("- (skipped)")
    else:
        md.append(f"- Exit code: `{pytest_rc}`")
        md.append("")
        md.append("```")
        md.append(pytest_out or "")
        md.append("```")
    md.append("")

    md.append("## Result Summary (raw)")
    md.append("")
    md.append("### Anisotropy")
    md.append(_summarize_anisotropy(anis))
    md.append("### Morphology")
    md.append(_summarize_morphology(morph))
    md.append("### Synonyms/Antonyms")
    md.append(_summarize_synant(synant))

    md.append("## Artifacts")
    md.append(f"- Results dir: `{paths.results_dir}`")
    md.append(f"- Figures dir: `{paths.figures_dir}`")
    md.append(f"- Embeddings dir: `{paths.embeddings_dir}`")
    md.append("")

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    write_json(paths.results_dir / "run_report.meta.json", {"report_path": str(out_path)})
    print(f"Wrote report: {out_path}")

