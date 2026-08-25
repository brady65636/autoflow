"""Stable version dimensions attached to every root trace."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _release_version() -> str:
    explicit = os.getenv("AUTOFLOW_RELEASE_VERSION")
    if explicit:
        return explicit
    try:
        return version("autoflow-scheduling")
    except PackageNotFoundError:
        return "unknown"


def file_version(path: str | Path | None) -> str | None:
    """Return a short content version without putting corpus content in telemetry."""
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class MonitoringContext:
    environment: str
    release_version: str
    pipeline_version: str
    corpus_version: str | None
    index_version: str | None
    embedding_model: str | None
    reranker_mode: str
    reranker_model: str | None
    prompt_version: str | None

    @classmethod
    def from_environment(cls) -> "MonitoringContext":
        corpus_path = os.getenv("AUTOFLOW_CORPUS_MANIFEST")
        return cls(
            environment=os.getenv("AUTOFLOW_ENVIRONMENT", "dev"),
            release_version=_release_version(),
            pipeline_version=os.getenv("AUTOFLOW_PIPELINE_VERSION", "4"),
            corpus_version=(
                os.getenv("AUTOFLOW_CORPUS_VERSION") or file_version(corpus_path)
            ),
            index_version=os.getenv("AUTOFLOW_INDEX_VERSION"),
            embedding_model=os.getenv("AUTOFLOW_EMBEDDING_MODEL"),
            reranker_mode=os.getenv("AUTOFLOW_RERANKER_MODE", "none"),
            reranker_model=os.getenv("AUTOFLOW_RERANKER_MODEL"),
            prompt_version=os.getenv("AUTOFLOW_PROMPT_VERSION"),
        )

    def metadata(self, **request_dimensions: Any) -> dict[str, Any]:
        values = asdict(self)
        values.update(request_dimensions)
        return values
