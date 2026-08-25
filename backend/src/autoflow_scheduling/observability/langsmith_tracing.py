"""Fault-isolated, privacy-conscious adapter for LangSmith tracing."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any, Callable, Iterator

from langsmith import Client
from langsmith.run_helpers import trace as _trace

from .monitoring_context import MonitoringContext

_LOG = logging.getLogger(__name__)
_MAX_ITEMS = 20
_MAX_STRING = 256
_SENSITIVE = {"text", "content", "document", "body", "pdf", "prompt", "output", "input", "query"}
_TRUTHY = {"1", "true", "yes", "on"}
_RETRIEVER_STAGES = {"parallel_retrieval", "parallel_case", "dense", "bm25", "rrf", "reranker"}
_AGENT_CHAIN_STAGES = {"agent_chat", "agent_graph", "agent_model"}


def _safe(value: Any, key: str = "") -> Any:
    """Keep telemetry bounded and never send document/query bodies."""
    if key.casefold() in _SENSITIVE:
        return "[redacted]"
    if isinstance(value, str):
        return value[:_MAX_STRING] + ("…" if len(value) > _MAX_STRING else "")
    if isinstance(value, dict):
        return {str(k): _safe(v, str(k)) for k, v in list(value.items())[:_MAX_ITEMS]}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in list(value)[:_MAX_ITEMS]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:_MAX_STRING]


def _enabled_from_environment() -> bool:
    tracing = os.getenv("LANGSMITH_TRACING", "").strip().casefold() in _TRUTHY
    return tracing and bool(os.getenv("LANGSMITH_API_KEY"))


def _stage_run_type(name: str) -> str:
    if name == "embedding":
        return "embedding"
    if name in _RETRIEVER_STAGES:
        return "retriever"
    if name in _AGENT_CHAIN_STAGES:
        return "chain"
    return "tool"


@lru_cache(maxsize=1)
def _shared_client() -> Client:
    """Share one background queue so the root CLI flush covers every nested run."""
    return Client()


class LangSmithTracer:
    """A no-op-by-default facade; LangSmith failures never break business work."""

    def __init__(
        self,
        client: Any = None,
        enabled: bool | None = None,
        trace_factory: Callable[..., Any] = _trace,
    ) -> None:
        self.enabled = _enabled_from_environment() if enabled is None else bool(enabled)
        self._client = client
        self._trace_factory = trace_factory
        self._project = os.getenv("LANGSMITH_PROJECT", "autoflow-rag")
        self._monitoring_context = MonitoringContext.from_environment()
        if self.enabled and self._client is None:
            try:
                self._client = _shared_client()
            except Exception:
                self.enabled = False
                _LOG.debug("LangSmith initialization failed", exc_info=True)

    @contextmanager
    def _run(
        self,
        name: str,
        *,
        run_type: str,
        metadata: dict[str, Any] | None,
    ) -> Iterator[Any]:
        if not self.enabled or self._client is None:
            with nullcontext(None) as run:
                yield run
            return
        try:
            context = self._trace_factory(
                name,
                run_type,
                inputs={},
                metadata=_safe(metadata or {}),
                project_name=self._project,
                client=self._client,
            )
            run = context.__enter__()
        except Exception:
            _LOG.debug("LangSmith run creation failed: %s", name, exc_info=True)
            yield None
            return
        try:
            yield run
        finally:
            try:
                context.__exit__(*sys.exc_info())
            except Exception:
                _LOG.debug("LangSmith run close failed: %s", name, exc_info=True)

    def root(self, name: str, *, metadata: dict[str, Any] | None = None) -> Any:
        dimensions = self._monitoring_context.metadata(**(metadata or {}))
        return self._run(name, run_type="chain", metadata=dimensions)

    def stage(self, name: str, *, metadata: dict[str, Any] | None = None) -> Any:
        return self._run(name, run_type=_stage_run_type(name), metadata=metadata)

    def update(self, run: Any, **values: Any) -> None:
        if run is None:
            return
        try:
            metadata = {key: _safe(value, key) for key, value in values.items()}
            run.add_metadata(metadata)
            status = str(values.get("status", "")).casefold()
            if status in {"error", "failed", "fallback"}:
                run.add_event(
                    {
                        "name": status,
                        "error_type": metadata.get("error_type"),
                        "message": metadata.get("error"),
                    }
                )
        except Exception:
            _LOG.debug("LangSmith run update failed", exc_info=True)

    def flush(self) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            _LOG.debug("LangSmith flush failed", exc_info=True)


def get_tracer() -> LangSmithTracer:
    return LangSmithTracer()
