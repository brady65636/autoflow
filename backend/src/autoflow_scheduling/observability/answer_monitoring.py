"""Instrumentation boundary for the future answer-generation chain.

The repository does not generate answers yet. Call ``record_answer_quality`` after a
future generator has split its answer into claims and verified citation support.
"""

from __future__ import annotations

from typing import Any

from .langsmith_tracing import LangSmithTracer, get_tracer
from .quality_monitoring import answer_quality


def record_answer_quality(
    record: dict[str, Any],
    *,
    trace_id: str | None = None,
    prompt_version: str | None = None,
    tracer: LangSmithTracer | None = None,
) -> dict[str, float | int | bool]:
    """Compute and trace citation, grounding, refusal, token and cost metrics."""
    active_tracer = tracer or get_tracer()
    metrics = answer_quality(record)
    with active_tracer.stage(
        "answer_quality",
        metadata={"trace_id": trace_id, "prompt_version": prompt_version},
    ) as run:
        active_tracer.update(run, status="complete", **metrics)
    return metrics
