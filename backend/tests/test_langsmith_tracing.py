from contextlib import contextmanager

from autoflow_scheduling.observability.langsmith_tracing import LangSmithTracer


class Run:
    def __init__(self, events):
        self.events = events

    def add_metadata(self, values):
        self.events.append(("metadata", values))

    def add_event(self, values):
        self.events.append(("run_event", values))


class Client:
    def __init__(self):
        self.events = []
        self.flushes = 0

    def flush(self):
        self.flushes += 1


class TraceFactory:
    def __init__(self, client):
        self.client = client

    @contextmanager
    def __call__(self, name, run_type, **values):
        self.client.events.append(("enter", {"name": name, "run_type": run_type, **values}))
        yield Run(self.client.events)
        self.client.events.append(("exit", name))


def test_missing_credentials_is_noop(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    client = Client()
    tracer = LangSmithTracer(client=client, trace_factory=TraceFactory(client))

    with tracer.root("root") as root:
        with tracer.stage("dense") as child:
            tracer.update(child, text="secret", candidates=list(range(100)))
    tracer.flush()

    assert root is None and child is None and client.events == [] and client.flushes == 0


def test_nested_runs_are_bounded_redacted_and_typed():
    client = Client()
    tracer = LangSmithTracer(client=client, enabled=True, trace_factory=TraceFactory(client))
    with tracer.root("query_evaluation", metadata={"query": "private"}):
        with tracer.stage("dense", metadata={"case_id": "case-1"}) as child:
            tracer.update(child, text="private", candidates=list(range(100)))
    tracer.flush()

    enters = [event[1] for event in client.events if event[0] == "enter"]
    assert enters[0]["run_type"] == "chain"
    assert enters[0]["metadata"]["query"] == "[redacted]"
    assert enters[1]["run_type"] == "retriever"
    metadata = [event[1] for event in client.events if event[0] == "metadata"]
    assert metadata == [{"text": "[redacted]", "candidates": list(range(20))}]
    assert client.flushes == 1


def test_error_and_fallback_create_diagnostic_events():
    client = Client()
    tracer = LangSmithTracer(client=client, enabled=True, trace_factory=TraceFactory(client))
    with tracer.root("root") as root:
        tracer.update(root, status="fallback", error_type="TimeoutError", error="offline")

    events = [event[1] for event in client.events if event[0] == "run_event"]
    assert events == [
        {"name": "fallback", "error_type": "TimeoutError", "message": "offline"}
    ]


def test_sdk_errors_and_flush_are_isolated():
    class BrokenClient:
        def flush(self):
            raise RuntimeError("offline")

    def broken_trace(*_args, **_kwargs):
        raise RuntimeError("offline")

    tracer = LangSmithTracer(client=BrokenClient(), enabled=True, trace_factory=broken_trace)
    with tracer.root("root") as root:
        assert root is None
    tracer.flush()
