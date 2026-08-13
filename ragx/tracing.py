"""One span per retrieval stage, carrying the latency that stage cost.

The frontier has two axes and only one of them comes from a formula. Recall is computed
against exact search; latency has to be measured, per stage, or "reranking is expensive" stays
an opinion. These spans are where the second axis comes from, which is why tracing is wired
before the API rather than after it.

The provider is held on an object rather than installed with `set_tracer_provider`, which can
only be called once per process. That one line is what lets every test own an in-memory
exporter instead of fighting over a global.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from .config import Settings
from .schemas import StageTiming


class Tracing:
    def __init__(self, provider: TracerProvider):
        self.provider = provider

    def tracer(self, name: str = "ragx"):
        return self.provider.get_tracer(name)

    def shutdown(self) -> None:
        self.provider.shutdown()


class StageClock:
    """Collects per-stage timings even when no exporter is configured.

    The timings are part of the API RESPONSE, not only of the trace: a caller deciding whether
    to ask for reranking needs to know what the last one cost, and it should not have to run a
    tracing backend to find out.
    """

    def __init__(self, tracing: Tracing | None = None):
        self.tracing = tracing
        self.stages: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str, **attrs):
        t0 = time.perf_counter()
        if self.tracing is None:
            yield None
        else:
            with self.tracing.tracer().start_as_current_span(f"ragx.{name}") as span:
                for k, v in attrs.items():
                    span.set_attribute(f"ragx.{k}", v)
                yield span
        ms = (time.perf_counter() - t0) * 1000
        self.stages.append(StageTiming(stage=name, ms=round(ms, 3)))

    @property
    def total_ms(self) -> float:
        return round(sum(s.ms for s in self.stages), 3)


def build(s: Settings, exporter=None) -> Tracing:
    provider = TracerProvider(resource=Resource.create({"service.name": s.service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif s.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=s.otlp_endpoint)))
    return Tracing(provider)
