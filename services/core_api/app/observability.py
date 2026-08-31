from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request, Response
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "trace_id",
            "user_id_hash",
            "thread_id",
            "run_id",
            "tool_name",
            "status",
            "latency_ms",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def configure_observability(app: FastAPI, *, service_name: str = "agentforge-core-api") -> None:
    global _configured
    if not _configured:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        trace.set_tracer_provider(provider)
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            root.addHandler(handler)
        root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        _configured = True

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz")
        HTTPXClientInstrumentor().instrument()
    except Exception:  # instrumentation must not prevent API startup
        logging.getLogger(__name__).exception("otel_fastapi_instrumentation_failed")

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or os.urandom(16).hex()
        request.state.request_id = request_id
        started = time.perf_counter()
        tracer = trace.get_tracer("agentforge.http")
        with tracer.start_as_current_span("http.request") as span:
            span.set_attribute("http.request_id", request_id)
            response = await call_next(request)
            context = span.get_span_context()
            trace_id = format(context.trace_id, "032x") if context.is_valid else request_id
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        logging.getLogger("agentforge.http").info(
            "http_request",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "user_id_hash": getattr(request.state, "user_id_hash", None),
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "thread_id": (
                    re.search(r"/threads/([0-9a-f-]+)", request.url.path) or [None, None]
                )[1],
                "run_id": (re.search(r"/runs/([0-9a-f-]+)", request.url.path) or [None, None])[1],
            },
        )
        return response


def get_tracer(name: str):
    return trace.get_tracer(name)
