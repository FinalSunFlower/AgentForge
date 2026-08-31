from __future__ import annotations

import asyncio
import secrets
import sys
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, status

from services.core_api.app.observability import configure_observability

from .config import get_settings
from .db import engine, init_db
from .executor import RunExecutor
from .provider import OpenAICompatibleProvider


def provider_factory(_thread):
    settings = get_settings()
    return OpenAICompatibleProvider(settings.llm_base_url, settings.llm_api_key, settings.llm_model)


executor = RunExecutor(provider_factory)
_scheduled: set[UUID] = set()
_schedule_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield
    if "pytest" not in sys.modules:
        await engine.dispose()


app = FastAPI(title="AgentForge Agent Runtime", version="0.1.0", lifespan=lifespan)
configure_observability(app, service_name="agentforge-agent-runtime")


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_configured": bool(settings.llm_api_key and settings.llm_model),
    }


@app.post("/internal/runs/{run_id}", status_code=202)
async def dispatch_run(
    run_id: UUID, x_runtime_token: str | None = Header(default=None, alias="X-Runtime-Token")
) -> dict[str, str]:
    expected = get_settings().runtime_internal_token
    if not x_runtime_token or not secrets.compare_digest(x_runtime_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="runtime_authentication_required"
        )
    async with _schedule_lock:
        if run_id in _scheduled:
            return {"run_id": str(run_id), "status": "already_scheduled"}
        _scheduled.add(run_id)

    async def run_and_release() -> None:
        try:
            await executor.execute(run_id)
        finally:
            async with _schedule_lock:
                _scheduled.discard(run_id)

    asyncio.create_task(run_and_release())
    return {"run_id": str(run_id), "status": "scheduled"}
