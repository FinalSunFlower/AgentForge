from __future__ import annotations

import httpx
from sqlalchemy import select

from services.core_api.app.config import get_settings
from services.core_api.app.models import Run, RunStatus


async def dispatch_created_runs_once(session, *, batch_size: int = 50) -> int:
    settings = get_settings()
    rows = list(
        await session.scalars(
            select(Run)
            .where(Run.status == RunStatus.CREATED)
            .order_by(Run.created_at)
            .limit(batch_size)
        )
    )
    dispatched = 0
    async with httpx.AsyncClient(timeout=5) as client:
        for run in rows:
            try:
                response = await client.post(
                    f"{settings.agent_runtime_url}/internal/runs/{run.id}",
                    headers={"X-Runtime-Token": settings.runtime_internal_token},
                )
                response.raise_for_status()
                dispatched += 1
            except (httpx.HTTPError, OSError):
                continue
    return dispatched
