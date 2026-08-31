from __future__ import annotations

import asyncio
import logging

from services.core_api.app.db import SessionFactory, init_db
from services.core_api.app.outbox import RedisNotConfigured, relay_once

from .billing import expire_pending_orders_once
from .push import dispatch_push_once
from .runtime_dispatch import dispatch_created_runs_once
from .summary import summarize_threads_once
from .usage import aggregate_usage_once

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentforge.worker")


async def run() -> None:
    await init_db()
    while True:
        async with SessionFactory() as session:
            try:
                count = await relay_once(session)
            except RedisNotConfigured:
                logger.warning("outbox relay skipped: REDIS_URL is not configured")
                count = 0
            if count:
                logger.info("published outbox events", extra={"count": count})
        async with SessionFactory() as session:
            usage_count = await aggregate_usage_once(session)
            if usage_count:
                logger.info("aggregated usage sessions", extra={"count": usage_count})
        async with SessionFactory() as session:
            push_count = await dispatch_push_once(session)
            if push_count:
                logger.info("processed push deliveries", extra={"count": push_count})
        async with SessionFactory() as session:
            expired = await expire_pending_orders_once(session)
            summaries = await summarize_threads_once(session)
            if expired or summaries:
                logger.info(
                    "ran maintenance tasks",
                    extra={"expired_orders": expired, "summaries": summaries},
                )
        async with SessionFactory() as session:
            dispatched = await dispatch_created_runs_once(session)
            if dispatched:
                logger.info("dispatched durable runs", extra={"count": dispatched})
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run())
