"""Start a real PostgreSQL when Docker/CI did not provide one.

The required concurrency gate must exercise FOR UPDATE, not SQLite writer
serialization. This module prefers DATABASE_URL=postgresql+asyncpg; otherwise
it starts a pip-installed pgserver so the suite does not skip the gate.
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from services.core_api.app.db import Base


def _async_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    raise ValueError(f"unsupported postgres url: {url}")


def configured_postgres_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith(("postgresql+asyncpg://", "postgresql://")):
        return _async_url(url)
    return None


@contextmanager
def postgres_url() -> Iterator[str]:
    existing = configured_postgres_url()
    if existing:
        yield existing
        return
    import pgserver
    from pgserver._commands import initdb

    os.environ.setdefault("LC_ALL", "C")
    os.environ.setdefault("LANG", "C")
    root = Path(__file__).resolve().parents[1] / ".pg-gate"
    root.mkdir(exist_ok=True)
    data_dir = root / f"pg{os.getpid()}"
    if data_dir.exists():
        import shutil

        shutil.rmtree(data_dir)
    initdb(
        ["--auth=trust", "--auth-local=trust", "--encoding=UTF8", "--locale=C", "-U", "postgres"],
        pgdata=data_dir,
    )
    server = pgserver.get_server(str(data_dir), cleanup_mode="delete")
    try:
        yield _async_url(server.get_uri())
    finally:
        server.cleanup()


@contextmanager
def redis_url() -> Iterator[str]:
    existing = os.environ.get("REDIS_URL")
    if existing:
        yield existing
        return
    from fakeredis import TcpFakeServer

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = TcpFakeServer(("127.0.0.1", port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        server.shutdown()
        server.server_close()


async def make_session_factory(url: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    from services.core_api.app import models  # noqa: F401

    engine = create_async_engine(url, pool_pre_ping=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)
