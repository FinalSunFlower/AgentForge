from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


@event.listens_for(engine.sync_engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    if engine.url.get_backend_name() == "sqlite":
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session


@asynccontextmanager
async def session_context() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session


async def init_db() -> None:
    from . import models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_upgrade_schema)


def _upgrade_schema(connection) -> None:
    table_additions = {
        "users": {
            "password_hash": "VARCHAR(256)",
            "role": "VARCHAR(32) DEFAULT 'user'",
            "plan": "VARCHAR(32) DEFAULT 'free'",
            "notification_preferences": "JSON",
        },
        "api_keys": {"name": "VARCHAR(80) DEFAULT 'default'"},
        "usage_sessions": {"aggregated_at": "DATETIME"},
        "posts": {
            "view_count": "INTEGER DEFAULT 0",
            "model_ref": "VARCHAR(120)",
            "hot_query": "VARCHAR(200)",
        },
        "chapters": {"content_uri": "VARCHAR(500)", "visibility": "VARCHAR(16) DEFAULT 'public'"},
        "runs": {
            "agent_version": "VARCHAR(40)",
            "prompt_version": "VARCHAR(40)",
            "model_ref": "VARCHAR(160)",
            "tool_schema_hash": "VARCHAR(128)",
            "input_summary": "VARCHAR(128)",
        },
        "agents": {"owner_id": "CHAR(36)"},
        "tool_calls": {
            "tool_version": "VARCHAR(40)",
            "schema_hash": "VARCHAR(128)",
            "policy_decision": "VARCHAR(32)",
            "duration_ms": "INTEGER",
            "output_summary": "VARCHAR(500)",
        },
        "registered_tools": {"origin_uri": "VARCHAR(500)", "origin_name": "VARCHAR(120)"},
    }
    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    for table, additions in table_additions.items():
        if table not in existing_tables:
            continue
        columns = {item["name"] for item in inspector.get_columns(table)}
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))
    if connection.dialect.name == "sqlite":
        connection.execute(text("PRAGMA user_version = 1"))
