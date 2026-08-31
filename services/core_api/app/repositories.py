from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Agent, Message, Thread, User
from .seed import CATALOG_SLUGS


async def get_user(session: AsyncSession, user_id: UUID) -> User | None:
    return await session.get(User, user_id)


async def get_agent(session: AsyncSession, agent_id: UUID) -> Agent | None:
    return await session.get(Agent, agent_id)


async def get_thread(session: AsyncSession, thread_id: UUID) -> Thread | None:
    return await session.get(Thread, thread_id)


async def list_agents(session: AsyncSession) -> list[Agent]:
    result = await session.scalars(
        select(Agent)
        .where(Agent.status == "active", Agent.slug.in_(CATALOG_SLUGS))
        .order_by(Agent.slug)
    )
    return list(result)


async def list_thread_messages(session: AsyncSession, thread_id: UUID) -> list[Message]:
    result = await session.scalars(
        select(Message)
        .where(Message.thread_id == thread_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(result)
