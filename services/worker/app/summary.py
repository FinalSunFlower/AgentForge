from __future__ import annotations

from sqlalchemy import select

from services.core_api.app.models import Message, Thread


async def summarize_threads_once(session, *, max_messages: int = 40, max_chars: int = 2000) -> int:
    rows = list(await session.scalars(select(Thread).order_by(Thread.updated_at).limit(100)))
    changed = 0
    for thread in rows:
        messages = list(
            await session.scalars(
                select(Message)
                .where(Message.thread_id == thread.id)
                .order_by(Message.created_at.desc())
                .limit(max_messages)
            )
        )
        if len(messages) < max_messages:
            continue
        summary = "\n".join(f"{item.role.value}: {item.content}" for item in reversed(messages))[
            -max_chars:
        ]
        if thread.summary != summary:
            thread.summary = summary
            thread.summary_version += 1
            changed += 1
    if changed:
        await session.commit()
    return changed
