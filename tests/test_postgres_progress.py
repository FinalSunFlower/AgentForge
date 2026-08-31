import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from services.core_api.app.models import AnnotationSync, Paper, PaperSection, User
from tests.pg_gate import make_session_factory, postgres_url


@pytest.mark.asyncio
async def test_postgres_concurrent_progress_writes_leave_one_row() -> None:
    """Force a first-write race: both SELECTs see no row before either INSERTs.

    FOR UPDATE on zero rows does not lock a future insert. The unique
    (user, paper) constraint plus IntegrityError retry is the real first-write
    gate. The barrier makes both observers see empty before either commits.
    """
    with postgres_url() as database_url:
        engine, factory = await make_session_factory(database_url)
        try:
            async with factory() as session:
                user = User(email=f"{uuid4()}@example.com", display_name="Reader")
                paper = Paper(title=f"Race {uuid4()}", author_name="Author", description="gate")
                session.add_all([user, paper])
                await session.flush()
                section = PaperSection(
                    paper_id=paper.id, number=1, title="One", content="content", is_published=True
                )
                session.add(section)
                await session.commit()
                user_id, paper_id, section_id = user.id, paper.id, section.id

            barrier = asyncio.Barrier(2)
            saw_empty: list[bool] = []
            integrity_hits = 0

            async def write(percent: int) -> None:
                nonlocal integrity_hits
                async with factory() as session:
                    query = (
                        select(AnnotationSync)
                        .where(
                            AnnotationSync.user_id == user_id, AnnotationSync.paper_id == paper_id
                        )
                        .with_for_update()
                    )
                    current = await session.scalar(query)
                    saw_empty.append(current is None)
                    await barrier.wait()
                    now = datetime.now(UTC)
                    if current is None:
                        session.add(
                            AnnotationSync(
                                user_id=user_id,
                                paper_id=paper_id,
                                section_id=section_id,
                                section_number=1,
                                progress_percent=percent,
                                paragraph_index=percent,
                                client_updated_at=now,
                            )
                        )
                    elif percent > current.progress_percent:
                        current.progress_percent = percent
                        current.paragraph_index = percent
                        current.client_updated_at = now
                    try:
                        await session.commit()
                    except IntegrityError:
                        integrity_hits += 1
                        await session.rollback()
                        current = await session.scalar(query)
                        assert current is not None
                        if percent > current.progress_percent:
                            current.progress_percent = percent
                            current.paragraph_index = percent
                            current.client_updated_at = datetime.now(UTC)
                        await session.commit()

            await asyncio.gather(write(30), write(80))
            assert saw_empty == [True, True]
            assert integrity_hits >= 1
            async with factory() as session:
                count = await session.scalar(
                    select(func.count())
                    .select_from(AnnotationSync)
                    .where(AnnotationSync.user_id == user_id, AnnotationSync.paper_id == paper_id)
                )
                assert session.bind is not None
                assert session.bind.dialect.name == "postgresql"
                assert int(count or 0) == 1
        finally:
            await engine.dispose()
