from uuid import uuid4

import pytest

from services.agent_runtime.app.tools import retrieval
from services.core_api.app.db import SessionFactory, init_db
from services.core_api.app.models import Chapter, Novel


@pytest.mark.asyncio
async def test_retrieval_returns_only_published_novel_sources() -> None:
    await init_db()
    async with SessionFactory() as session:
        session.add(
            Novel(
                title=f"Research {uuid4()}",
                author_name="Author",
                description="agent planning notes",
            )
        )
        await session.commit()
    result = await retrieval({"query": "planning", "limit": 5})
    assert result["results"]
    assert result["results"][0]["kind"] == "novel"
    assert result["results"][0]["source_id"]
    assert result["results"][0]["passage_id"]
    assert result["method"] == "hybrid_rrf_late_interaction"
    assert result["embedding"] == "all-MiniLM-L6-v2"
    assert result["reranker"] in {"late_interaction_maxsim", "rrf_selective"}


@pytest.mark.asyncio
async def test_retrieval_excludes_private_or_unpublished_chapters() -> None:
    async with SessionFactory() as session:
        unpublished = Novel(title=f"Draft {uuid4()}", author_name="Author", description="needle")
        private_novel = Novel(title=f"Private {uuid4()}", author_name="Author", description="other")
        session.add_all([unpublished, private_novel])
        await session.flush()
        session.add_all(
            [
                Chapter(
                    novel_id=unpublished.id,
                    number=1,
                    title="Draft",
                    content="needle",
                    is_published=True,
                ),
                Chapter(
                    novel_id=private_novel.id,
                    number=1,
                    title="Private",
                    content="needle",
                    is_published=True,
                    visibility="private",
                ),
            ]
        )
        unpublished.status = "draft"
        await session.commit()
    result = await retrieval({"query": "needle", "limit": 20})
    assert all(item["title"] not in {"Draft", "Private"} for item in result["results"])
