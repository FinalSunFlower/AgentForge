from fastapi.testclient import TestClient
from sqlalchemy import func, select

from services.core_api.app.db import SessionFactory
from services.core_api.app.main import app
from services.core_api.app.models import Chapter, Novel
from services.core_api.app.routers.evals import load_evals_snapshot
from services.core_api.app.seed import DEMO_NOVEL_TITLE


def test_public_status_never_leaks_secrets() -> None:
    with TestClient(app) as client:
        response = client.get("/v1/status")
        assert response.status_code == 200
        body = response.json()
        assert body["api"] == "ok"
        assert "llm_configured" in body
        dumped = response.text.lower()
        assert "sk-" not in dumped
        assert "api_key" not in dumped


def test_demo_corpus_is_seeded_for_retrieval() -> None:
    import asyncio

    async def count_rows() -> tuple[int, int]:
        async with SessionFactory() as session:
            novel = await session.scalar(select(Novel).where(Novel.title == DEMO_NOVEL_TITLE))
            if novel is None:
                return 0, 0
            chapters = await session.scalar(
                select(func.count()).select_from(Chapter).where(Chapter.novel_id == novel.id)
            )
            return 1, int(chapters or 0)

    with TestClient(app) as client:
        assert client.get("/v1/status").status_code == 200
        novels, chapters = asyncio.run(count_rows())
    assert novels == 1
    assert chapters >= 9


def test_evals_summary_serves_checked_in_snapshot() -> None:
    snapshot = load_evals_snapshot()
    assert snapshot is not None
    with TestClient(app) as client:
        response = client.get("/v1/evals/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "snapshot"
        assert body["retrieval"]["hybrid_cross_encoder"]["queries"] == 19
        assert (
            body["routing"]["hard_keyword"]["task_success_rate"]
            < body["routing"]["keyword"]["task_success_rate"]
        )
        assert body["hard_retrieval"]["keyword"]["recall_at_k"] == 0.0
