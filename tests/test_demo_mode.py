from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from services.agent_runtime.app.tools import builtin_tools, catalog_for_run
from services.core_api.app.config import get_settings
from services.core_api.app.db import SessionFactory, init_db
from services.core_api.app.demo import (
    allow_ip,
    check_demo_budget,
    get_global_usage_today,
    reset_demo_rate_limiter,
)
from services.core_api.app.main import app
from services.core_api.app.models import UsageDaily, User


@pytest.fixture
def demo_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "100")
    monkeypatch.setenv("DEMO_RUNS_PER_IP_PER_HOUR", "2")
    get_settings.cache_clear()
    reset_demo_rate_limiter()
    yield
    get_settings.cache_clear()
    reset_demo_rate_limiter()


def _auth_headers(client: TestClient) -> tuple[dict[str, str], dict]:
    email = f"{uuid4()}@example.com"
    password = "correct horse battery staple"
    user = client.post(
        "/v1/auth/register",
        json={"email": email, "display_name": "Demo", "password": password},
    ).json()
    token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}, user


@pytest.mark.asyncio
async def test_global_usage_sums_all_users_not_one_account() -> None:
    await init_db()
    today = datetime.now(UTC).date().isoformat()
    async with SessionFactory() as session:
        first = User(email=f"{uuid4()}@example.com", display_name="A")
        second = User(email=f"{uuid4()}@example.com", display_name="B")
        session.add_all([first, second])
        await session.flush()
        before = await get_global_usage_today(session)
        session.add_all(
            [
                UsageDaily(user_id=first.id, day=today, runs=1, input_tokens=40, output_tokens=10),
                UsageDaily(user_id=second.id, day=today, runs=2, input_tokens=30, output_tokens=20),
            ]
        )
        await session.flush()
        usage = await get_global_usage_today(session)
        assert usage.total_tokens == before.total_tokens + 100
        assert usage.runs == before.runs + 3
        await session.rollback()


@pytest.mark.asyncio
async def test_demo_budget_uses_global_daily_total(demo_on) -> None:
    await init_db()
    today = datetime.now(UTC).date().isoformat()
    async with SessionFactory() as session:
        user = User(email=f"{uuid4()}@example.com", display_name="Budget")
        session.add(user)
        await session.flush()
        session.add(
            UsageDaily(user_id=user.id, day=today, runs=1, input_tokens=80, output_tokens=20)
        )
        await session.flush()
        assert await check_demo_budget(session) is False
        await session.rollback()


@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_is_per_ip() -> None:
    reset_demo_rate_limiter()
    assert await allow_ip("1.1.1.1", limit=2, window_seconds=60)
    assert await allow_ip("1.1.1.1", limit=2, window_seconds=60)
    assert await allow_ip("1.1.1.1", limit=2, window_seconds=60) is False
    assert await allow_ip("2.2.2.2", limit=2, window_seconds=60)
    reset_demo_rate_limiter()


def test_demo_mode_hides_high_risk_tools() -> None:
    tools = builtin_tools()
    allowed = set(tools)
    visible = catalog_for_run(tools, allowed, demo_mode=True)
    assert "readonly_sql" not in visible
    assert tools["readonly_sql"].risk == "high"
    assert "calculator" in catalog_for_run(tools, allowed, demo_mode=False)
    assert "readonly_sql" in catalog_for_run(tools, allowed, demo_mode=False)


def test_create_run_demo_budget_uses_existing_terminal(
    demo_on, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def deny(_session) -> bool:
        return False

    monkeypatch.setattr("services.core_api.app.main.check_demo_budget", deny)
    with TestClient(app) as client:
        headers, user = _auth_headers(client)
        agent = client.get("/v1/agents").json()[0]
        thread = client.post(
            "/v1/threads",
            headers=headers,
            json={"user_id": user["id"], "agent_id": agent["id"], "title": "Budget"},
        ).json()
        response = client.post(
            f"/v1/threads/{thread['id']}/runs",
            headers=headers,
            json={"user_id": user["id"], "content": "Calculate 7*8"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "budget_exceeded"
        assert body["terminal_reason"] == "demo_daily_budget"


def test_create_run_demo_ip_limit_returns_429(demo_on, monkeypatch: pytest.MonkeyPatch) -> None:
    async def block(_ip: str) -> bool:
        return False

    monkeypatch.setattr("services.core_api.app.main.enforce_demo_ip_limit", block)
    with TestClient(app) as client:
        headers, user = _auth_headers(client)
        agent = client.get("/v1/agents").json()[0]
        thread = client.post(
            "/v1/threads",
            headers=headers,
            json={"user_id": user["id"], "agent_id": agent["id"], "title": "Limit"},
        ).json()
        response = client.post(
            f"/v1/threads/{thread['id']}/runs",
            headers=headers,
            json={"user_id": user["id"], "content": "Calculate 1+1"},
        )
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "demo_ip_rate_limited"


def test_evals_summary_is_public_and_has_minilm_alias() -> None:
    with TestClient(app) as client:
        response = client.get("/v1/evals/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["zero_overlap"]["minilm_recall_at_3"] == 1.0
        assert body["zero_overlap"]["keyword"]["recall_at_k"] == 0.0
        assert body["needle"]["survives_compression"] is True
        assert "retrieval" in body
        assert "memory" in body
        assert body["routing"]["keyword"]["eval_kind"] == "deterministic_keyword_router"
        assert body["routing"]["embedding"]["eval_kind"] == "minilm_embedding_router"
        assert "hybrid_late_interaction" in body["retrieval"]
        assert "hybrid_cross_encoder" in body["retrieval"]
        assert body["routing"]["hard_keyword"]["suite"] == "hard"
        assert "hard_retrieval" in body
        assert body["source"] in {"snapshot", "live"}
