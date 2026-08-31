from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from services.agent_runtime.app.foresight import (
    Action,
    HeuristicForesightPolicy,
    preview_calculator,
    simulate_calculator,
    simulate_retrieval,
    simulate_sql,
    simulate_tool,
)
from services.agent_runtime.app.hybrid_retrieval import EVAL_CORPUS
from services.core_api.app.main import app


def test_feed_creates_and_likes_public_posts() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Creator", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}
        post = client.post("/v1/posts", headers=headers, json={"body": "agent systems"})
        assert post.status_code == 201
        assert (
            client.post(f"/v1/posts/{post.json()['id']}/like", headers=headers).status_code == 204
        )
        assert client.get("/v1/feed", headers=headers).status_code == 200


@pytest.mark.asyncio
async def test_foresight_gate_uses_candidate_variance() -> None:
    policy = HeuristicForesightPolicy(uncertainty_threshold=0.2)
    low = await policy.decide([Action("a", 1.0), Action("b", 1.1)])
    high = await policy.decide([Action("a", 0.0), Action("b", 1.0)])
    assert low.should_predict is False
    assert high.should_predict is True
    assert preview_calculator("2 + 3") == 1.0
    assert preview_calculator("1 / 0") == 0.0
    ok = simulate_calculator({"expression": "12 * (3 + 4)"})
    assert ok.ok is True and ok.kind == "ast" and ok.predicted["value"] == 84.0
    denied = simulate_sql({"query": "DELETE FROM novels"})
    assert denied.ok is False and denied.kind == "sql_validate"
    allowed = simulate_sql({"query": "SELECT title FROM novels LIMIT 5"})
    assert allowed.ok is True and allowed.predicted["would_execute"] is True
    preview = simulate_retrieval({"query": "undo purchase", "limit": 3}, EVAL_CORPUS)
    assert preview.ok is True and preview.kind == "vector_preview"
    assert preview.predicted["top_ids"]
    generic = simulate_tool("unknown_tool", {})
    assert generic.kind == "uninformative_prior"
