from uuid import uuid4

from fastapi.testclient import TestClient

from services.core_api.app.main import app


def test_core_api_bootstraps_agent_and_thread() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        response = client.post(
            "/v1/auth/register",
            json={
                "email": email,
                "display_name": "Test",
                "password": "correct horse battery staple",
            },
        )
        assert response.status_code == 201
        user = response.json()

        response = client.post(
            "/v1/auth/login", json={"email": email, "password": "correct horse battery staple"}
        )
        assert response.status_code == 200
        headers = {"Authorization": f"Bearer {response.json()['access_token']}"}

        response = client.get("/v1/agents")
        assert response.status_code == 200
        slugs = {row["slug"] for row in response.json()}
        assert slugs == {
            "default-assistant",
            "supervisor",
            "science-specialist",
            "retrieval-specialist",
        }
        agent = next(row for row in response.json() if row["slug"] == "default-assistant")

        response = client.post(
            "/v1/threads",
            json={"user_id": user["id"], "agent_id": agent["id"], "title": "Test"},
            headers=headers,
        )
        assert response.status_code == 201
        thread = response.json()
        assert thread["user_id"] == user["id"]


def test_thread_creation_requires_bearer_token() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/threads",
            json={"user_id": str(uuid4()), "agent_id": str(uuid4()), "title": "Unauthorized"},
        )
        assert response.status_code == 401


def test_health_exposes_request_and_trace_correlation_headers() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz", headers={"X-Request-ID": "test-request-123"})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "test-request-123"
        assert len(response.headers["X-Trace-ID"]) == 32
