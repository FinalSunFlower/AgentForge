from uuid import uuid4

from fastapi.testclient import TestClient

from services.core_api.app.main import app


def test_notification_is_persisted_and_queued_for_authorized_device() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Notify", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}
        assert (
            client.post(
                "/v1/devices/push",
                headers=headers,
                json={"platform": "web", "token": "token-" + str(uuid4()), "permission": "granted"},
            ).status_code
            == 204
        )
        created = client.post(
            "/v1/notifications",
            headers=headers,
            json={"type": "run.completed", "payload": {"run_id": str(uuid4())}},
        )
        assert created.status_code == 201
        listed = client.get("/v1/notifications", headers=headers)
        assert listed.status_code == 200 and listed.json()[0]["type"] == "run.completed"
        assert (
            client.post(
                f"/v1/notifications/{created.json()['id']}/read", headers=headers
            ).status_code
            == 204
        )


def test_notification_preferences_disable_external_delivery_but_keep_inbox_fact() -> None:
    with TestClient(app) as client:
        email = f"prefs-{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Prefs", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}
        updated = client.put(
            "/v1/notifications/preferences",
            headers=headers,
            json={"enabled": False, "muted_types": [], "cooldown_seconds": 1800, "daily_cap": 3},
        )
        assert updated.status_code == 200 and updated.json()["enabled"] is False
        created = client.post(
            "/v1/notifications", headers=headers, json={"type": "run.completed", "payload": {}}
        )
        assert created.status_code == 201
        assert (
            client.get("/v1/notifications", headers=headers).json()[0]["id"] == created.json()["id"]
        )
