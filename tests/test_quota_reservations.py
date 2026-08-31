from uuid import uuid4

from fastapi.testclient import TestClient

from services.core_api.app.main import app


def test_quota_reservation_and_webhook_are_idempotent() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        user = client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Annotator", "password": password},
        ).json()
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": f"quota-{uuid4()}"}
        payload = {
            "type": "tokens",
            "product_ref": "tokens-100",
            "amount": 500,
            "unit": "token",
        }
        first = client.post("/v1/quota/reservations", headers=headers, json=payload)
        second = client.post("/v1/quota/reservations", headers=headers, json=payload)
        assert first.status_code == 201 and second.status_code == 201
        assert first.json()["id"] == second.json()["id"]
        webhook = {
            "event_id": f"evt-{uuid4()}",
            "reservation_id": first.json()["id"],
            "provider_ref": "tx-1",
            "success": True,
        }
        assert client.post("/v1/webhooks/test", json=webhook).status_code == 200
        assert (
            client.post("/v1/webhooks/test", json=webhook).json()["status"] == "duplicate_ignored"
        )
        assert (
            client.get(
                f"/v1/quota/reservations/{first.json()['id']}",
                headers={"Authorization": f"Bearer {token}"},
            ).json()["status"]
            == "granted"
        )
        assert user["id"]


def test_quota_reservation_rejects_reuse_of_key_for_different_product() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Annotator", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": f"quota-{uuid4()}"}
        first = client.post(
            "/v1/quota/reservations",
            headers=headers,
            json={
                "type": "tokens",
                "product_ref": "tokens-100",
                "amount": 500,
                "unit": "token",
            },
        )
        conflict = client.post(
            "/v1/quota/reservations",
            headers=headers,
            json={
                "type": "tokens",
                "product_ref": "tokens-500",
                "amount": 2000,
                "unit": "token",
            },
        )
        assert first.status_code == 201
        assert conflict.status_code == 409


def test_failed_webhook_does_not_revert_granted_reservation() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Annotator", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        auth = {"Authorization": f"Bearer {token}"}
        headers = {**auth, "Idempotency-Key": f"quota-{uuid4()}"}
        reservation = client.post(
            "/v1/quota/reservations",
            headers=headers,
            json={
                "type": "tokens",
                "product_ref": "tokens-100",
                "amount": 500,
                "unit": "token",
            },
        ).json()
        assert (
            client.post(
                "/v1/webhooks/test",
                json={
                    "event_id": f"evt-{uuid4()}",
                    "reservation_id": reservation["id"],
                    "provider_ref": "tx-ok",
                    "success": True,
                },
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/webhooks/test",
                json={
                    "event_id": f"evt-{uuid4()}",
                    "reservation_id": reservation["id"],
                    "provider_ref": "tx-fail",
                    "success": False,
                },
            ).status_code
            == 200
        )
        assert (
            client.get(f"/v1/quota/reservations/{reservation['id']}", headers=auth).json()["status"]
            == "granted"
        )
