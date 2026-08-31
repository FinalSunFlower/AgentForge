from uuid import uuid4

from fastapi.testclient import TestClient

from services.core_api.app.main import app


def test_checkout_and_webhook_are_idempotent() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        user = client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Buyer", "password": password},
        ).json()
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": f"checkout-{uuid4()}"}
        payload = {
            "type": "credits",
            "product_ref": "credits-100",
            "amount": 500,
            "currency": "usd",
        }
        first = client.post("/v1/checkout", headers=headers, json=payload)
        second = client.post("/v1/checkout", headers=headers, json=payload)
        assert first.status_code == 201 and second.status_code == 201
        assert first.json()["id"] == second.json()["id"]
        webhook = {
            "event_id": f"evt-{uuid4()}",
            "order_id": first.json()["id"],
            "provider_tx_id": "tx-1",
            "success": True,
        }
        assert client.post("/v1/webhooks/test", json=webhook).status_code == 200
        assert (
            client.post("/v1/webhooks/test", json=webhook).json()["status"] == "duplicate_ignored"
        )
        assert (
            client.get(
                f"/v1/orders/{first.json()['id']}", headers={"Authorization": f"Bearer {token}"}
            ).json()["status"]
            == "fulfilled"
        )
        assert user["id"]


def test_checkout_rejects_reuse_of_key_for_different_product() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Buyer", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": f"checkout-{uuid4()}"}
        first = client.post(
            "/v1/checkout",
            headers=headers,
            json={
                "type": "credits",
                "product_ref": "credits-100",
                "amount": 500,
                "currency": "usd",
            },
        )
        conflict = client.post(
            "/v1/checkout",
            headers=headers,
            json={
                "type": "credits",
                "product_ref": "credits-500",
                "amount": 2000,
                "currency": "usd",
            },
        )
        assert first.status_code == 201
        assert conflict.status_code == 409


def test_failed_webhook_does_not_revert_fulfilled_order() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Buyer", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        auth = {"Authorization": f"Bearer {token}"}
        headers = {**auth, "Idempotency-Key": f"checkout-{uuid4()}"}
        order = client.post(
            "/v1/checkout",
            headers=headers,
            json={
                "type": "credits",
                "product_ref": "credits-100",
                "amount": 500,
                "currency": "usd",
            },
        ).json()
        assert (
            client.post(
                "/v1/webhooks/test",
                json={
                    "event_id": f"evt-{uuid4()}",
                    "order_id": order["id"],
                    "provider_tx_id": "tx-ok",
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
                    "order_id": order["id"],
                    "provider_tx_id": "tx-fail",
                    "success": False,
                },
            ).status_code
            == 200
        )
        assert client.get(f"/v1/orders/{order['id']}", headers=auth).json()["status"] == "fulfilled"
