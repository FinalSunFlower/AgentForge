import hashlib
import secrets
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from services.core_api.app.auth import hash_password, verify_password
from services.core_api.app.db import SessionFactory
from services.core_api.app.main import app
from services.core_api.app.models import User


def test_new_password_hash_is_argon2id_and_verifies() -> None:
    encoded = hash_password("correct horse battery staple")
    assert encoded.startswith("$argon2id$")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)


def test_legacy_pbkdf2_hash_still_verifies() -> None:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", b"correct horse battery staple", salt, 310_000)
    encoded = f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)


def test_successful_legacy_login_migrates_hash() -> None:
    with TestClient(app) as client:
        email = f"legacy-{uuid4()}@example.com"
        password = "correct horse battery staple"
        response = client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Legacy", "password": password},
        )
        assert response.status_code == 201

        # Replace the newly-created hash with a legacy value to exercise migration.
        import asyncio

        async def replace_hash() -> None:
            async with SessionFactory() as session:
                user = await session.scalar(select(User).where(User.email == email))
                assert user is not None
                salt = secrets.token_bytes(16)
                digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
                user.password_hash = f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"
                await session.commit()

        asyncio.run(replace_hash())
        login = client.post("/v1/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200

        async def read_hash() -> str:
            async with SessionFactory() as session:
                user = await session.scalar(select(User).where(User.email == email))
                assert user is not None and user.password_hash is not None
                return user.password_hash

        assert asyncio.run(read_hash()).startswith("$argon2id$")


def test_refresh_tokens_rotate_and_api_keys_are_revocable() -> None:
    with TestClient(app) as client:
        email = f"tokens-{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Tokens", "password": password},
        )
        login = client.post("/v1/auth/login", json={"email": email, "password": password}).json()
        assert login["refresh_token"]
        rotated = client.post("/v1/auth/refresh", json={"refresh_token": login["refresh_token"]})
        assert (
            rotated.status_code == 200 and rotated.json()["refresh_token"] != login["refresh_token"]
        )
        assert (
            client.post(
                "/v1/auth/refresh", json={"refresh_token": login["refresh_token"]}
            ).status_code
            == 401
        )
        headers = {"Authorization": f"Bearer {rotated.json()['access_token']}"}
        created = client.post(
            "/v1/auth/api-keys", headers=headers, json={"name": "ci", "scopes": ["runs:read"]}
        )
        assert created.status_code == 201
        raw_key = created.json()["key"]
        assert (
            client.get("/v1/auth/me", headers={"Authorization": f"Bearer {raw_key}"}).status_code
            == 200
        )
        assert (
            client.delete(f"/v1/auth/api-keys/{created.json()['id']}", headers=headers).status_code
            == 204
        )
        assert (
            client.get("/v1/auth/me", headers={"Authorization": f"Bearer {raw_key}"}).status_code
            == 401
        )
