from uuid import uuid4

from fastapi.testclient import TestClient

from services.core_api.app.db import SessionFactory
from services.core_api.app.main import app
from services.core_api.app.models import User


def test_mcp_tools_start_quarantined_and_require_admin_approval() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Admin", "password": password},
        )

        import asyncio

        async def promote() -> None:
            async with SessionFactory() as session:
                user = await session.scalar(
                    __import__("sqlalchemy").select(User).where(User.email == email)
                )
                assert user is not None
                user.role = "admin"
                await session.commit()

        asyncio.run(promote())
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}
        created = client.post(
            "/v1/admin/tools",
            headers=headers,
            json={
                "name": f"mcp-{uuid4()}",
                "version": "1",
                "source": "mcp",
                "description": "search",
                "input_schema": {"type": "object"},
                "risk_level": "medium",
            },
        )
        assert created.status_code == 201 and created.json()["status"] == "quarantined"
        approved = client.post(f"/v1/admin/tools/{created.json()['id']}/approve", headers=headers)
        assert approved.status_code == 200 and approved.json()["status"] == "approved"
