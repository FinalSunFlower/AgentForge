from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from services.agent_runtime.app.mcp import MCPClient, MCPTool, registry_name, validate_mcp_endpoint
from services.core_api.app.db import SessionFactory
from services.core_api.app.main import app
from services.core_api.app.models import AgentTool, RegisteredTool, User


def _admin_headers(client: TestClient) -> dict[str, str]:
    email = f"{uuid4()}@example.com"
    password = "correct horse battery staple"
    client.post(
        "/v1/auth/register", json={"email": email, "display_name": "Admin", "password": password}
    )

    import asyncio

    async def promote() -> None:
        async with SessionFactory() as session:
            user = await session.scalar(select(User).where(User.email == email))
            assert user is not None
            user.role = "admin"
            await session.commit()

    asyncio.run(promote())
    token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


def test_mcp_endpoint_rejects_non_http() -> None:
    with pytest.raises(ValueError):
        validate_mcp_endpoint("file:///tmp/tools")
    assert registry_name("search/docs", "mcp") == "mcp.search_docs"


def test_mcp_sync_quarantines_discovered_tools_until_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_list(self: MCPClient) -> list[MCPTool]:
        return [
            MCPTool(
                name="echo",
                description="Echo text",
                inputSchema={"type": "object", "properties": {"text": {"type": "string"}}},
            )
        ]

    monkeypatch.setattr(MCPClient, "list_tools", fake_list)
    with TestClient(app) as client:
        headers = _admin_headers(client)
        prefix = f"t{uuid4().hex[:8]}"
        synced = client.post(
            "/v1/admin/mcp/sync",
            headers=headers,
            json={
                "endpoint": "https://mcp.example.test/rpc",
                "risk_level": "medium",
                "name_prefix": prefix,
            },
        )
        assert synced.status_code == 200
        body = synced.json()
        assert body["discovered"] == 1, body
        assert len(body["created"]) == 1, body
        created = body["created"][0]
        assert created["status"] == "quarantined"
        assert created["source"] == "mcp"
        assert created["name"] == f"{prefix}.echo"
        assert created["origin_name"] == "echo"
        catalog = client.get("/v1/tools/catalog").json()
        assert all(row["name"] != created["name"] for row in catalog)
        approved = client.post(f"/v1/admin/tools/{created['id']}/approve", headers=headers)
        assert approved.status_code == 200 and approved.json()["status"] == "approved"
        again = client.post(
            "/v1/admin/mcp/sync",
            headers=headers,
            json={"endpoint": "https://mcp.example.test/rpc", "name_prefix": prefix},
        )
        assert again.json()["skipped"] == [created["name"]]


def test_approved_mcp_tool_can_be_attached_to_an_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_list(self: MCPClient) -> list[MCPTool]:
        return [MCPTool(name="ping", description="Ping", inputSchema={"type": "object"})]

    monkeypatch.setattr(MCPClient, "list_tools", fake_list)
    with TestClient(app) as client:
        headers = _admin_headers(client)
        prefix = f"t{uuid4().hex[:8]}"
        created = client.post(
            "/v1/admin/mcp/sync",
            headers=headers,
            json={"endpoint": "https://mcp.example.test/rpc", "name_prefix": prefix},
        ).json()["created"][0]
        client.post(f"/v1/admin/tools/{created['id']}/approve", headers=headers)
        agents = client.get("/v1/agents").json()
        assistant = next(row for row in agents if row["slug"] == "default-assistant")
        attached = client.post(
            f"/v1/admin/agents/{assistant['id']}/tools",
            headers=headers,
            json={"tool_ref": created["name"]},
        )
        assert attached.status_code == 201

    import asyncio

    async def check() -> None:
        async with SessionFactory() as session:
            row = await session.scalar(
                select(AgentTool).where(
                    AgentTool.agent_id == UUID(assistant["id"]),
                    AgentTool.tool_ref == created["name"],
                )
            )
            assert row is not None
            tool = await session.scalar(
                select(RegisteredTool).where(RegisteredTool.name == created["name"])
            )
            assert tool is not None and tool.status == "approved" and tool.origin_uri

    asyncio.run(check())
