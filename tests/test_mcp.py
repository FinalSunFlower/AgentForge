import json

import httpx
import pytest

from services.agent_runtime.app.mcp import MCPClient


@pytest.mark.asyncio
async def test_mcp_client_discovers_and_calls_tools() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["method"] == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "search",
                                "description": "Search",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"content": [{"type": "text", "text": "ok"}]},
            },
        )

    transport = httpx.MockTransport(handler)
    client = MCPClient(
        "https://mcp.example.test",
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )
    tools = await client.list_tools()
    result = await client.call_tool("search", {"q": "agent"})
    assert tools[0].name == "search"
    assert result["is_error"] is False
    assert len(requests) == 2
