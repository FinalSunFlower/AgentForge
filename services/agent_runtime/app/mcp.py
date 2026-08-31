from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field


class MCPError(RuntimeError):
    pass


class MCPEndpointError(ValueError):
    pass


class MCPTool(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(min_length=1)
    description: str = ""
    inputSchema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})


def validate_mcp_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise MCPEndpointError("mcp_endpoint_must_be_http")
    if not parsed.hostname:
        raise MCPEndpointError("mcp_endpoint_host_required")
    return endpoint


def registry_name(remote_name: str, prefix: str = "mcp") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in remote_name)
    stem = cleaned.strip("._-") or "tool"
    return f"{prefix}.{stem}"[:120]


def tool_definition_from_registration(row: Any) -> Any:
    """Build a runtime ToolDefinition that calls the MCP endpoint after approval."""
    from .tools import ToolDefinition, ToolError

    class MCPArguments(BaseModel):
        model_config = ConfigDict(extra="allow")

    endpoint = getattr(row, "origin_uri", None)
    remote_name = str(getattr(row, "origin_name", None) or getattr(row, "name", "tool"))
    timeout = 10.0
    token = None

    async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
        if not endpoint:
            raise ToolError("mcp_endpoint_missing")
        client = MCPClient(endpoint, bearer_token=token, timeout_seconds=timeout)
        result = await client.call_tool(remote_name, arguments)
        if result.get("is_error"):
            raise ToolError("mcp_tool_error")
        return {"mcp": True, "remote_name": remote_name, "endpoint": endpoint, **result}

    return ToolDefinition(
        name=row.name,
        description=row.description or f"MCP tool {remote_name}",
        input_model=MCPArguments,
        risk=getattr(row, "risk_level", "medium") or "medium",
        timeout_seconds=timeout,
        execute=execute,
    )


@dataclass
class MCPClient:
    endpoint: str
    bearer_token: str | None = None
    timeout_seconds: float = 10.0
    client_factory: Callable[..., Any] = httpx.AsyncClient

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        async with self.client_factory(timeout=self.timeout_seconds) as client:
            response = await client.post(self.endpoint, headers=headers, json=body)
            response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise MCPError(str(payload["error"]))
        return payload.get("result") or {}

    async def list_tools(self) -> list[MCPTool]:
        result = await self._request("tools/list")
        return [MCPTool.model_validate(item) for item in result.get("tools", [])]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        return {
            "content": result.get("content", []),
            "is_error": bool(result.get("isError", False)),
        }
