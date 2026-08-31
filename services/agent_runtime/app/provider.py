from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCallDelta:
    tool_call_id: str
    name: str = ""
    arguments: str = ""
    index: int | None = None


@dataclass
class ModelTurn:
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    response_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class ModelProvider(Protocol):
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn: ...


class ProviderConfigurationError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str | None, model: str | None) -> None:
        if not api_key or not model:
            raise ProviderConfigurationError("LLM_API_KEY and LLM_MODEL are required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = True
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=body
            )
            response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        message = choice["message"]
        calls = [
            ToolCallDelta(
                tool_call_id=item["id"],
                name=item["function"]["name"],
                arguments=item["function"].get("arguments", "{}"),
            )
            for item in message.get("tool_calls", [])
        ]
        usage = payload.get("usage") or {}
        return ModelTurn(
            text=message.get("content") or "",
            reasoning=message.get("reasoning_content") or "",
            tool_calls=calls,
            response_id=payload.get("id"),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def complete_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ModelTurn]:
        """Yield OpenAI-compatible SSE deltas as they arrive from the provider."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body.update({"tools": tools, "tool_choice": "auto", "parallel_tool_calls": True})
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with (
            httpx.AsyncClient(timeout=90) as client,
            client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=body
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                choices = payload.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                calls = []
                for item in delta.get("tool_calls") or []:
                    function = item.get("function") or {}
                    calls.append(
                        ToolCallDelta(
                            tool_call_id=item.get("id", ""),
                            name=function.get("name", ""),
                            arguments=function.get("arguments", ""),
                            index=item.get("index"),
                        )
                    )
                usage = payload.get("usage") or {}
                yield ModelTurn(
                    text=delta.get("content") or "",
                    reasoning=delta.get("reasoning_content") or "",
                    tool_calls=calls,
                    response_id=payload.get("id"),
                    input_tokens=int(usage.get("prompt_tokens", 0)),
                    output_tokens=int(usage.get("completion_tokens", 0)),
                )
