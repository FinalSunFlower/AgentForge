import asyncio
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from packages.contracts.events import EventType
from services.agent_runtime.app.executor import RunExecutor
from services.agent_runtime.app.provider import ModelTurn, ToolCallDelta
from services.core_api.app.db import SessionFactory
from services.core_api.app.main import app
from services.core_api.app.models import Message, MessageRole, Run, RunEvent


class ScriptedProvider:
    async def complete(self, messages, tools):
        if any(message.get("role") == "tool" for message in messages):
            return ModelTurn(text="The result is 5.", input_tokens=8, output_tokens=4)
        return ModelTurn(tool_calls=[ToolCallDelta("tc-1", "calculator", '{"expression":"2 + 3"}')])


def test_sse_replays_from_last_event_id_and_snapshot_lists_sql_events() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        user = client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "SSE", "password": password},
        ).json()
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}
        agent = client.get("/v1/agents").json()[0]
        thread = client.post(
            "/v1/threads",
            headers=headers,
            json={"user_id": user["id"], "agent_id": agent["id"], "title": "SSE"},
        ).json()

        async def _seed_and_run():
            async with SessionFactory() as session:
                session.add(
                    Message(thread_id=UUID(thread["id"]), role=MessageRole.USER, content="2 + 3")
                )
                run = Run(thread_id=UUID(thread["id"]))
                session.add(run)
                await session.commit()
                run_id = run.id
            await RunExecutor(lambda _unused: ScriptedProvider()).execute(run_id)
            return run_id

        run_id = asyncio.run(_seed_and_run())
        snapshot = client.get(f"/v1/runs/{run_id}/snapshot", headers=headers)
        assert snapshot.status_code == 200
        events = snapshot.json()["events"]
        assert events
        first_sequence = events[0]["sequence"]
        with client.stream(
            "GET",
            f"/v1/runs/{run_id}/events",
            headers={**headers, "Last-Event-ID": str(first_sequence)},
        ) as stream:
            payload = "".join(stream.iter_text())
        assert "event:" in payload
        assert EventType.USAGE_FINAL.value in payload or EventType.RUN_COMPLETED.value in payload

        async def _event_count() -> int:
            async with SessionFactory() as session:
                return len(
                    list(await session.scalars(select(RunEvent).where(RunEvent.run_id == run_id)))
                )

        assert asyncio.run(_event_count()) >= 2
