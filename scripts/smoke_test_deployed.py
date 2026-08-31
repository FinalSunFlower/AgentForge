"""Black-box smoke against a deployed Core API.

This is not part of the local pytest suite. It talks to a public URL and
requires a live runtime + cheap model.

CI must skip at the workflow `if` when SMOKE_BASE_URL is unset (grey
"skipped"), not by running this script and exiting 0. The local unset-env
path below is only for a manual invocation on a laptop.
"""

from __future__ import annotations

import json
import os
import sys
from uuid import uuid4

import httpx

TERMINAL = {
    "run.completed",
    "run.failed",
    "run.canceled",
    "run.budget_exceeded",
    "run.approval_required",
}


def stream_sse(client: httpx.Client, path: str, headers: dict[str, str]) -> list[dict]:
    events: list[dict] = []
    with client.stream("GET", path, headers=headers) as response:
        response.raise_for_status()
        buffer = ""
        for chunk in response.iter_text():
            buffer += chunk
            frames = buffer.split("\n\n")
            buffer = frames.pop() or ""
            for frame in frames:
                data = next(
                    (line[5:] for line in frame.splitlines() if line.startswith("data: ")), None
                )
                if not data:
                    continue
                parsed = json.loads(data)
                events.append(parsed)
                if parsed.get("type") in TERMINAL:
                    return events
    return events


def main() -> int:
    base_url = os.environ.get("SMOKE_BASE_URL", "").rstrip("/")
    if not base_url:
        print("SMOKE_BASE_URL is unset; skipping deployed smoke (nothing is claimed as live yet).")
        return 0

    password = "correct horse battery staple"
    email = f"smoke-{uuid4()}@example.com"
    client = httpx.Client(base_url=base_url, timeout=60.0)
    try:
        register = client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Smoke", "password": password},
        )
        register.raise_for_status()
        login = client.post("/v1/auth/login", json={"email": email, "password": password})
        login.raise_for_status()
        token = login.json()["access_token"]
        auth = {"Authorization": f"Bearer {token}"}

        agents = client.get("/v1/agents").json()
        agent_id = next(
            (row["id"] for row in agents if row.get("slug") == "default-assistant"), agents[0]["id"]
        )
        thread = client.post(
            "/v1/threads",
            headers=auth,
            json={
                "user_id": register.json()["id"],
                "agent_id": agent_id,
                "title": "Deployed smoke",
            },
        )
        thread.raise_for_status()
        run = client.post(
            f"/v1/threads/{thread.json()['id']}/runs",
            headers=auth,
            json={"user_id": register.json()["id"], "content": "Calculate 7*8"},
        )
        run.raise_for_status()
        events = stream_sse(client, f"/v1/runs/{run.json()['id']}/events", auth)
        types = {event.get("type") for event in events}
        if not (types & TERMINAL):
            print("smoke failed: no terminal SSE event", types, file=sys.stderr)
            return 1
        if "run.failed" in types:
            print("smoke failed: run.failed", events[-1], file=sys.stderr)
            return 1

        evals = client.get("/v1/evals/summary")
        evals.raise_for_status()
        payload = evals.json()
        if payload["zero_overlap"]["minilm_recall_at_3"] != 1.0:
            print("smoke failed: minilm_recall_at_3", payload["zero_overlap"], file=sys.stderr)
            return 1
    finally:
        client.close()

    print("smoke test passed against", base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
