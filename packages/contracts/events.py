from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    RUN_STARTED = "run.started"
    MESSAGE_DELTA = "message.delta"
    REASONING_DELTA = "reasoning.delta"
    TOOL_REQUESTED = "tool.requested"
    TOOL_STARTED = "tool.started"
    TOOL_PROGRESS = "tool.progress"
    TOOL_FORESIGHT = "tool.foresight"
    TOOL_ROUTING = "tool.routing"
    TOOL_RESULT = "tool.result"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELED = "run.canceled"
    RUN_BUDGET_EXCEEDED = "run.budget_exceeded"
    APPROVAL_REQUIRED = "run.approval_required"
    AGENT_HANDOFF = "agent.handoff"
    CONTEXT_COMPRESSED = "context.compressed"
    USAGE_FINAL = "usage.final"


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    thread_id: UUID
    sequence: int = Field(ge=1)
    type: EventType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=1, ge=1)
