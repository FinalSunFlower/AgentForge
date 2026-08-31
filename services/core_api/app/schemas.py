from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .models import MessageRole, RunStatus


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserCreate(StrictModel):
    email: str = Field(min_length=3, max_length=320)
    display_name: str = Field(min_length=1, max_length=120)


class RegisterRequest(UserCreate):
    password: str = Field(min_length=12, max_length=256)


class LoginRequest(StrictModel):
    email: str
    password: str


class TokenResponse(StrictModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: str | None = None


class RefreshRequest(StrictModel):
    refresh_token: str = Field(min_length=32, max_length=256)


class ApiKeyCreate(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    scopes: list[str] = Field(default_factory=list, max_length=20)


class ApiKeyRead(StrictModel):
    id: UUID
    name: str
    prefix: str
    scopes: list[str]
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKeyRead):
    key: str


class PaperCreate(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    author_name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=20_000)


class PaperRead(StrictModel):
    id: UUID
    title: str
    author_name: str
    description: str
    status: str


class SectionCreate(StrictModel):
    number: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=500_000)
    is_published: bool = True
    publish_at: datetime | None = None


class SectionRead(StrictModel):
    id: UUID
    paper_id: UUID
    number: int
    title: str
    content: str
    content_uri: str | None = None
    visibility: str = "public"
    is_published: bool
    publish_at: datetime | None


class AnnotationSyncUpdate(StrictModel):
    section_id: UUID
    section_number: int = Field(ge=1)
    progress_percent: int = Field(ge=0, le=100)
    paragraph_index: int = Field(ge=0)
    client_updated_at: datetime


class AnnotationSyncRead(StrictModel):
    accepted: bool
    stale_reason: str | None = None
    section_id: UUID | None = None
    section_number: int | None = None
    progress_percent: int | None = None
    paragraph_index: int | None = None


class CollectionRead(StrictModel):
    paper_id: UUID
    created_at: datetime


class QuotaGrantRequest(StrictModel):
    type: str = Field(pattern="^(plan|tokens)$")
    product_ref: str = Field(min_length=1, max_length=100)
    amount: int = Field(gt=0, le=10_000_000)
    unit: str = Field(default="token", min_length=3, max_length=16)


class ReservationRead(StrictModel):
    id: UUID
    type: str
    product_ref: str
    amount: int
    unit: str
    status: str
    expires_at: datetime


class EntitlementRead(StrictModel):
    id: UUID
    kind: str
    plan: str
    status: str
    period_start: datetime
    period_end: datetime


class WebhookQuotaGrant(StrictModel):
    event_id: str = Field(min_length=1, max_length=200)
    reservation_id: UUID
    provider_ref: str = Field(min_length=1, max_length=200)
    success: bool


class NotificationCreate(StrictModel):
    type: str = Field(min_length=1, max_length=48)
    payload: dict[str, object] = Field(default_factory=dict)


class NotificationRead(StrictModel):
    id: UUID
    type: str
    payload: dict[str, object]
    read_at: datetime | None
    created_at: datetime


class NotificationPreferenceUpdate(StrictModel):
    enabled: bool = True
    muted_types: list[str] = Field(default_factory=list, max_length=100)
    cooldown_seconds: int = Field(default=1800, ge=0, le=604800)
    daily_cap: int = Field(default=3, ge=0, le=1000)


class PushDeviceCreate(StrictModel):
    platform: str = Field(pattern="^(web|ios|android)$")
    token: str = Field(min_length=16, max_length=4096)
    permission: str = Field(default="granted", pattern="^(granted|denied|unknown)$")


class PostCreate(StrictModel):
    body: str = Field(min_length=1, max_length=20_000)
    model_ref: str | None = Field(default=None, max_length=120)
    hot_query: str | None = Field(default=None, max_length=200)


class PostRead(StrictModel):
    id: UUID
    author_id: UUID
    body: str
    quality_score: float
    like_count: int
    view_count: int
    created_at: datetime


class CommentCreate(StrictModel):
    body: str = Field(min_length=1, max_length=20_000)


class CommentRead(StrictModel):
    id: UUID
    post_id: UUID
    author_id: UUID
    body: str
    moderation_status: str
    created_at: datetime


class ToolRegistrationCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=40)
    source: str = Field(pattern="^(builtin|mcp|skill|expert)$")
    description: str = Field(default="", max_length=10_000)
    input_schema: dict[str, object]
    risk_level: str = Field(pattern="^(low|medium|high)$")


class ToolRegistrationRead(StrictModel):
    id: UUID
    name: str
    version: str
    source: str
    description: str
    risk_level: str
    status: str
    schema_hash: str
    origin_uri: str | None = None
    origin_name: str | None = None


class MCPSyncRequest(StrictModel):
    endpoint: str = Field(min_length=8, max_length=500)
    bearer_token: str | None = None
    risk_level: str = Field(default="medium", pattern="^(low|medium|high)$")
    name_prefix: str = Field(default="mcp", min_length=1, max_length=32)


class MCPSyncRead(StrictModel):
    endpoint: str
    discovered: int
    created: list[ToolRegistrationRead]
    skipped: list[str]


class AgentToolAttach(StrictModel):
    tool_ref: str = Field(min_length=1, max_length=120)


class UserRead(StrictModel):
    id: UUID
    email: str
    display_name: str
    status: str
    role: str
    plan: str


class AgentRead(StrictModel):
    id: UUID
    slug: str
    version: str
    model_ref: str
    status: str


class ThreadCreate(StrictModel):
    user_id: UUID
    agent_id: UUID
    title: str = Field(default="New conversation", min_length=1, max_length=200)


class ThreadRead(StrictModel):
    id: UUID
    user_id: UUID
    agent_id: UUID
    title: str
    created_at: datetime


class RunCreate(StrictModel):
    user_id: UUID
    content: str = Field(min_length=1, max_length=32_000)


class RunRead(StrictModel):
    id: UUID
    thread_id: UUID
    status: RunStatus
    terminal_reason: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    agent_version: str | None = None
    prompt_version: str | None = None
    model_ref: str | None = None
    tool_schema_hash: str | None = None


class MessageRead(StrictModel):
    id: UUID
    role: MessageRole
    content: str
    reasoning_content: str
    created_at: datetime


class ThreadDetail(ThreadRead):
    messages: list[MessageRead]
