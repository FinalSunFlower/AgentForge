from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    BUDGET_EXCEEDED = "budget_exceeded"
    APPROVAL_REQUIRED = "approval_required"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class BaseEntity(Base):
    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class User(BaseEntity):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str | None] = mapped_column(String(256), nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="user")
    plan: Mapped[str] = mapped_column(String(32), default="free")
    status: Mapped[str] = mapped_column(String(32), default="active")
    notification_preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    threads: Mapped[list[Thread]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserSession(BaseEntity):
    __tablename__ = "user_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    refresh_token_hash: Mapped[str] = mapped_column(String(128), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ApiKey(BaseEntity):
    __tablename__ = "api_keys"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(80), default="default")
    key_hash: Mapped[str] = mapped_column(String(128), unique=True)
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Agent(BaseEntity):
    __tablename__ = "agents"

    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    version: Mapped[str] = mapped_column(String(40), default="1.0.0")
    system_policy: Mapped[str] = mapped_column(Text)
    model_ref: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(32), default="active")


class AgentTool(BaseEntity):
    __tablename__ = "agent_tools"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), index=True
    )
    tool_ref: Mapped[str] = mapped_column(String(120), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    policy_override: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    __table_args__ = (UniqueConstraint("agent_id", "tool_ref", name="uq_agent_tool_ref"),)


class AgentCapability(BaseEntity):
    __tablename__ = "agent_capabilities"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), index=True
    )
    capability: Mapped[str] = mapped_column(String(120))
    risk_level: Mapped[str] = mapped_column(String(16))
    quota_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    __table_args__ = (UniqueConstraint("agent_id", "capability", name="uq_agent_capability"),)


class Thread(BaseEntity):
    __tablename__ = "threads"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_version: Mapped[int] = mapped_column(Integer, default=0)
    user: Mapped[User] = relationship(back_populates="threads")
    agent: Mapped[Agent] = relationship()
    messages: Mapped[list[Message]] = relationship(
        back_populates="thread", cascade="all, delete-orphan"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="thread", cascade="all, delete-orphan")


class Run(BaseEntity):
    __tablename__ = "runs"

    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("threads.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus), default=RunStatus.CREATED, index=True
    )
    response_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    terminal_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    agent_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    model_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    tool_schema_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    input_summary: Mapped[str | None] = mapped_column(String(128), nullable=True)
    thread: Mapped[Thread] = relationship(back_populates="runs")
    messages: Mapped[list[Message]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    events: Mapped[list[RunEvent]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Message(BaseEntity):
    __tablename__ = "messages"

    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("threads.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole))
    content: Mapped[str] = mapped_column(Text, default="")
    reasoning_content: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[int] = mapped_column(Integer, default=1)
    thread: Mapped[Thread] = relationship(back_populates="messages")
    run: Mapped[Run | None] = relationship(back_populates="messages")


class ToolCall(BaseEntity):
    __tablename__ = "tool_calls"

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    tool_call_id: Mapped[str] = mapped_column(String(200), index=True)
    tool_name: Mapped[str] = mapped_column(String(120))
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    output_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="requested")
    depth: Mapped[int] = mapped_column(Integer, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    tool_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    schema_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    policy_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(String(500), nullable=True)
    run: Mapped[Run] = relationship(back_populates="tool_calls")
    __table_args__ = (UniqueConstraint("run_id", "tool_call_id", name="uq_tool_call_per_run"),)


class RunEvent(BaseEntity):
    __tablename__ = "run_events"

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    event_id: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    run: Mapped[Run] = relationship(back_populates="events")
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_run_event_sequence"),)


class UsageSession(BaseEntity):
    __tablename__ = "usage_sessions"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), unique=True
    )
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_micros: Mapped[int] = mapped_column(Integer, default=0)
    aggregated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExpertUsage(BaseEntity):
    __tablename__ = "expert_usage"

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    expert: Mapped[str] = mapped_column(String(80), index=True)
    model_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    executed: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens: Mapped[int] = mapped_column(Integer, default=0)


class OutboxEvent(BaseEntity):
    __tablename__ = "outbox_events"

    aggregate_type: Mapped[str] = mapped_column(String(80), index=True)
    aggregate_id: Mapped[str] = mapped_column(String(100), index=True)
    event_type: Mapped[str] = mapped_column(String(100), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Paper(BaseEntity):
    __tablename__ = "papers"

    title: Mapped[str] = mapped_column(String(200))
    author_name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="published", index=True)
    sections: Mapped[list[PaperSection]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class PaperSection(BaseEntity):
    __tablename__ = "paper_sections"

    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    content_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    visibility: Mapped[str] = mapped_column(String(16), default="public", index=True)
    is_published: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    publish_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paper: Mapped[Paper] = relationship(back_populates="sections")
    __table_args__ = (UniqueConstraint("paper_id", "number", name="uq_section_number"),)


class CollectionEntry(BaseEntity):
    __tablename__ = "collection_entries"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    __table_args__ = (UniqueConstraint("user_id", "paper_id", name="uq_collection_user_paper"),)


class AnnotationSync(BaseEntity):
    __tablename__ = "annotation_sync"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("paper_sections.id"))
    section_number: Mapped[int] = mapped_column(Integer)
    progress_percent: Mapped[int] = mapped_column(Integer)
    paragraph_index: Mapped[int] = mapped_column(Integer, default=0)
    client_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        UniqueConstraint("user_id", "paper_id", name="uq_annotation_user_paper"),
        CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100", name="ck_progress_percent"
        ),
        CheckConstraint("section_number >= 1", name="ck_annotation_section_number"),
    )


class RegisteredTool(BaseEntity):
    __tablename__ = "registered_tools"

    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    version: Mapped[str] = mapped_column(String(40))
    source: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(Text, default="")
    input_schema: Mapped[dict[str, Any]] = mapped_column(JSON)
    risk_level: Mapped[str] = mapped_column(String(16), default="medium")
    status: Mapped[str] = mapped_column(String(24), default="quarantined", index=True)
    schema_hash: Mapped[str] = mapped_column(String(128))
    approved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    origin_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    origin_name: Mapped[str | None] = mapped_column(String(120), nullable=True)


class AuditLog(BaseEntity):
    __tablename__ = "audit_logs"

    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(120), index=True)
    resource: Mapped[str] = mapped_column(String(120), index=True)
    resource_id: Mapped[str] = mapped_column(String(120), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class UsageDaily(BaseEntity):
    __tablename__ = "usage_daily"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    day: Mapped[str] = mapped_column(String(10), index=True)
    runs: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_micros: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (UniqueConstraint("user_id", "day", name="uq_usage_daily_user_day"),)


class ReservationStatus(StrEnum):
    CREATED = "created"
    AWAITING_CONFIRM = "awaiting_confirm"
    CONFIRMED = "confirmed"
    GRANTING = "granting"
    GRANTED = "granted"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REVERSED = "reversed"


class QuotaReservation(BaseEntity):
    __tablename__ = "quota_reservations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String(32))
    product_ref: Mapped[str] = mapped_column(String(100))
    amount: Mapped[int] = mapped_column(Integer)
    unit: Mapped[str] = mapped_column(String(16), default="token")
    status: Mapped[ReservationStatus] = mapped_column(
        Enum(ReservationStatus), default=ReservationStatus.CREATED, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (CheckConstraint("amount > 0", name="ck_reservation_amount_positive"),)


class QuotaGrantAttempt(BaseEntity):
    __tablename__ = "quota_grant_attempts"

    reservation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("quota_reservations.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    provider_ref: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    raw_ref: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class WebhookEvent(BaseEntity):
    __tablename__ = "webhook_events"

    provider: Mapped[str] = mapped_column(String(32))
    event_id: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(24), default="received")
    __table_args__ = (UniqueConstraint("provider", "event_id", name="uq_webhook_provider_event"),)


class Entitlement(BaseEntity):
    __tablename__ = "entitlements"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32))
    plan: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(24), default="active")
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reservation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("quota_reservations.id"), unique=True
    )


class TokenLedger(BaseEntity):
    __tablename__ = "token_ledger"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    delta: Mapped[int] = mapped_column(Integer)
    balance_after: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(120))
    reservation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("quota_reservations.id"), nullable=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True, index=True)


class Notification(BaseEntity):
    __tablename__ = "notifications"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String(48), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PushDevice(BaseEntity):
    __tablename__ = "push_devices"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    platform: Mapped[str] = mapped_column(String(16))
    token_hash: Mapped[str] = mapped_column(String(128), unique=True)
    permission: Mapped[str] = mapped_column(String(16), default="unknown")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class PushDelivery(BaseEntity):
    __tablename__ = "push_deliveries"

    notification_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("notifications.id", ondelete="CASCADE"), index=True
    )
    device_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("push_devices.id", ondelete="CASCADE"), index=True
    )
    channel: Mapped[str] = mapped_column(String(16), default="simulated")
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    collapse_key: Mapped[str | None] = mapped_column(String(120), nullable=True)
    provider_message_id: Mapped[str | None] = mapped_column(String(200), nullable=True)


class Post(BaseEntity):
    __tablename__ = "posts"

    author_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    body: Mapped[str] = mapped_column(Text)
    visibility: Mapped[str] = mapped_column(String(16), default="public", index=True)
    moderation_status: Mapped[str] = mapped_column(String(24), default="approved", index=True)
    quality_score: Mapped[float] = mapped_column(default=0.0)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    model_ref: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    hot_query: Mapped[str | None] = mapped_column(String(200), nullable=True)


class PostLike(BaseEntity):
    __tablename__ = "post_likes"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    post_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("posts.id", ondelete="CASCADE"), index=True
    )
    __table_args__ = (UniqueConstraint("user_id", "post_id", name="uq_post_like_user_post"),)


class Comment(BaseEntity):
    __tablename__ = "comments"

    post_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("posts.id", ondelete="CASCADE"), index=True
    )
    author_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    body: Mapped[str] = mapped_column(Text)
    moderation_status: Mapped[str] = mapped_column(String(24), default="pending", index=True)


class Follow(BaseEntity):
    __tablename__ = "follows"

    follower_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    following_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    __table_args__ = (
        UniqueConstraint("follower_id", "following_id", name="uq_follow_pair"),
        CheckConstraint("follower_id <> following_id", name="ck_follow_not_self"),
    )


class ContentImpression(BaseEntity):
    __tablename__ = "content_impressions"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    post_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("posts.id", ondelete="CASCADE"), index=True
    )
    shown_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    dwell_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    clicked: Mapped[bool] = mapped_column(Boolean, default=False)


class FeaturedSnapshot(BaseEntity):
    __tablename__ = "featured_snapshots"

    source: Mapped[str] = mapped_column(String(80), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
