from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from packages.contracts.events import EventType

from .auth import (
    create_access_token,
    create_refresh_token,
    current_user,
    hash_password,
    hash_refresh_token,
    needs_password_rehash,
    require_role,
    verify_password,
)
from .config import cors_origin_list, get_settings
from .db import engine, get_session, init_db, session_context
from .demo import (
    DEMO_DAILY_BUDGET_MESSAGE,
    DEMO_DAILY_BUDGET_REASON,
    DEMO_IP_RATE_MESSAGE,
    check_demo_budget,
    client_ip,
    enforce_demo_ip_limit,
)
from .event_store import append_event, iter_events
from .feed_ranking import deduplicate_brigading, ranking_score
from .models import (
    Agent,
    AgentTool,
    ApiKey,
    AuditLog,
    BookshelfEntry,
    Chapter,
    Comment,
    ContentImpression,
    CreditLedger,
    Entitlement,
    Follow,
    Message,
    MessageRole,
    Notification,
    Novel,
    Order,
    OrderStatus,
    PaymentTransaction,
    Post,
    PostLike,
    PushDelivery,
    PushDevice,
    ReadingProgress,
    RegisteredTool,
    Run,
    RunStatus,
    Thread,
    User,
    UserSession,
    WebhookEvent,
)
from .observability import configure_observability
from .outbox import enqueue
from .quota import QuotaExceeded, enforce_run_quota
from .redis_events import RedisRunEventStore
from .repositories import get_agent, get_thread, list_agents, list_thread_messages
from .routers.evals import router as evals_router
from .schemas import (
    AgentRead,
    AgentToolAttach,
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyRead,
    BookshelfRead,
    ChapterCreate,
    ChapterRead,
    CheckoutRequest,
    CommentCreate,
    CommentRead,
    EntitlementRead,
    LoginRequest,
    MCPSyncRead,
    MCPSyncRequest,
    MessageRead,
    NotificationCreate,
    NotificationPreferenceUpdate,
    NotificationRead,
    NovelCreate,
    NovelRead,
    OrderRead,
    PostCreate,
    PostRead,
    PushDeviceCreate,
    ReadingProgressRead,
    ReadingProgressUpdate,
    RefreshRequest,
    RegisterRequest,
    RunCreate,
    RunRead,
    ThreadCreate,
    ThreadDetail,
    ThreadRead,
    TokenResponse,
    ToolRegistrationCreate,
    ToolRegistrationRead,
    UserCreate,
    UserRead,
    WebhookPayment,
)
from .seed import (
    ensure_builtin_tools,
    ensure_default_agent,
    ensure_demo_corpus,
    ensure_handoff_agents,
)

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("agentforge.core_api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    async with session_context() as session:
        await ensure_default_agent(session)
        await ensure_handoff_agents(session)
        await ensure_builtin_tools(session)
        await ensure_demo_corpus(session)
        await session.commit()
    yield
    await engine.dispose()


app = FastAPI(title="AgentForge Core API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origin_list(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
configure_observability(app)
app.include_router(evals_router)


def _error_response(
    request: Request, code: str, message: str, status_code: int, *, retryable: bool = False
) -> JSONResponse:
    request_id = (
        getattr(request.state, "request_id", None)
        or request.headers.get("X-Request-ID")
        or uuid4().hex
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        code = str(detail.get("code", "HTTP_ERROR"))
        message = str(detail.get("message", code))
        retryable = bool(detail.get("retryable", exc.status_code >= 500))
    else:
        code = str(detail or "HTTP_ERROR").upper().replace("-", "_")
        message = str(detail or "request failed")
        retryable = exc.status_code >= 500 or exc.status_code in {408, 409, 429}
    return _error_response(request, code, message, exc.status_code, retryable=retryable)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error_response(request, "VALIDATION_ERROR", "request validation failed", 422)


def user_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        status=user.status,
        role=user.role,
        plan=user.plan,
    )


def agent_read(agent: Agent) -> AgentRead:
    return AgentRead(
        id=agent.id,
        slug=agent.slug,
        version=agent.version,
        model_ref=agent.model_ref,
        status=agent.status,
    )


def run_read(run: Run) -> RunRead:
    return RunRead(
        id=run.id,
        thread_id=run.thread_id,
        status=run.status,
        terminal_reason=run.terminal_reason,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        agent_version=run.agent_version,
        prompt_version=run.prompt_version,
        model_ref=run.model_ref,
        tool_schema_hash=run.tool_schema_hash,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", response_model=None)
async def ready() -> JSONResponse | dict[str, object]:
    settings = get_settings()
    db_ok = False
    async with session_context() as session:
        await session.execute(text("SELECT 1"))
        db_ok = True
    runtime = await _runtime_health()
    payload: dict[str, object] = {
        "status": "ok" if db_ok and runtime["reachable"] else "degraded",
        "database": db_ok,
        "runtime": runtime,
        "demo_mode": settings.demo_mode,
    }
    if payload["status"] != "ok":
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/v1/status")
async def public_status() -> dict[str, object]:
    """Public probe for the console. Never includes secrets or model names."""
    settings = get_settings()
    runtime = await _runtime_health()
    return {
        "api": "ok",
        "runtime": runtime["reachable"],
        "llm_configured": runtime["llm_configured"],
        "demo_mode": settings.demo_mode,
        "evals_source": "snapshot",
    }


async def _runtime_health() -> dict[str, bool]:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(f"{settings.agent_runtime_url.rstrip('/')}/healthz")
        body = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        return {
            "reachable": response.is_success,
            "llm_configured": bool(body.get("llm_configured")),
        }
    except (httpx.HTTPError, ValueError):
        return {"reachable": False, "llm_configured": False}


@app.post("/v1/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(require_role("admin")),
) -> UserRead:  # noqa: B008
    user = User(email=payload.email.lower(), display_name=payload.display_name)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user_read(user)


@app.post("/v1/auth/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest, session: AsyncSession = Depends(get_session)
) -> UserRead:  # noqa: B008
    existing = await session.scalar(select(User).where(User.email == payload.email.lower()))
    if existing is not None:
        raise HTTPException(status_code=409, detail="email_already_registered")
    user = User(
        email=payload.email.lower(),
        display_name=payload.display_name,
        password_hash=hash_password(payload.password),
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user_read(user)


@app.post("/v1/auth/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest, session: AsyncSession = Depends(get_session)
) -> TokenResponse:  # noqa: B008
    user = await session.scalar(select(User).where(User.email == payload.email.lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if needs_password_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        await session.commit()
    refresh_token = create_refresh_token()
    session.add(
        UserSession(
            user_id=user.id,
            refresh_token_hash=hash_refresh_token(refresh_token),
            expires_at=datetime.now(UTC) + timedelta(days=30),
        )
    )
    await session.commit()
    return TokenResponse(access_token=create_access_token(user), refresh_token=refresh_token)


@app.post("/v1/auth/refresh", response_model=TokenResponse)
async def refresh(
    payload: RefreshRequest, session: AsyncSession = Depends(get_session)
) -> TokenResponse:  # noqa: B008
    stored = await session.scalar(
        select(UserSession).where(
            UserSession.refresh_token_hash == hash_refresh_token(payload.refresh_token)
        )
    )
    now = datetime.now(UTC)
    expires_at = stored.expires_at if stored is not None else None
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if stored is None or stored.revoked_at is not None or expires_at is None or expires_at < now:
        raise HTTPException(status_code=401, detail="invalid_refresh_token")
    user = await session.get(User, stored.user_id)
    if user is None or user.status != "active":
        raise HTTPException(status_code=401, detail="user_inactive")
    stored.revoked_at = now
    refresh_token = create_refresh_token()
    session.add(
        UserSession(
            user_id=user.id,
            refresh_token_hash=hash_refresh_token(refresh_token),
            expires_at=now + timedelta(days=30),
        )
    )
    await session.commit()
    return TokenResponse(access_token=create_access_token(user), refresh_token=refresh_token)


def api_key_read(key: ApiKey) -> ApiKeyRead:
    return ApiKeyRead(
        id=key.id,
        name=key.name,
        prefix=key.prefix,
        scopes=key.scopes,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        revoked_at=key.revoked_at,
    )


@app.post("/v1/auth/api-keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: ApiKeyCreate,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(current_user),
) -> ApiKeyCreated:  # noqa: B008
    raw = f"afk_{secrets.token_urlsafe(36)}"
    key = ApiKey(
        user_id=user.id,
        name=payload.name,
        prefix=raw[:12],
        key_hash=hashlib.sha256(raw.encode()).hexdigest(),
        scopes=payload.scopes,
    )
    session.add(key)
    await session.commit()
    await session.refresh(key)
    return ApiKeyCreated(**api_key_read(key).model_dump(), key=raw)


@app.get("/v1/auth/api-keys", response_model=list[ApiKeyRead])
async def list_api_keys(
    session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> list[ApiKeyRead]:  # noqa: B008
    rows = list(
        await session.scalars(
            select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
        )
    )
    return [api_key_read(row) for row in rows]


@app.delete("/v1/auth/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> None:  # noqa: B008
    key = await session.get(ApiKey, key_id)
    if key is None or key.user_id != user.id:
        raise HTTPException(status_code=404, detail="api_key_not_found")
    key.revoked_at = datetime.now(UTC)
    await session.commit()


@app.get("/v1/auth/me", response_model=UserRead)
async def me(user: User = Depends(current_user)) -> UserRead:  # noqa: B008
    return user_read(user)


@app.get("/v1/agents", response_model=list[AgentRead])
async def agents(session: AsyncSession = Depends(get_session)) -> list[AgentRead]:  # noqa: B008
    return [agent_read(item) for item in await list_agents(session)]


@app.post("/v1/posts", response_model=PostRead, status_code=status.HTTP_201_CREATED)
async def create_post(
    payload: PostCreate,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> PostRead:
    post = Post(
        author_id=user.id,
        body=payload.body,
        model_ref=payload.model_ref,
        hot_query=payload.hot_query,
        visibility="public",
        moderation_status="approved",
        quality_score=0.0,
        like_count=0,
        view_count=0,
    )
    session.add(post)
    await session.flush()
    await enqueue(
        session,
        aggregate_type="post",
        aggregate_id=str(post.id),
        event_type="post.created",
        payload={"post_id": str(post.id), "author_id": str(user.id)},
    )
    await session.commit()
    await session.refresh(post)
    return PostRead.model_validate(post, from_attributes=True)


@app.post("/v1/posts/{post_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def like_post(
    post_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> None:  # noqa: B008
    post = await session.get(Post, post_id)
    if post is None or post.visibility != "public" or post.moderation_status != "approved":
        raise HTTPException(status_code=404, detail="post_not_found")
    like = await session.scalar(
        select(PostLike).where(PostLike.user_id == user.id, PostLike.post_id == post_id)
    )
    if like is None:
        session.add(PostLike(user_id=user.id, post_id=post_id))
        post.like_count += 1
        await enqueue(
            session,
            aggregate_type="post",
            aggregate_id=str(post_id),
            event_type="post.liked",
            payload={"post_id": str(post_id), "user_id": str(user.id)},
        )
        await session.commit()


@app.post(
    "/v1/posts/{post_id}/comments", response_model=CommentRead, status_code=status.HTTP_201_CREATED
)
async def create_comment(
    post_id: UUID,
    payload: CommentCreate,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(current_user),
) -> CommentRead:  # noqa: B008
    post = await session.get(Post, post_id)
    if post is None or post.visibility != "public" or post.moderation_status != "approved":
        raise HTTPException(status_code=404, detail="post_not_found")
    comment = Comment(
        post_id=post_id, author_id=user.id, body=payload.body, moderation_status="approved"
    )
    session.add(comment)
    await session.flush()
    await enqueue(
        session,
        aggregate_type="post",
        aggregate_id=str(post_id),
        event_type="comment.created",
        payload={"comment_id": str(comment.id), "post_id": str(post_id), "author_id": str(user.id)},
    )
    await session.commit()
    await session.refresh(comment)
    return CommentRead.model_validate(comment, from_attributes=True)


@app.post("/v1/users/{user_id}/follow", status_code=status.HTTP_204_NO_CONTENT)
async def follow_user(
    user_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> None:  # noqa: B008
    if user_id == user.id or await session.get(User, user_id) is None:
        raise HTTPException(status_code=400, detail="invalid_follow_target")
    existing = await session.scalar(
        select(Follow).where(Follow.follower_id == user.id, Follow.following_id == user_id)
    )
    if existing is None:
        session.add(Follow(follower_id=user.id, following_id=user_id))
        await session.commit()


@app.delete("/v1/users/{user_id}/follow", status_code=status.HTTP_204_NO_CONTENT)
async def unfollow_user(
    user_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> None:  # noqa: B008
    existing = await session.scalar(
        select(Follow).where(Follow.follower_id == user.id, Follow.following_id == user_id)
    )
    if existing is not None:
        await session.delete(existing)
        await session.commit()


@app.get("/v1/feed", response_model=list[PostRead])
async def feed(
    response: Response,
    query: str | None = None,
    cursor: str | None = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(current_user),
) -> list[PostRead]:  # noqa: B008
    if not 1 <= limit <= 50:
        raise HTTPException(status_code=422, detail="invalid_feed_limit")
    base = select(Post).where(Post.visibility == "public", Post.moderation_status == "approved")
    if cursor:
        try:
            decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
            timestamp, post_id = decoded.split("|", 1)
            cursor_time = datetime.fromisoformat(timestamp)
            if cursor_time.tzinfo is None:
                cursor_time = cursor_time.replace(tzinfo=UTC)
            base = base.where(
                (Post.created_at < cursor_time)
                | ((Post.created_at == cursor_time) & (Post.id < UUID(post_id)))
            )
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid_feed_cursor") from exc
    latest = list(
        await session.scalars(base.order_by(Post.created_at.desc(), Post.id.desc()).limit(limit))
    )
    pivot = uuid4()
    exploratory = list(
        await session.scalars(base.where(Post.id >= pivot).order_by(Post.id).limit(25))
    )
    if len(exploratory) < 25:
        exploratory.extend(
            list(
                await session.scalars(
                    base.where(Post.id < pivot).order_by(Post.id).limit(25 - len(exploratory))
                )
            )
        )
    by_id = {post.id: post for post in latest + exploratory}
    candidates = deduplicate_brigading(list(by_id.values()))
    ranked = sorted(candidates, key=lambda post: ranking_score(post, query=query), reverse=True)
    selected = ranked[:limit]
    for post in selected:
        post.view_count += 1
        session.add(ContentImpression(user_id=user.id, post_id=post.id, shown_at=datetime.now(UTC)))
    if selected:
        await session.commit()
        last = selected[-1]
        raw_cursor = f"{last.created_at.isoformat()}|{last.id}"
        response.headers["X-Next-Cursor"] = base64.urlsafe_b64encode(raw_cursor.encode()).decode()
    return [PostRead.model_validate(post, from_attributes=True) for post in selected]


@app.post("/v1/novels", response_model=NovelRead, status_code=status.HTTP_201_CREATED)
async def create_novel(
    payload: NovelCreate,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    _: User = Depends(current_user),  # noqa: B008
) -> NovelRead:
    novel = Novel(
        title=payload.title, author_name=payload.author_name, description=payload.description
    )
    session.add(novel)
    await session.commit()
    await session.refresh(novel)
    return NovelRead.model_validate(novel, from_attributes=True)


@app.get("/v1/novels/{novel_id}", response_model=NovelRead)
async def get_novel(novel_id: UUID, session: AsyncSession = Depends(get_session)) -> NovelRead:  # noqa: B008
    novel = await session.get(Novel, novel_id)
    if novel is None or novel.status != "published":
        raise HTTPException(status_code=404, detail="novel_not_found")
    return NovelRead.model_validate(novel, from_attributes=True)


@app.post(
    "/v1/novels/{novel_id}/chapters",
    response_model=ChapterRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_chapter(
    novel_id: UUID,
    payload: ChapterCreate,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    _: User = Depends(current_user),  # noqa: B008
) -> ChapterRead:
    if await session.get(Novel, novel_id) is None:
        raise HTTPException(status_code=404, detail="novel_not_found")
    chapter = Chapter(
        novel_id=novel_id,
        number=payload.number,
        title=payload.title,
        content=payload.content,
        is_published=payload.is_published,
        publish_at=payload.publish_at,
    )
    session.add(chapter)
    try:
        await session.flush()
        await enqueue(
            session,
            aggregate_type="chapter",
            aggregate_id=str(chapter.id),
            event_type="chapter.published" if chapter.is_published else "chapter.created",
            payload={"novel_id": str(novel_id), "chapter_id": str(chapter.id)},
        )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        if "UNIQUE" in str(exc).upper() or "unique" in str(exc):
            raise HTTPException(status_code=409, detail="chapter_number_exists") from exc
        raise
    await session.refresh(chapter)
    return ChapterRead.model_validate(chapter, from_attributes=True)


@app.get("/v1/novels/{novel_id}/chapters", response_model=list[ChapterRead])
async def list_chapters(
    novel_id: UUID, session: AsyncSession = Depends(get_session)
) -> list[ChapterRead]:  # noqa: B008
    novel = await session.get(Novel, novel_id)
    if novel is None or novel.status != "published":
        raise HTTPException(status_code=404, detail="novel_not_found")
    rows = list(
        await session.scalars(
            select(Chapter)
            .where(
                Chapter.novel_id == novel_id,
                Chapter.is_published.is_(True),
                Chapter.visibility == "public",
            )
            .order_by(Chapter.number)
        )
    )
    now = datetime.now(UTC)
    visible = []
    for row in rows:
        publish_at = row.publish_at
        if publish_at is not None:
            if publish_at.tzinfo is None:
                publish_at = publish_at.replace(tzinfo=UTC)
            if publish_at > now:
                continue
        visible.append(ChapterRead.model_validate(row, from_attributes=True))
    return visible


@app.post("/v1/bookshelf/{novel_id}", status_code=status.HTTP_204_NO_CONTENT)
async def add_to_bookshelf(
    novel_id: UUID,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> None:
    novel = await session.get(Novel, novel_id)
    if novel is None or novel.status != "published":
        raise HTTPException(status_code=404, detail="novel_not_found")
    existing = await session.scalar(
        select(BookshelfEntry).where(
            BookshelfEntry.user_id == user.id, BookshelfEntry.novel_id == novel_id
        )
    )
    if existing is None:
        session.add(BookshelfEntry(user_id=user.id, novel_id=novel_id))
        await session.commit()


@app.get("/v1/bookshelf", response_model=list[BookshelfRead])
async def list_bookshelf(
    session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> list[BookshelfRead]:  # noqa: B008
    rows = list(
        await session.scalars(
            select(BookshelfEntry)
            .join(Novel, Novel.id == BookshelfEntry.novel_id)
            .where(BookshelfEntry.user_id == user.id, Novel.status == "published")
            .order_by(BookshelfEntry.created_at.desc())
        )
    )
    return [BookshelfRead(novel_id=row.novel_id, created_at=row.created_at) for row in rows]


@app.put("/v1/novels/{novel_id}/reading-progress", response_model=ReadingProgressRead)
async def update_reading_progress(
    novel_id: UUID,
    payload: ReadingProgressUpdate,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> ReadingProgressRead:
    novel = await session.get(Novel, novel_id)
    if novel is None or novel.status != "published":
        raise HTTPException(status_code=404, detail="novel_not_found")
    chapter = await session.get(Chapter, payload.chapter_id)
    if (
        chapter is None
        or chapter.novel_id != novel_id
        or not chapter.is_published
        or chapter.visibility != "public"
    ):
        raise HTTPException(status_code=400, detail="chapter_not_readable")
    publish_at = chapter.publish_at
    if publish_at is not None:
        if publish_at.tzinfo is None:
            publish_at = publish_at.replace(tzinfo=UTC)
        if publish_at > datetime.now(UTC):
            raise HTTPException(status_code=400, detail="chapter_not_readable")
    if payload.chapter_number != chapter.number:
        raise HTTPException(status_code=400, detail="chapter_number_mismatch")
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))
    progress_query = select(ReadingProgress).where(
        ReadingProgress.user_id == user.id, ReadingProgress.novel_id == novel_id
    )
    if dialect == "postgresql":
        progress_query = progress_query.with_for_update()
    current = await session.scalar(progress_query)
    incoming_time = payload.client_updated_at
    if incoming_time.tzinfo is None:
        incoming_time = incoming_time.replace(tzinfo=UTC)
    current_time = current.client_updated_at if current is not None else None
    if current_time is not None and current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    stale_reason = None
    if current is not None:
        if current_time is not None and current_time > incoming_time:
            stale_reason = "client_timestamp_older"
        elif payload.chapter_number < current.chapter_number:
            stale_reason = "chapter_regression"
        elif payload.progress_percent < current.progress_percent:
            stale_reason = "progress_regression"
        elif payload.paragraph_index < current.paragraph_index:
            stale_reason = "paragraph_regression"
    if stale_reason:
        assert current is not None
        return ReadingProgressRead(
            accepted=False,
            stale_reason=stale_reason,
            chapter_id=current.chapter_id,
            chapter_number=current.chapter_number,
            progress_percent=current.progress_percent,
            paragraph_index=current.paragraph_index,
        )
    if current is None:
        current = ReadingProgress(
            user_id=user.id,
            novel_id=novel_id,
            chapter_id=chapter.id,
            chapter_number=payload.chapter_number,
            progress_percent=payload.progress_percent,
            paragraph_index=payload.paragraph_index,
            client_updated_at=incoming_time,
        )
        session.add(current)
    else:
        current.chapter_id = chapter.id
        current.chapter_number = payload.chapter_number
        current.progress_percent = payload.progress_percent
        current.paragraph_index = payload.paragraph_index
        current.client_updated_at = incoming_time
        current.synced_at = datetime.now(UTC)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        if dialect == "sqlite":
            await session.execute(text("BEGIN IMMEDIATE"))
        current = await session.scalar(progress_query)
        if current is None:
            raise
        retry_stale = None
        current_time = current.client_updated_at
        if current_time is not None and current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        if current_time is not None and current_time > incoming_time:
            retry_stale = "client_timestamp_older"
        elif payload.chapter_number < current.chapter_number:
            retry_stale = "chapter_regression"
        elif payload.progress_percent < current.progress_percent:
            retry_stale = "progress_regression"
        elif payload.paragraph_index < current.paragraph_index:
            retry_stale = "paragraph_regression"
        if retry_stale:
            return ReadingProgressRead(
                accepted=False,
                stale_reason=retry_stale,
                chapter_id=current.chapter_id,
                chapter_number=current.chapter_number,
                progress_percent=current.progress_percent,
                paragraph_index=current.paragraph_index,
            )
        current.chapter_id = chapter.id
        current.chapter_number = payload.chapter_number
        current.progress_percent = payload.progress_percent
        current.paragraph_index = payload.paragraph_index
        current.client_updated_at = incoming_time
        current.synced_at = datetime.now(UTC)
        await session.commit()
    return ReadingProgressRead(
        accepted=True,
        chapter_id=chapter.id,
        chapter_number=payload.chapter_number,
        progress_percent=payload.progress_percent,
        paragraph_index=payload.paragraph_index,
    )


def order_read(order: Order) -> OrderRead:
    return OrderRead(
        id=order.id,
        type=order.type,
        product_ref=order.product_ref,
        amount=order.amount,
        currency=order.currency,
        status=order.status.value,
        expires_at=order.expires_at,
    )


@app.post("/v1/checkout", response_model=OrderRead, status_code=status.HTTP_201_CREATED)
async def checkout(
    payload: CheckoutRequest,
    idempotency_key: str = Header(min_length=8, max_length=200, alias="Idempotency-Key"),
    session: AsyncSession = Depends(get_session),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> OrderRead:
    product_catalog = {
        "credits-100": ("credits", 500, 100),
        "credits-500": ("credits", 2000, 500),
        "subscription-basic": ("subscription", 999, 30),
    }
    product = product_catalog.get(payload.product_ref)
    if product is None or product[0] != payload.type or product[1] != payload.amount:
        raise HTTPException(status_code=400, detail="invalid_product_or_amount")
    existing = await session.scalar(select(Order).where(Order.idempotency_key == idempotency_key))
    if existing is not None:
        if (
            existing.user_id != user.id
            or existing.product_ref != payload.product_ref
            or existing.amount != payload.amount
        ):
            raise HTTPException(status_code=409, detail="idempotency_key_conflict")
        return order_read(existing)
    order = Order(
        user_id=user.id,
        type=payload.type,
        product_ref=payload.product_ref,
        amount=payload.amount,
        currency=payload.currency.lower(),
        status=OrderStatus.AWAITING_PAYMENT,
        idempotency_key=idempotency_key,
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    session.add(order)
    try:
        await session.flush()
        session.add(
            PaymentTransaction(
                order_id=order.id,
                provider="test",
                provider_tx_id=f"test_session:{order.id}",
                status="pending",
                raw_ref={"mode": "local_simulated", "created_at": datetime.now(UTC).isoformat()},
            )
        )
        await enqueue(
            session,
            aggregate_type="order",
            aggregate_id=str(order.id),
            event_type="order.created",
            payload={
                "order_id": str(order.id),
                "user_id": str(user.id),
                "product_ref": order.product_ref,
            },
        )
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await session.scalar(
            select(Order).where(Order.idempotency_key == idempotency_key)
        )
        if existing is None:
            raise
        return order_read(existing)
    await session.refresh(order)
    return order_read(order)


@app.get("/v1/orders/{order_id}", response_model=OrderRead)
async def get_order(
    order_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> OrderRead:  # noqa: B008
    order = await session.get(Order, order_id)
    if order is None or order.user_id != user.id:
        raise HTTPException(status_code=404, detail="order_not_found")
    return order_read(order)


@app.get("/v1/entitlements", response_model=list[EntitlementRead])
async def list_entitlements(
    session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> list[EntitlementRead]:  # noqa: B008
    rows = list(
        await session.scalars(
            select(Entitlement)
            .where(Entitlement.user_id == user.id)
            .order_by(Entitlement.period_end.desc())
        )
    )
    return [
        EntitlementRead(
            id=row.id,
            kind=row.kind,
            plan=row.plan,
            status=row.status,
            period_start=row.period_start,
            period_end=row.period_end,
        )
        for row in rows
    ]


@app.post("/v1/novels/{novel_id}/reading-progress/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_reading_progress(
    novel_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> None:  # noqa: B008
    novel = await session.get(Novel, novel_id)
    if novel is None or novel.status != "published":
        raise HTTPException(status_code=404, detail="novel_not_found")
    current = await session.scalar(
        select(ReadingProgress).where(
            ReadingProgress.user_id == user.id, ReadingProgress.novel_id == novel_id
        )
    )
    if current is not None:
        await session.delete(current)
        await session.commit()


@app.post("/v1/webhooks/{provider}", status_code=status.HTTP_200_OK)
async def payment_webhook(
    provider: str,
    payload: WebhookPayment,
    request: Request,
    x_webhook_signature: str | None = Header(default=None, alias="X-Webhook-Signature"),
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> dict[str, str]:
    if provider not in {"test", "stripe"}:
        raise HTTPException(status_code=404, detail="provider_not_supported")
    if provider != "test":
        body = await request.body()
        expected = hmac.new(
            get_settings().webhook_signing_secret.encode(), body, hashlib.sha256
        ).hexdigest()
        if not x_webhook_signature or not hmac.compare_digest(x_webhook_signature, expected):
            raise HTTPException(status_code=401, detail="invalid_webhook_signature")
    duplicate = await session.scalar(
        select(WebhookEvent).where(
            WebhookEvent.provider == provider, WebhookEvent.event_id == payload.event_id
        )
    )
    if duplicate is not None:
        return {"status": "duplicate_ignored"}
    event = WebhookEvent(provider=provider, event_id=payload.event_id, status="processing")
    session.add(event)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        return {"status": "duplicate_ignored"}
    order = await session.get(Order, payload.order_id)
    if order is None:
        event.status = "rejected"
        await session.commit()
        raise HTTPException(status_code=404, detail="order_not_found")
    expires_at = (
        order.expires_at if order.expires_at.tzinfo else order.expires_at.replace(tzinfo=UTC)
    )
    if expires_at < datetime.now(UTC) and order.status not in {
        OrderStatus.PAID,
        OrderStatus.FULFILLED,
    }:
        event.status = "rejected"
        await session.commit()
        raise HTTPException(status_code=409, detail="order_expired")
    transaction = await session.scalar(
        select(PaymentTransaction)
        .where(
            PaymentTransaction.order_id == order.id,
            PaymentTransaction.provider == provider,
            PaymentTransaction.status == "pending",
        )
        .order_by(PaymentTransaction.created_at.desc())
    )
    if transaction is None:
        transaction = PaymentTransaction(
            order_id=order.id,
            provider=provider,
            provider_tx_id=payload.provider_tx_id,
            status="pending",
        )
        session.add(transaction)
    transaction.provider_tx_id = payload.provider_tx_id
    transaction.status = "completed" if payload.success else "failed"
    if payload.success and order.status not in {OrderStatus.PAID, OrderStatus.FULFILLED}:
        order.status = OrderStatus.PAID
        order.status = OrderStatus.FULFILLING
        if order.type == "subscription":
            now = datetime.now(UTC)
            session.add(
                Entitlement(
                    user_id=order.user_id,
                    kind="subscription",
                    plan=order.product_ref,
                    status="active",
                    period_start=now,
                    period_end=now + timedelta(days=30),
                    order_id=order.id,
                )
            )
        else:
            ledger_key = f"order:{order.id}"
            if (
                await session.scalar(
                    select(CreditLedger).where(CreditLedger.idempotency_key == ledger_key)
                )
                is None
            ):
                previous = await session.scalar(
                    select(func.coalesce(func.sum(CreditLedger.delta), 0)).where(
                        CreditLedger.user_id == order.user_id
                    )
                )
                grant = {"credits-100": 100, "credits-500": 500}.get(order.product_ref)
                if grant is None:
                    raise HTTPException(status_code=400, detail="invalid_credit_product")
                session.add(
                    CreditLedger(
                        user_id=order.user_id,
                        delta=grant,
                        balance_after=int(previous or 0) + grant,
                        reason=f"topup:{order.product_ref}",
                        order_id=order.id,
                        idempotency_key=ledger_key,
                    )
                )
        order.status = OrderStatus.FULFILLED
    elif not payload.success and order.status not in {OrderStatus.FULFILLED, OrderStatus.PAID}:
        order.status = OrderStatus.FAILED
    event.status = "processed"
    await enqueue(
        session,
        aggregate_type="order",
        aggregate_id=str(order.id),
        event_type="order.paid" if payload.success else "order.failed",
        payload={"order_id": str(order.id), "provider": provider, "event_id": payload.event_id},
    )
    await session.commit()
    return {"status": "processed"}


@app.post("/v1/devices/push", status_code=status.HTTP_204_NO_CONTENT)
async def register_push_device(
    payload: PushDeviceCreate,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> None:
    token_hash = hashlib.sha256(payload.token.encode()).hexdigest()
    device = await session.scalar(select(PushDevice).where(PushDevice.token_hash == token_hash))
    if device is None:
        session.add(
            PushDevice(
                user_id=user.id,
                platform=payload.platform,
                token_hash=token_hash,
                permission=payload.permission,
                is_active=True,
            )
        )
    else:
        device.user_id = user.id
        device.platform = payload.platform
        device.permission = payload.permission
        device.is_active = True
    await session.commit()


@app.post("/v1/notifications", response_model=NotificationRead, status_code=status.HTTP_201_CREATED)
async def create_notification(
    payload: NotificationCreate,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> NotificationRead:
    notification = Notification(user_id=user.id, type=payload.type, payload_json=payload.payload)
    session.add(notification)
    await session.flush()
    preferences = user.notification_preferences or {}
    enabled = bool(preferences.get("enabled", True))
    muted_types = set(preferences.get("muted_types", []))
    cooldown_seconds = int(preferences.get("cooldown_seconds", 1800))
    daily_cap = int(preferences.get("daily_cap", 3))
    now = datetime.now(UTC)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    daily_count = await session.scalar(
        select(func.count(Notification.id)).where(
            Notification.user_id == user.id, Notification.created_at >= day_start
        )
    )
    recent_count = await session.scalar(
        select(func.count(Notification.id)).where(
            Notification.user_id == user.id,
            Notification.type == notification.type,
            Notification.created_at >= now - timedelta(seconds=cooldown_seconds),
        )
    )
    can_push = (
        enabled
        and notification.type not in muted_types
        and int(daily_count or 0) <= daily_cap
        and int(recent_count or 0) <= 1
    )
    if can_push:
        devices = list(
            await session.scalars(
                select(PushDevice).where(
                    PushDevice.user_id == user.id,
                    PushDevice.is_active.is_(True),
                    PushDevice.permission == "granted",
                )
            )
        )
        for device in devices:
            session.add(
                PushDelivery(
                    notification_id=notification.id,
                    device_id=device.id,
                    channel="simulated",
                    status="queued",
                    attempts=0,
                    next_attempt_at=now,
                )
            )
    await enqueue(
        session,
        aggregate_type="notification",
        aggregate_id=str(notification.id),
        event_type="notification.created",
        payload={
            "notification_id": str(notification.id),
            "user_id": str(user.id),
            "type": notification.type,
        },
    )
    await session.commit()
    await session.refresh(notification)
    return NotificationRead(
        id=notification.id,
        type=notification.type,
        payload=notification.payload_json,
        read_at=notification.read_at,
        created_at=notification.created_at,
    )


@app.get("/v1/notifications/preferences")
async def get_notification_preferences(user: User = Depends(current_user)) -> dict:  # noqa: B008
    return user.notification_preferences or {
        "enabled": True,
        "muted_types": [],
        "cooldown_seconds": 1800,
        "daily_cap": 3,
    }


@app.put("/v1/notifications/preferences")
async def update_notification_preferences(
    payload: NotificationPreferenceUpdate,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(current_user),
) -> dict:  # noqa: B008
    user.notification_preferences = payload.model_dump()
    await session.commit()
    return user.notification_preferences


@app.get("/v1/notifications", response_model=list[NotificationRead])
async def list_notifications(
    session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> list[NotificationRead]:  # noqa: B008
    rows = list(
        await session.scalars(
            select(Notification)
            .where(Notification.user_id == user.id)
            .order_by(Notification.created_at.desc())
            .limit(100)
        )
    )
    return [
        NotificationRead(
            id=row.id,
            type=row.type,
            payload=row.payload_json,
            read_at=row.read_at,
            created_at=row.created_at,
        )
        for row in rows
    ]


@app.post("/v1/notifications/{notification_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_notification_read(
    notification_id: UUID,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(current_user),
) -> None:  # noqa: B008
    notification = await session.get(Notification, notification_id)
    if notification is None or notification.user_id != user.id:
        raise HTTPException(status_code=404, detail="notification_not_found")
    notification.read_at = datetime.now(UTC)
    await session.commit()


def tool_read(tool: RegisteredTool) -> ToolRegistrationRead:
    return ToolRegistrationRead(
        id=tool.id,
        name=tool.name,
        version=tool.version,
        source=tool.source,
        description=tool.description,
        risk_level=tool.risk_level,
        status=tool.status,
        schema_hash=tool.schema_hash,
        origin_uri=tool.origin_uri,
        origin_name=tool.origin_name,
    )


@app.get("/v1/tools/catalog", response_model=list[ToolRegistrationRead])
async def tool_catalog(session: AsyncSession = Depends(get_session)) -> list[ToolRegistrationRead]:  # noqa: B008
    rows = list(
        await session.scalars(
            select(RegisteredTool)
            .where(RegisteredTool.status == "approved")
            .order_by(RegisteredTool.name)
        )
    )
    return [tool_read(row) for row in rows]


@app.post(
    "/v1/admin/tools", response_model=ToolRegistrationRead, status_code=status.HTTP_201_CREATED
)
async def register_tool(
    payload: ToolRegistrationCreate,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    admin: User = Depends(require_role("admin")),  # noqa: B008
) -> ToolRegistrationRead:
    schema_hash = hashlib.sha256(
        json.dumps(payload.input_schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    existing = await session.scalar(
        select(RegisteredTool).where(RegisteredTool.name == payload.name)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="tool_name_exists")
    tool = RegisteredTool(
        name=payload.name,
        version=payload.version,
        source=payload.source,
        description=payload.description,
        input_schema=payload.input_schema,
        risk_level=payload.risk_level,
        status="quarantined",
        schema_hash=schema_hash,
    )
    session.add(tool)
    await session.commit()
    await session.refresh(tool)
    return tool_read(tool)


@app.post("/v1/admin/tools/{tool_id}/approve", response_model=ToolRegistrationRead)
async def approve_tool(
    tool_id: UUID,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(require_role("admin")),
) -> ToolRegistrationRead:  # noqa: B008
    tool = await session.get(RegisteredTool, tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="tool_not_found")
    tool.status = "approved"
    tool.approved_by = admin.id
    await session.commit()
    await session.refresh(tool)
    return tool_read(tool)


@app.post("/v1/admin/mcp/sync", response_model=MCPSyncRead)
async def sync_mcp_tools(
    payload: MCPSyncRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    admin: User = Depends(require_role("admin")),  # noqa: B008
) -> MCPSyncRead:
    from services.agent_runtime.app.mcp import (
        MCPClient,
        MCPEndpointError,
        MCPError,
        registry_name,
        validate_mcp_endpoint,
    )

    try:
        endpoint = validate_mcp_endpoint(payload.endpoint)
    except MCPEndpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    client = MCPClient(endpoint, bearer_token=payload.bearer_token)
    try:
        discovered = await client.list_tools()
    except (MCPError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail="mcp_discovery_failed") from exc
    created: list[ToolRegistrationRead] = []
    skipped: list[str] = []
    for remote in discovered:
        name = registry_name(remote.name, payload.name_prefix)
        existing = await session.scalar(select(RegisteredTool).where(RegisteredTool.name == name))
        if existing is not None:
            skipped.append(name)
            continue
        schema = remote.inputSchema or {"type": "object"}
        schema_hash = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        tool = RegisteredTool(
            name=name,
            version="1.0.0",
            source="mcp",
            description=remote.description or remote.name,
            input_schema=schema,
            risk_level=payload.risk_level,
            status="quarantined",
            schema_hash=schema_hash,
            origin_uri=endpoint,
            origin_name=remote.name,
        )
        session.add(tool)
        await session.flush()
        created.append(tool_read(tool))
    session.add(
        AuditLog(
            actor_id=admin.id,
            action="mcp.sync",
            resource="mcp",
            resource_id=endpoint,
            metadata_json={
                "discovered": len(discovered),
                "created": [item.name for item in created],
                "skipped": skipped,
            },
        )
    )
    await session.commit()
    return MCPSyncRead(
        endpoint=endpoint, discovered=len(discovered), created=created, skipped=skipped
    )


@app.post("/v1/admin/agents/{agent_id}/tools", status_code=status.HTTP_201_CREATED)
async def attach_agent_tool(
    agent_id: UUID,
    payload: AgentToolAttach,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    admin: User = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, str]:
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent_not_found")
    tool = await session.scalar(
        select(RegisteredTool).where(RegisteredTool.name == payload.tool_ref)
    )
    if tool is None or tool.status != "approved":
        raise HTTPException(status_code=400, detail="tool_not_approved")
    existing = await session.scalar(
        select(AgentTool).where(
            AgentTool.agent_id == agent_id, AgentTool.tool_ref == payload.tool_ref
        )
    )
    if existing is None:
        session.add(AgentTool(agent_id=agent_id, tool_ref=payload.tool_ref, enabled=True))
        await session.commit()
    return {"agent_id": str(agent_id), "tool_ref": payload.tool_ref}


@app.post("/v1/threads", response_model=ThreadRead, status_code=status.HTTP_201_CREATED)
async def create_thread(
    payload: ThreadCreate,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> ThreadRead:
    if payload.user_id != user.id:
        raise HTTPException(status_code=403, detail="user_mismatch")
    if await get_agent(session, payload.agent_id) is None:
        raise HTTPException(status_code=404, detail="agent_not_found")
    thread = Thread(user_id=payload.user_id, agent_id=payload.agent_id, title=payload.title)
    session.add(thread)
    await session.commit()
    await session.refresh(thread)
    return ThreadRead(
        id=thread.id,
        user_id=thread.user_id,
        agent_id=thread.agent_id,
        title=thread.title,
        created_at=thread.created_at,
    )


@app.get("/v1/threads/{thread_id}", response_model=ThreadDetail)
async def thread_detail(
    thread_id: UUID,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(current_user),
) -> ThreadDetail:  # noqa: B008
    thread = await get_thread(session, thread_id)
    if thread is None or thread.user_id != user.id:
        raise HTTPException(status_code=404, detail="thread_not_found")
    messages = await list_thread_messages(session, thread_id)
    return ThreadDetail(
        id=thread.id,
        user_id=thread.user_id,
        agent_id=thread.agent_id,
        title=thread.title,
        created_at=thread.created_at,
        messages=[
            MessageRead(
                id=m.id,
                role=m.role,
                content=m.content,
                reasoning_content=m.reasoning_content,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@app.post(
    "/v1/threads/{thread_id}/runs", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED
)
async def create_run(
    thread_id: UUID,
    payload: RunCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    user: User = Depends(current_user),  # noqa: B008
) -> RunRead:
    thread = await get_thread(session, thread_id)
    if thread is None or thread.user_id != payload.user_id or payload.user_id != user.id:
        raise HTTPException(status_code=404, detail="thread_not_found")
    settings = get_settings()
    if settings.demo_mode and not await enforce_demo_ip_limit(client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "demo_ip_rate_limited",
                "message": DEMO_IP_RATE_MESSAGE,
                "retryable": True,
            },
            headers={"Retry-After": "3600"},
        )
    try:
        await enforce_run_quota(session, user, thread_id, payload.content)
    except QuotaExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail={"code": exc.code, "limit": exc.limit, "retryable": True},
            headers={"Retry-After": "1"},
        ) from exc
    if settings.demo_mode and not await check_demo_budget(session):
        user_message = Message(thread_id=thread_id, role=MessageRole.USER, content=payload.content)
        run = Run(
            thread_id=thread_id,
            status=RunStatus.BUDGET_EXCEEDED,
            terminal_reason=DEMO_DAILY_BUDGET_REASON,
            completed_at=datetime.now(UTC),
        )
        session.add_all([user_message, run])
        await session.flush()
        await append_event(
            session,
            run_id=run.id,
            thread_id=thread.id,
            event_type=EventType.RUN_BUDGET_EXCEEDED,
            payload={"reason": DEMO_DAILY_BUDGET_REASON, "message": DEMO_DAILY_BUDGET_MESSAGE},
        )
        await session.commit()
        return run_read(run)
    user_message = Message(thread_id=thread_id, role=MessageRole.USER, content=payload.content)
    run = Run(thread_id=thread_id, status=RunStatus.CREATED)
    session.add_all([user_message, run])
    await session.flush()
    await enqueue(
        session,
        aggregate_type="run",
        aggregate_id=str(run.id),
        event_type="run.created",
        payload={"run_id": str(run.id), "thread_id": str(thread.id)},
    )
    await session.commit()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{settings.agent_runtime_url}/internal/runs/{run.id}",
                headers={"X-Runtime-Token": settings.runtime_internal_token},
            )
            response.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        logger.exception("runtime dispatch failed", extra={"run_id": str(run.id)})
        # Keep the durable run in CREATED so the Worker can retry dispatch.
        raise HTTPException(status_code=503, detail="agent_runtime_unavailable") from exc
    return run_read(run)


@app.post("/v1/runs/{run_id}/cancel", response_model=RunRead)
async def cancel_run(
    run_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> RunRead:  # noqa: B008
    run = await session.get(Run, run_id)
    thread = await session.get(Thread, run.thread_id) if run is not None else None
    if run is None or thread is None or thread.user_id != user.id:
        raise HTTPException(status_code=404, detail="run_not_found")
    if run.status in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELED,
        RunStatus.BUDGET_EXCEEDED,
        RunStatus.APPROVAL_REQUIRED,
    }:
        return run_read(run)
    run.status = RunStatus.CANCELED
    run.terminal_reason = "user_canceled"
    await append_event(
        session,
        run_id=run.id,
        thread_id=thread.id,
        event_type=EventType.RUN_CANCELED,
        payload={"reason": "user_canceled"},
    )
    await session.commit()
    return run_read(run)


@app.get("/v1/runs/{run_id}", response_model=RunRead)
async def get_run(
    run_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> RunRead:  # noqa: B008
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    thread = await session.get(Thread, run.thread_id)
    if thread is None or thread.user_id != user.id:
        raise HTTPException(status_code=404, detail="run_not_found")
    return run_read(run)


@app.get("/v1/runs/{run_id}/snapshot")
async def run_snapshot(
    run_id: UUID, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)
) -> dict:  # noqa: B008
    run = await session.get(Run, run_id)
    thread = await session.get(Thread, run.thread_id) if run is not None else None
    if run is None or thread is None or thread.user_id != user.id:
        raise HTTPException(status_code=404, detail="run_not_found")
    events = [event async for event in iter_events(session, run_id, 0)]
    return {
        "run": run_read(run).model_dump(mode="json"),
        "events": [event.model_dump(mode="json") for event in events],
    }


@app.get("/v1/runs/{run_id}/events")
async def run_events(
    run_id: UUID,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    user: User = Depends(current_user),  # noqa: B008
):
    try:
        after = int(last_event_id or 0)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid_last_event_id") from exc
    if after < 0:
        raise HTTPException(status_code=400, detail="invalid_last_event_id")

    async with session_context() as session:
        run = await session.get(Run, run_id)
        thread = await session.get(Thread, run.thread_id) if run is not None else None
        if run is None or thread is None or thread.user_id != user.id:
            raise HTTPException(status_code=404, detail="run_not_found")
    redis_url = get_settings().redis_url
    if after > 0 and redis_url:
        from redis.asyncio import Redis

        redis_client = Redis.from_url(redis_url, decode_responses=True)
        try:
            oldest = await RedisRunEventStore(redis_client).oldest_sequence(run_id)
            if oldest is not None and after < oldest - 1:
                raise HTTPException(
                    status_code=410,
                    detail={
                        "code": "RESUME_WINDOW_EXPIRED",
                        "message": "the requested event cursor is outside the Redis replay window",
                        "snapshot_url": f"/v1/runs/{run_id}/snapshot",
                    },
                )
        finally:
            await redis_client.aclose()

    async def stream():
        cursor = after
        if get_settings().redis_url:
            from redis.asyncio import Redis

            redis_client = Redis.from_url(get_settings().redis_url, decode_responses=True)
            try:
                store = RedisRunEventStore(redis_client)
                oldest = await store.oldest_sequence(run_id)
                # A new client starts at zero. SQL fills the history that Redis
                # has trimmed, then Redis supplies the hot portion of the stream.
                if oldest is not None and cursor < oldest - 1:
                    async with session_context() as session:
                        async for event in iter_events(session, run_id, cursor):
                            if event.sequence >= oldest:
                                break
                            cursor = event.sequence
                            yield f"id: {cursor}\nevent: {event.type.value}\ndata: {event.model_dump_json()}\n\n"
                async for event in store.replay(run_id, cursor):
                    cursor = event.sequence
                    yield f"id: {cursor}\nevent: {event.type.value}\ndata: {event.model_dump_json()}\n\n"
            except Exception:
                logger.warning("redis replay unavailable; using SQL event store", exc_info=True)
            finally:
                await redis_client.aclose()
        while True:
            if await request.is_disconnected():
                return
            async with session_context() as session:
                async for event in iter_events(session, run_id, cursor):
                    cursor = event.sequence
                    yield f"id: {cursor}\nevent: {event.type.value}\ndata: {event.model_dump_json()}\n\n"
                current = await session.get(Run, run_id)
            if (
                current
                and current.status
                in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.CANCELED,
                    RunStatus.BUDGET_EXCEEDED,
                    RunStatus.APPROVAL_REQUIRED,
                }
                and cursor > 0
            ):
                return
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
