from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import get_session
from .models import ApiKey, User

bearer = HTTPBearer(auto_error=False)

_ARGON2 = PasswordHasher(time_cost=3, memory_cost=65_536, parallelism=4, hash_len=32, salt_len=16)
_PBKDF2_PREFIX = "pbkdf2_sha256$"


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("password_must_have_at_least_12_characters")
    return _ARGON2.hash(password)


def verify_password(password: str, encoded: str | None) -> bool:
    if not encoded:
        return False
    if encoded.startswith("$argon2id$"):
        try:
            return _ARGON2.verify(encoded, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
    if not encoded.startswith(_PBKDF2_PREFIX):
        return False
    try:
        _, rounds, salt_hex, digest_hex = encoded.split("$", 3)
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate.hex(), digest_hex)


def needs_password_rehash(encoded: str | None) -> bool:
    """Return whether a successful login should migrate the stored hash."""
    if not encoded or encoded.startswith(_PBKDF2_PREFIX):
        return True
    if not encoded.startswith("$argon2id$"):
        return True
    try:
        return _ARGON2.check_needs_rehash(encoded)
    except (InvalidHashError, VerificationError):
        return True


def create_access_token(user: User) -> str:
    now = datetime.now(UTC)
    settings = get_settings()
    return jwt.encode(
        {
            "sub": str(user.id),
            "role": user.role,
            "plan": user.plan,
            "iat": now,
            "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
        },
        settings.jwt_secret,
        algorithm="HS256",
    )


def create_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),  # noqa: B008
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication_required"
        )
    if credentials.credentials.startswith("afk_"):
        key = await session.scalar(
            select(ApiKey).where(
                ApiKey.key_hash == hashlib.sha256(credentials.credentials.encode()).hexdigest(),
                ApiKey.revoked_at.is_(None),
            )
        )
        if key is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")
        user = await session.get(User, key.user_id)
        if user is None or user.status != "active":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_inactive")
        key.last_used_at = datetime.now(UTC)
        request.state.user_id_hash = hashlib.sha256(str(user.id).encode()).hexdigest()
        return user
    try:
        claims = jwt.decode(
            credentials.credentials, get_settings().jwt_secret, algorithms=["HS256"]
        )
        user_id = UUID(claims["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_access_token"
        ) from exc
    user = await session.get(User, user_id)
    if user is None or user.status != "active":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_inactive")
    request.state.user_id_hash = hashlib.sha256(str(user.id).encode()).hexdigest()
    return user


def require_role(*roles: str):
    async def dependency(user: User = Depends(current_user)) -> User:  # noqa: B008
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient_role")
        return user

    return dependency
