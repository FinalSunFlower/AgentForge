from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from services.core_api.app.models import QuotaReservation, ReservationStatus


async def expire_pending_quota_reservations_once(session, *, batch_size: int = 200) -> int:
    """Expire unconfirmed token grants or plan renewals past their reservation window."""
    now = datetime.now(UTC)
    rows = list(
        await session.scalars(
            select(QuotaReservation)
            .where(
                QuotaReservation.status == ReservationStatus.AWAITING_CONFIRM,
                QuotaReservation.expires_at < now,
            )
            .limit(batch_size)
        )
    )
    for reservation in rows:
        reservation.status = ReservationStatus.EXPIRED
    if rows:
        await session.commit()
    return len(rows)
