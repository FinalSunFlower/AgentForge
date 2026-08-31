from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from services.core_api.app.models import PushDelivery


async def dispatch_push_once(session, *, batch_size: int = 100, max_attempts: int = 5) -> int:
    """Deliver queued notifications through the local adapter with bounded retry.

    The adapter is intentionally simulated in the local profile; delivery state is
    still durable and can be replaced by APNs/FCM/Web Push without changing callers.
    """
    now = datetime.now(UTC)
    rows = list(
        await session.scalars(
            select(PushDelivery)
            .where(
                PushDelivery.status == "queued",
                (PushDelivery.next_attempt_at.is_(None) | (PushDelivery.next_attempt_at <= now)),
            )
            .order_by(PushDelivery.created_at)
            .limit(batch_size)
        )
    )
    for delivery in rows:
        delivery.attempts += 1
        delivery.status = (
            "sent"
            if delivery.channel == "simulated"
            else ("dead_letter" if delivery.attempts >= max_attempts else "queued")
        )
        delivery.provider_message_id = f"sim-{delivery.id}" if delivery.status == "sent" else None
        if delivery.status == "queued":
            delivery.next_attempt_at = now + timedelta(seconds=min(300, 2**delivery.attempts))
    await session.commit()
    return len(rows)
