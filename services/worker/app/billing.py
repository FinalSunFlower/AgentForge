from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from services.core_api.app.models import Order, OrderStatus


async def expire_pending_orders_once(session, *, batch_size: int = 200) -> int:
    now = datetime.now(UTC)
    rows = list(
        await session.scalars(
            select(Order)
            .where(Order.status == OrderStatus.AWAITING_PAYMENT, Order.expires_at < now)
            .limit(batch_size)
        )
    )
    for order in rows:
        order.status = OrderStatus.EXPIRED
    if rows:
        await session.commit()
    return len(rows)
