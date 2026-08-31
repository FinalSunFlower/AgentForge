from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from .models import Post


def character_bigrams(text: str) -> set[str]:
    normalized = "".join(text.lower().split())
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def jaccard_similarity(left: str, right: str) -> float:
    a, b = character_bigrams(left), character_bigrams(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def hot_match_bonus(body: str, query: str | None) -> float:
    if not query:
        return 0.0
    similarity = jaccard_similarity(body, query)
    if similarity >= 0.75:
        return 4.0
    if similarity >= 0.5:
        return 2.0
    if similarity >= 0.25:
        return 0.75
    return 0.0


def ranking_score(post: Post, *, query: str | None = None, now: datetime | None = None) -> float:
    reference = now or datetime.now(UTC)
    created = post.created_at if post.created_at.tzinfo else post.created_at.replace(tzinfo=UTC)
    age_hours = max(0.0, (reference - created).total_seconds() / 3600)
    return (
        (post.view_count + hot_match_bonus(post.body, query or post.hot_query))
        / ((age_hours + 2.0) ** 1.5)
        + post.quality_score
        + post.like_count * 0.05
    )


def deduplicate_brigading(posts: list[Post], *, similarity_threshold: float = 0.88) -> list[Post]:
    """Keep the strongest post from the same model burst within a three-hour window."""
    accepted: list[Post] = []
    for post in sorted(posts, key=lambda item: ranking_score(item, query=None), reverse=True):
        duplicate = False
        for existing in accepted:
            if post.model_ref and post.model_ref == existing.model_ref:
                created = (
                    post.created_at
                    if post.created_at.tzinfo
                    else post.created_at.replace(tzinfo=UTC)
                )
                other = (
                    existing.created_at
                    if existing.created_at.tzinfo
                    else existing.created_at.replace(tzinfo=UTC)
                )
                within_window = abs((created - other).total_seconds()) <= 3 * 3600
                if (
                    within_window
                    and jaccard_similarity(post.body, existing.body) >= similarity_threshold
                ):
                    duplicate = True
                    break
        if not duplicate:
            accepted.append(post)
    return accepted


def uuid_pivot_candidates(posts: list[Post], pivot: UUID, limit: int) -> list[Post]:
    """Reference implementation for tests and non-SQL adapters."""
    ordered = sorted(posts, key=lambda item: item.id)
    after = [post for post in ordered if post.id >= pivot]
    before = [post for post in ordered if post.id < pivot]
    return (after + before)[:limit]
