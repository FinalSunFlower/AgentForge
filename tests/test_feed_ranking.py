from datetime import UTC, datetime, timedelta
from uuid import uuid4

from services.core_api.app.feed_ranking import (
    deduplicate_brigading,
    hot_match_bonus,
    jaccard_similarity,
    ranking_score,
    uuid_pivot_candidates,
)
from services.core_api.app.models import Post


def make_post(body: str, *, model_ref: str | None = None, age_hours: float = 0) -> Post:
    post = Post(
        id=uuid4(),
        author_id=uuid4(),
        body=body,
        model_ref=model_ref,
        quality_score=0,
        like_count=0,
        view_count=10,
    )
    post.created_at = datetime.now(UTC) - timedelta(hours=age_hours)
    return post


def test_bigram_jaccard_and_hot_bonus_are_deterministic() -> None:
    assert jaccard_similarity("Agent systems", "agent systems") == 1.0
    assert hot_match_bonus("Agent systems", "agent systems") == 4.0
    assert hot_match_bonus("unrelated", "agent systems") == 0.0


def test_ranking_applies_time_decay() -> None:
    fresh, old = make_post("fresh"), make_post("old", age_hours=24)
    assert ranking_score(fresh) > ranking_score(old)


def test_model_burst_duplicates_are_removed() -> None:
    first = make_post("A useful agent architecture", model_ref="model-1")
    second = make_post("A useful agent architecture", model_ref="model-1")
    unique = make_post("Different research note", model_ref="model-1")
    result = deduplicate_brigading([second, unique, first])
    assert len(result) == 2


def test_uuid_pivot_scans_after_then_before() -> None:
    posts = [make_post(str(index)) for index in range(5)]
    posts.sort(key=lambda post: post.id)
    pivot = posts[2].id
    assert uuid_pivot_candidates(posts, pivot, 3)[0].id >= pivot
