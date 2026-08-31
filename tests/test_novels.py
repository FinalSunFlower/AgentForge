import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from services.core_api.app.db import SessionFactory
from services.core_api.app.main import app
from services.core_api.app.models import ReadingProgress


def test_reading_progress_only_moves_forward() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        user = client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Reader", "password": password},
        ).json()
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}
        novel = client.post(
            "/v1/novels", headers=headers, json={"title": "Test novel", "author_name": "Author"}
        ).json()
        chapter = client.post(
            f"/v1/novels/{novel['id']}/chapters",
            headers=headers,
            json={"number": 1, "title": "One", "content": "content"},
        ).json()
        newer = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
        response = client.put(
            f"/v1/novels/{novel['id']}/reading-progress",
            headers=headers,
            json={
                "chapter_id": chapter["id"],
                "chapter_number": 1,
                "progress_percent": 80,
                "paragraph_index": 8,
                "client_updated_at": newer,
            },
        )
        assert response.status_code == 200 and response.json()["accepted"] is True
        stale = client.put(
            f"/v1/novels/{novel['id']}/reading-progress",
            headers=headers,
            json={
                "chapter_id": chapter["id"],
                "chapter_number": 1,
                "progress_percent": 20,
                "paragraph_index": 2,
                "client_updated_at": (datetime.now(UTC) + timedelta(seconds=2)).isoformat(),
            },
        )
        assert stale.status_code == 200
        assert stale.json()["accepted"] is False
        assert stale.json()["stale_reason"] == "progress_regression"

        second_chapter = client.post(
            f"/v1/novels/{novel['id']}/chapters",
            headers=headers,
            json={"number": 2, "title": "Two", "content": "content"},
        ).json()
        cross_chapter_regression = client.put(
            f"/v1/novels/{novel['id']}/reading-progress",
            headers=headers,
            json={
                "chapter_id": second_chapter["id"],
                "chapter_number": 2,
                "progress_percent": 0,
                "paragraph_index": 0,
                "client_updated_at": (datetime.now(UTC) + timedelta(seconds=3)).isoformat(),
            },
        )
        assert cross_chapter_regression.status_code == 200
        assert cross_chapter_regression.json()["accepted"] is False
        assert cross_chapter_regression.json()["stale_reason"] == "progress_regression"
        assert user["id"]


def test_reading_progress_reset_and_timestamp_regression() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Reader", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}
        novel = client.post(
            "/v1/novels", headers=headers, json={"title": "Reset novel", "author_name": "Author"}
        ).json()
        chapter = client.post(
            f"/v1/novels/{novel['id']}/chapters",
            headers=headers,
            json={"number": 1, "title": "One", "content": "content"},
        ).json()
        first_ts = datetime.now(UTC).isoformat()
        accepted = client.put(
            f"/v1/novels/{novel['id']}/reading-progress",
            headers=headers,
            json={
                "chapter_id": chapter["id"],
                "chapter_number": 1,
                "progress_percent": 40,
                "paragraph_index": 4,
                "client_updated_at": first_ts,
            },
        )
        assert accepted.json()["accepted"] is True
        older = client.put(
            f"/v1/novels/{novel['id']}/reading-progress",
            headers=headers,
            json={
                "chapter_id": chapter["id"],
                "chapter_number": 1,
                "progress_percent": 50,
                "paragraph_index": 5,
                "client_updated_at": (datetime.now(UTC) - timedelta(seconds=30)).isoformat(),
            },
        )
        assert older.json()["accepted"] is False
        assert older.json()["stale_reason"] == "client_timestamp_older"
        reset = client.post(f"/v1/novels/{novel['id']}/reading-progress/reset", headers=headers)
        assert reset.status_code == 204
        restarted = client.put(
            f"/v1/novels/{novel['id']}/reading-progress",
            headers=headers,
            json={
                "chapter_id": chapter["id"],
                "chapter_number": 1,
                "progress_percent": 5,
                "paragraph_index": 1,
                "client_updated_at": datetime.now(UTC).isoformat(),
            },
        )
        assert restarted.json()["accepted"] is True


def test_concurrent_first_progress_writes_leave_one_row() -> None:
    with TestClient(app) as client:
        email = f"{uuid4()}@example.com"
        password = "correct horse battery staple"
        client.post(
            "/v1/auth/register",
            json={"email": email, "display_name": "Reader", "password": password},
        )
        token = client.post("/v1/auth/login", json={"email": email, "password": password}).json()[
            "access_token"
        ]
        headers = {"Authorization": f"Bearer {token}"}
        novel = client.post(
            "/v1/novels", headers=headers, json={"title": "Race novel", "author_name": "Author"}
        ).json()
        chapter = client.post(
            f"/v1/novels/{novel['id']}/chapters",
            headers=headers,
            json={"number": 1, "title": "One", "content": "content"},
        ).json()

        def write(percent: int, offset: int) -> dict:
            response = client.put(
                f"/v1/novels/{novel['id']}/reading-progress",
                headers=headers,
                json={
                    "chapter_id": chapter["id"],
                    "chapter_number": 1,
                    "progress_percent": percent,
                    "paragraph_index": percent,
                    "client_updated_at": (
                        datetime.now(UTC) + timedelta(seconds=offset)
                    ).isoformat(),
                },
            )
            return {"status": response.status_code, "body": response.json()}

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(write, 30, 1)
            second = pool.submit(write, 80, 2)
            results = [first.result(), second.result()]
        assert all(item["status"] == 200 for item in results)
        accepted = [item["body"] for item in results if item["body"].get("accepted")]
        assert accepted
        assert max(item["progress_percent"] for item in accepted) in {30, 80}

        async def count_rows() -> tuple[int, str]:
            async with SessionFactory() as session:
                dialect = session.bind.dialect.name if session.bind is not None else ""
                total = await session.scalar(
                    select(func.count())
                    .select_from(ReadingProgress)
                    .where(ReadingProgress.novel_id == UUID(novel["id"]))
                )
                return int(total or 0), dialect

        count, dialect = asyncio.run(count_rows())
        assert count == 1
        assert dialect in {"sqlite", "postgresql"}
