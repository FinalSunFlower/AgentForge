from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .embedding_router import select_intent

_ARXIV_CATALOG: tuple[dict[str, str], ...] = (
    {
        "arxiv_id": "2305.14992",
        "title": "Reasoning with language model is planning with world model",
        "abstract": "Tree search over a world model can improve multi-step reasoning without extra training.",
    },
    {
        "arxiv_id": "2210.03629",
        "title": "ReAct: synergizing reasoning and acting in language models",
        "abstract": "Interleaving thought traces with tool actions improves grounded question answering.",
    },
    {
        "arxiv_id": "2005.14165",
        "title": "Retrieval-augmented generation for knowledge-intensive NLP",
        "abstract": "Hybrid retrieval supplies cited passages so a generator can stay extractive.",
    },
    {
        "arxiv_id": "1704.08863",
        "title": "Okapi BM25 and lexical saturation",
        "abstract": "Term frequency saturates with k1 and document length is normalized by b.",
    },
)


class ArxivSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=400)
    limit: int = Field(default=3, ge=1, le=8)


class PlotInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    xs: list[float] = Field(min_length=2, max_length=64)
    ys: list[float] = Field(min_length=2, max_length=64)
    title: str = Field(default="series", max_length=80)

    @model_validator(mode="after")
    def matching_lengths(self) -> PlotInput:
        if len(self.xs) != len(self.ys):
            raise ValueError("series_length_mismatch")
        return self


class SQLAnalyticsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=7, max_length=2_000)

    @model_validator(mode="after")
    def validate_read_only(self) -> SQLAnalyticsInput:
        normalized = re.sub(r"\s+", " ", self.query.strip().lower())
        blocked = r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum|grant|revoke|detach|reindex)\b"
        if (
            re.search(blocked, normalized)
            or "--" in normalized
            or "/*" in normalized
            or "*/" in normalized
        ):
            raise ValueError("unsafe_sql_rejected")
        if not re.match(r"^(select|with)\b", normalized) or ";" in normalized:
            raise ValueError("read_only_select_required")
        table_matches = re.findall(
            r"\b(?:from|join)\s+(?:\"([^\"]+)\"|`([^`]+)`|([a-z_][a-z0-9_]*))", normalized
        )
        tables = {part for match in table_matches for part in match if part}
        if not tables.issubset({"papers", "paper_sections", "posts", "usage_daily"}):
            raise ValueError("table_not_allowlisted")
        if " limit " not in f" {normalized} ":
            self.query = f"{self.query.rstrip()} LIMIT 100"
        return self


class IntentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4_000)


def _score_paper(query: str, paper: dict[str, str]) -> int:
    haystack = f"{paper['arxiv_id']} {paper['title']} {paper['abstract']}".lower()
    return sum(1 for token in re.findall(r"[a-z0-9]+", query.lower()) if token in haystack)


async def arxiv_search(arguments: dict[str, Any]) -> dict[str, Any]:
    payload = ArxivSearchInput.model_validate(arguments)
    ranked = sorted(
        ((_score_paper(payload.query, paper), paper) for paper in _ARXIV_CATALOG),
        key=lambda item: item[0],
        reverse=True,
    )
    hits = [{**paper, "score": score} for score, paper in ranked if score > 0][: payload.limit]
    if not hits:
        hits = [{**paper, "score": 0} for paper in _ARXIV_CATALOG[: payload.limit]]
    return {
        "query": payload.query,
        "method": "closed_catalog_keyword",
        "source": "local_arxiv_style_catalog",
        "results": hits,
        "claim": "Deterministic catalog lookup. Not a live arXiv HTTP client.",
    }


def _svg_polyline(xs: list[float], ys: list[float]) -> str:
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    points = []
    for x, y in zip(xs, ys, strict=True):
        px = 8 + 184 * (x - x_min) / x_span
        py = 72 - 56 * (y - y_min) / y_span
        points.append(f"{px:.1f},{py:.1f}")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 80">'
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{" ".join(points)}"/>'
        "</svg>"
    )


async def plot_generator(arguments: dict[str, Any]) -> dict[str, Any]:
    payload = PlotInput.model_validate(arguments)
    mean = sum(payload.ys) / len(payload.ys)
    variance = sum((value - mean) ** 2 for value in payload.ys) / len(payload.ys)
    slope = (payload.ys[-1] - payload.ys[0]) / (payload.xs[-1] - payload.xs[0] or 1.0)
    return {
        "title": payload.title,
        "n": len(payload.ys),
        "mean": round(mean, 6),
        "stdev": round(math.sqrt(variance), 6),
        "slope": round(slope, 6),
        "svg": _svg_polyline(payload.xs, payload.ys),
        "method": "closed_form_series_stats",
    }


async def readonly_sql(arguments: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy import text

    from .db import SessionFactory

    payload = SQLAnalyticsInput.model_validate(arguments)
    async with SessionFactory() as session:
        result = await session.execute(text(payload.query))
        rows = [dict(row) for row in result.mappings().fetchmany(100)]
    return {
        "columns": list(rows[0]) if rows else [],
        "rows": rows,
        "row_count": len(rows),
        "read_only": True,
    }


_INTENT_KEYWORDS = {
    "research": ("arxiv", "paper", "citation", "literature", "abstract", "preprint"),
    "data": ("sql", "database", "query", "analytics", "plot", "chart", "series"),
    "math": ("calculate", "equation", "expression"),
    "writing": ("summarize", "fact-check", "align", "abstract"),
}


async def intent_router(arguments: dict[str, Any]) -> dict[str, Any]:
    payload = IntentInput.model_validate(arguments)
    text = payload.text.lower()
    keyword_scores = {
        intent: sum(1 for keyword in keywords if keyword in text)
        for intent, keywords in _INTENT_KEYWORDS.items()
    }
    keyword_selected, keyword_score = max(keyword_scores.items(), key=lambda item: item[1])
    if keyword_score == 0:
        keyword_selected, keyword_confidence = "general_text", 0.35
    else:
        keyword_confidence = min(0.99, 0.55 + keyword_score * 0.12)
    embedding_selected, embedding_scores = select_intent(payload.text)
    selected = embedding_selected
    if embedding_selected == "general_text" and keyword_selected != "general_text":
        selected = keyword_selected
    adapters = {
        "research": "arxiv.catalog",
        "data": "plot_or_sql",
        "math": "calculator",
        "writing": "retrieval.citations",
        "general_text": "llm.text",
    }
    return {
        "intent": selected,
        "method": "minilm_embedding_with_keyword_fallback",
        "confidence": round(max(embedding_scores.values()), 4) if embedding_scores else 0.0,
        "keyword": {
            "intent": keyword_selected,
            "confidence": round(keyword_confidence, 4),
            "scores": keyword_scores,
        },
        "embedding": {
            "intent": embedding_selected,
            "scores": {key: round(value, 4) for key, value in embedding_scores.items()},
        },
        "claim": "Keyword counts and MiniLM cosine over frozen intent prototypes. Not live-LLM routing.",
        "next_adapter": adapters[selected],
        "scores": keyword_scores,
    }
