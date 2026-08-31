from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .embedding_router import select_intent
from .tools import ToolError


class SonarSensor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    bearing_deg: float = Field(ge=-180, le=180)


class SonarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensors: list[SonarSensor] = Field(min_length=2, max_length=32)


class WindTunnelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    velocity_mps: float = Field(gt=0, le=500)
    angle_deg: float = Field(ge=-30, le=30)
    cylinder_radius_m: float = Field(gt=0.001, le=10)
    grid_size: int = Field(default=25, ge=9, le=65)


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
        # Include quoted identifiers and every nested FROM/JOIN occurrence.
        table_matches = re.findall(
            r"\b(?:from|join)\s+(?:\"([^\"]+)\"|`([^`]+)`|([a-z_][a-z0-9_]*))", normalized
        )
        tables = {part for match in table_matches for part in match if part}
        if not tables.issubset({"novels", "chapters", "posts", "usage_daily"}):
            raise ValueError("table_not_allowlisted")
        if " limit " not in f" {normalized} ":
            self.query = f"{self.query.rstrip()} LIMIT 100"
        return self


class IntentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4_000)


def _least_squares(lines: Sequence[SonarSensor]) -> tuple[float, float, float]:
    a11 = a12 = a22 = b1 = b2 = 0.0
    for sensor in lines:
        angle = math.radians(sensor.bearing_deg)
        nx, ny = -math.sin(angle), math.cos(angle)
        rhs = nx * sensor.x + ny * sensor.y
        a11 += nx * nx
        a12 += nx * ny
        a22 += ny * ny
        b1 += nx * rhs
        b2 += ny * rhs
    determinant = a11 * a22 - a12 * a12
    if abs(determinant) < 1e-9:
        raise ToolError("sonar_geometry_singular")
    source_x = (b1 * a22 - b2 * a12) / determinant
    source_y = (a11 * b2 - a12 * b1) / determinant
    residual = 0.0
    for sensor in lines:
        angle = math.radians(sensor.bearing_deg)
        residual += (
            -math.sin(angle) * source_x
            + math.cos(angle) * source_y
            - (-math.sin(angle) * sensor.x + math.cos(angle) * sensor.y)
        ) ** 2
    return source_x, source_y, math.sqrt(residual / len(lines))


async def passive_sonar(arguments: dict[str, Any]) -> dict[str, Any]:
    payload = SonarInput.model_validate(arguments)
    x, y, residual = _least_squares(payload.sensors)
    return {
        "source_x": round(x, 6),
        "source_y": round(y, 6),
        "rms_bearing_residual": round(residual, 6),
        "sensor_count": len(payload.sensors),
        "method": "2d_line_intersection_least_squares",
    }


async def wind_tunnel(arguments: dict[str, Any]) -> dict[str, Any]:
    payload = WindTunnelInput.model_validate(arguments)
    alpha = math.radians(payload.angle_deg)
    radius, velocity, n = payload.cylinder_radius_m, payload.velocity_mps, payload.grid_size
    extent = 3.0 * radius
    samples: list[float] = []
    pressure_sum = 0.0
    for row in range(n):
        y = -extent + 2 * extent * row / (n - 1)
        for column in range(n):
            x = -extent + 2 * extent * column / (n - 1)
            r = math.hypot(x, y)
            if r <= radius:
                continue
            theta = math.atan2(y, x) - alpha
            tangential = 2 * velocity * math.sin(theta) * radius / r
            cp = 1.0 - (tangential / velocity) ** 2
            samples.append(cp)
            pressure_sum += cp
    # Inviscid potential flow predicts zero integrated drag and lift in the ideal case.
    mean_cp = pressure_sum / len(samples)
    return {
        "model": "2d_inviscid_cylinder_potential_flow",
        "velocity_mps": velocity,
        "angle_deg": payload.angle_deg,
        "grid_points": len(samples),
        "mean_pressure_coefficient": round(mean_cp, 8),
        "lift_coefficient": 0.0,
        "drag_coefficient": 0.0,
        "limitations": "boundary-layer and stall effects are not modeled",
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
    "audio": ("audio", "music", "singer", "sing", "voice", "歌曲", "歌手", "音频"),
    "science": ("sonar", "bearing", "triangulate", "wind tunnel", "aerodynamic", "声呐", "风洞"),
    "data": ("sql", "database", "query", "analytics", "数据", "数据库"),
    "math": ("calculate", "equation", "算", "数学"),
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
        "audio": "multimodal.audio",
        "science": "science.experts",
        "data": "private.sql.readonly",
        "math": "calculator",
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
