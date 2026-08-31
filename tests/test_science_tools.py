import math

import pytest
from pydantic import ValidationError

from services.agent_runtime.app.science_tools import intent_router, passive_sonar, wind_tunnel


@pytest.mark.asyncio
async def test_passive_sonar_triangulates_known_source() -> None:
    source = (3.0, 4.0)
    sensors = []
    for x, y in ((0.0, 0.0), (6.0, 0.0), (0.0, 8.0)):
        sensors.append(
            {"x": x, "y": y, "bearing_deg": math.degrees(math.atan2(source[1] - y, source[0] - x))}
        )
    result = await passive_sonar({"sensors": sensors})
    assert abs(result["source_x"] - source[0]) < 1e-5
    assert abs(result["source_y"] - source[1]) < 1e-5
    assert result["rms_bearing_residual"] < 1e-5


@pytest.mark.asyncio
async def test_wind_tunnel_returns_physical_model_metadata() -> None:
    result = await wind_tunnel(
        {"velocity_mps": 30, "angle_deg": 5, "cylinder_radius_m": 0.2, "grid_size": 11}
    )
    assert result["model"] == "2d_inviscid_cylinder_potential_flow"
    assert result["grid_points"] > 0
    assert result["drag_coefficient"] == 0.0


@pytest.mark.asyncio
async def test_intent_router_selects_science_and_audio_adapters() -> None:
    science = await intent_router({"text": "triangulate passive sonar bearings"})
    audio = await intent_router({"text": "write a song for a virtual singer"})
    assert science["intent"] == "science"
    assert science["next_adapter"] == "science.experts"
    assert audio["intent"] == "audio"


def test_readonly_sql_contract_rejects_writes_and_unknown_tables() -> None:
    from services.agent_runtime.app.science_tools import SQLAnalyticsInput

    with pytest.raises(ValidationError, match="unsafe_sql_rejected"):
        SQLAnalyticsInput(query="DELETE FROM posts")
    with pytest.raises(ValidationError, match="table_not_allowlisted"):
        SQLAnalyticsInput(query="SELECT * FROM users")
