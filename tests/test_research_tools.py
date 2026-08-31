import pytest
from pydantic import ValidationError

from services.agent_runtime.app.research_tools import arxiv_search, intent_router, plot_generator


@pytest.mark.asyncio
async def test_arxiv_search_ranks_react_preprint() -> None:
    result = await arxiv_search({"query": "ReAct tool actions", "limit": 3})
    assert result["method"] == "closed_catalog_keyword"
    assert result["results"]
    assert any(hit["arxiv_id"] == "2210.03629" for hit in result["results"])


@pytest.mark.asyncio
async def test_plot_generator_returns_closed_form_stats() -> None:
    result = await plot_generator(
        {"xs": [0.0, 1.0, 2.0], "ys": [1.0, 3.0, 5.0], "title": "linear"}
    )
    assert result["n"] == 3
    assert result["mean"] == 3.0
    assert result["slope"] == 2.0
    assert "<svg" in result["svg"]


@pytest.mark.asyncio
async def test_intent_router_selects_research_and_data_adapters() -> None:
    research = await intent_router({"text": "search the arxiv literature catalog"})
    data = await intent_router({"text": "plot this numeric series for analytics"})
    assert research["intent"] == "research"
    assert research["next_adapter"] == "arxiv.catalog"
    assert data["intent"] == "data"


def test_readonly_sql_contract_rejects_writes_and_unknown_tables() -> None:
    from services.agent_runtime.app.research_tools import SQLAnalyticsInput

    with pytest.raises(ValidationError, match="unsafe_sql_rejected"):
        SQLAnalyticsInput(query="DELETE FROM posts")
    with pytest.raises(ValidationError, match="table_not_allowlisted"):
        SQLAnalyticsInput(query="SELECT * FROM users")
