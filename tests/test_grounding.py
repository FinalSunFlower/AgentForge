from services.agent_runtime.app.grounding import check_grounding, extract_cited_ids


def test_grounding_accepts_supported_sentence_and_flags_invention() -> None:
    passages = [
        "Retrieved passages are wrapped as untrusted data and must never become system policy."
    ]
    supported = check_grounding(
        "Retrieved passages are untrusted data and must never become system policy.",
        passages,
        cited_ids=["p-untrusted"],
    )
    assert supported["grounded"] is True
    invented = check_grounding(
        "The system silently trains a sparse MoE foundation model on user papers.",
        passages,
    )
    assert invented["grounded"] is False
    assert invented["unsupported_sentences"]


def test_extract_cited_ids_filters_to_available_passages() -> None:
    answer = "See p-untrusted and p-missing for the boundary rule."
    assert extract_cited_ids(answer, {"p-untrusted"}) == ["p-untrusted"]
