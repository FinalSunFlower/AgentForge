from __future__ import annotations

from .hybrid_retrieval import cosine, default_embedding

# Frozen catalog text. This is MiniLM nearest-neighbor routing, not a learned
# policy and not live-LLM tool selection.
TOOL_PROTOTYPES: tuple[tuple[str, str], ...] = (
    (
        "intent_router",
        "Classify the request intent or adapter: math, research, data, writing, or general text.",
    ),
    (
        "readonly_sql",
        "Run a read-only SQL SELECT analytics query against the papers database.",
    ),
    (
        "arxiv_search",
        "Search an arXiv-style paper catalog by title, abstract, or preprint identifier.",
    ),
    (
        "plot_generator",
        "Plot a numeric series and report mean, standard deviation, and slope.",
    ),
    (
        "calculator",
        "Evaluate a numeric arithmetic expression. Calculate or compute sums and products.",
    ),
    (
        "retrieval",
        "Search, retrieve, find, or look up published papers, sections, primers, and authors.",
    ),
    (
        "handoff",
        "Transfer this run to the code-data specialist, retrieval specialist, or writer with a brief.",
    ),
)

SPECIALIST_PROTOTYPES: tuple[tuple[str, str], ...] = (
    (
        "code_data",
        "Code and data specialist for calculate and compute arithmetic, plot a series, and read-only SQL.",
    ),
    (
        "retrieval",
        "Retrieval specialist for searching published notes and looking up document passages.",
    ),
    (
        "writer",
        "Academic writer for citation-grounded summaries and extractive fact alignment.",
    ),
)

CANONICAL_TOOL_ORDER: tuple[str, ...] = tuple(name for name, _ in TOOL_PROTOTYPES)
_SCORE_FLOOR = 0.16
_SPECIALIST_FLOOR = 0.05
_RELATIVE_KEEP = 0.72
_RETRIEVAL_HINTS = ("search", "retrieve", "find", "look up", "who wrote", "when do")
_SQL_HINTS = ("select ", " sql", "sql ", "analytics")
_ARXIV_HINTS = ("arxiv literature catalog", "literature catalog", "use the arxiv")
_HANDOFF_HINTS = ("handoff", "transfer", "specialist")
_INTENT_HINTS = ("classify", "intent", "adapter")


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _blocked(name: str, prompt: str) -> bool:
    return name == "retrieval" and _contains_any(
        prompt, ("without looking", "without search", "without retrieving")
    )


def _embed_scores(prompt: str, prototypes: tuple[tuple[str, str], ...]) -> list[tuple[str, float]]:
    embedder = default_embedding()
    texts = [prompt, *[text for _, text in prototypes]]
    vectors = embedder.embed_many(texts)
    query = vectors[0]
    return [
        (name, cosine(query, vector))
        for (name, _), vector in zip(prototypes, vectors[1:], strict=True)
    ]


def select_tool_names(prompt: str) -> list[str]:
    """Return catalog tools whose MiniLM description is close to the prompt."""
    if _contains_any(prompt, _INTENT_HINTS):
        return ["intent_router"]
    scored = [
        (name, score)
        for name, score in _embed_scores(prompt, TOOL_PROTOTYPES)
        if not _blocked(name, prompt)
    ]
    if not scored:
        return []
    top_name, top_score = max(scored, key=lambda item: item[1])
    if top_score < _SCORE_FLOOR:
        if _contains_any(prompt, _RETRIEVAL_HINTS):
            return ["retrieval"]
        return []
    kept = [
        name for name, score in scored if score >= max(_SCORE_FLOOR, _RELATIVE_KEEP * top_score)
    ]
    if _contains_any(prompt, _SQL_HINTS):
        kept = [name for name in kept if name == "readonly_sql"]
        if "readonly_sql" not in kept:
            kept.append("readonly_sql")
    elif _contains_any(prompt, _ARXIV_HINTS):
        kept = [name for name in kept if name not in {"retrieval", "plot_generator"}]
        if "arxiv_search" not in kept:
            kept.append("arxiv_search")
    elif _contains_any(prompt, _RETRIEVAL_HINTS):
        kept = [name for name in kept if name not in {"arxiv_search", "plot_generator"}]
        if "retrieval" not in kept:
            kept.append("retrieval")
    kept = [
        name for name in kept if name != "intent_router" or _contains_any(prompt, _INTENT_HINTS)
    ]
    kept = [name for name in kept if name != "handoff" or _contains_any(prompt, _HANDOFF_HINTS)]
    if top_name == "intent_router":
        margin = top_score - max(
            (score for name, score in scored if name != "intent_router"), default=0.0
        )
        if margin >= 0.02:
            return ["intent_router"]
    order = {name: index for index, name in enumerate(CANONICAL_TOOL_ORDER)}
    return sorted(kept, key=lambda name: order.get(name, 99))


def select_specialist(prompt: str) -> str | None:
    scored = _embed_scores(prompt, SPECIALIST_PROTOTYPES)
    if not scored:
        return None
    name, score = max(scored, key=lambda item: item[1])
    if score < _SPECIALIST_FLOOR:
        return None
    return name


INTENT_PROTOTYPES: tuple[tuple[str, str], ...] = (
    ("math", "Calculate or compute an arithmetic expression or equation."),
    ("research", "Search arXiv-style papers, citations, abstracts, or literature."),
    ("data", "Read-only SQL analytics query or plot a numeric series."),
    ("writing", "Summarize cited passages and fact-check an extractive claim."),
)


def select_intent(prompt: str) -> tuple[str, dict[str, float]]:
    scored = _embed_scores(prompt, INTENT_PROTOTYPES)
    mapping = dict(scored)
    if not scored:
        return "general_text", mapping
    name, score = max(scored, key=lambda item: item[1])
    if score < _SCORE_FLOOR:
        return "general_text", mapping
    return name, mapping


def prefer_tool_order(preferred: list[str], catalog_names: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in [*preferred, *catalog_names]:
        if name in catalog_names and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
