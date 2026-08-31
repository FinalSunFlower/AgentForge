from __future__ import annotations

import re
from typing import Any

from .hybrid_retrieval import tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_PASSAGE_ID_RE = re.compile(r"\b(?:p-[\w-]+|(?:novel|chapter):[0-9a-f-]+:\d+)\b", re.IGNORECASE)
_DEFAULT_THRESHOLD = 0.35


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def citation_overlap(sentence: str, passages: list[str]) -> float:
    tokens = set(tokenize(sentence))
    if not tokens:
        return 1.0
    best = 0.0
    for passage in passages:
        overlap = len(tokens & set(tokenize(passage)))
        best = max(best, overlap / len(tokens))
    return best


def extract_cited_ids(answer: str, available_ids: set[str] | None = None) -> list[str]:
    found = [match.group(0) for match in _PASSAGE_ID_RE.finditer(answer)]
    if available_ids is None:
        return list(dict.fromkeys(found))
    return [item for item in dict.fromkeys(found) if item in available_ids]


def check_grounding(
    answer: str,
    passages: list[str],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    cited_ids: list[str] | None = None,
) -> dict[str, Any]:
    sentences = split_sentences(answer)
    unsupported: list[dict[str, Any]] = []
    supported = 0
    for sentence in sentences:
        overlap = citation_overlap(sentence, passages)
        if overlap < threshold:
            unsupported.append({"sentence": sentence, "overlap": round(overlap, 4)})
        else:
            supported += 1
    return {
        "grounded": not unsupported,
        "supported_sentences": supported,
        "unsupported_sentences": unsupported,
        "citation_ids": cited_ids or [],
        "passage_count": len(passages),
        "threshold": threshold,
    }
