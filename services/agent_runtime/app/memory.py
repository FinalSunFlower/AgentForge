from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_CODE_RE = re.compile(r"\b[A-Z]{2,}[-_][A-Z0-9]{2,}\b")
_FACT_RE = re.compile(
    r"\b(i (prefer|like|need|want|use)|my (name|timezone|plan|code) is|remember that|"
    r"the (vault |access )?code is|pin is)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d{2,}\b")


@dataclass
class MemoryFact:
    text: str
    source_index: int
    kind: str


@dataclass
class AssembledContext:
    messages: list[dict[str, Any]]
    compressed: bool
    report: dict[str, Any] = field(default_factory=dict)


def extract_facts(messages: list[dict[str, Any]]) -> list[MemoryFact]:
    """Keep structured facts; drop chit-chat. This is extractive, not LLM summary.

    IDs, explicit preference keywords, numbers and short successful tool results
    are in scope. Implicit preferences in ordinary prose are not.
    """
    facts: list[MemoryFact] = []
    seen: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role", "")
        content = (message.get("content") or "").strip()
        if not content or role == "system":
            continue
        keep = False
        kind = "context"
        if _CODE_RE.search(content) or _FACT_RE.search(content):
            keep = True
            kind = "explicit_fact"
        elif role == "user" and _NUMBER_RE.search(content) and len(content) < 280:
            keep = True
            kind = "numeric_fact"
        elif role == "tool" and '"ok":true' in content.replace(" ", "").lower():
            keep = True
            kind = "tool_result"
            content = content[:240]
        elif role == "user" and any(
            mark in content.lower() for mark in ("prefer", "remember", "always", "never")
        ):
            keep = True
            kind = "preference"
        if not keep:
            continue
        key = re.sub(r"\s+", " ", content.lower())
        if key in seen:
            continue
        seen.add(key)
        facts.append(MemoryFact(text=content, source_index=index, kind=kind))
    return facts


def assemble_context(
    history: list[dict[str, Any]],
    *,
    system_policy: str,
    max_chars: int,
    keep_recent: int = 6,
    compress_after: int = 12,
) -> AssembledContext:
    """Window recent turns and fold earlier facts into a compact memory block.

    Older turns are not blindly truncated: extractive facts stay in context so a
    needle mentioned many turns ago can still be recovered after compression.
    """
    usable = max_chars - len(system_policy)
    recent_count = min(max(1, keep_recent), len(history))
    must_compress = (
        len(history) > compress_after
        or sum(len(item.get("content") or "") for item in history) > usable
    )
    if not must_compress:
        clipped: list[dict[str, Any]] = []
        remaining = usable
        for message in reversed(history):
            if remaining <= 0:
                break
            content = (message.get("content") or "")[-remaining:]
            clipped.append({"role": message.get("role", "user"), "content": content})
            remaining -= len(content)
        return AssembledContext(
            messages=[{"role": "system", "content": system_policy}, *reversed(clipped)],
            compressed=False,
            report={"facts_kept": 0, "dropped_turns": 0, "kept_recent": len(clipped)},
        )

    older = history[:-recent_count] if recent_count < len(history) else []
    recent = history[-recent_count:]
    facts = extract_facts(older)
    memory_lines = [f"- ({fact.kind}, turn {fact.source_index + 1}) {fact.text}" for fact in facts]
    memory_block = (
        "Retained facts from earlier turns:\n" + "\n".join(memory_lines) if memory_lines else ""
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_policy}]
    if memory_block:
        messages.append({"role": "system", "content": memory_block[: max(0, usable // 2)]})
    remaining = usable - sum(len(item.get("content") or "") for item in messages)
    for message in recent:
        content = (message.get("content") or "")[: max(0, remaining)]
        messages.append({"role": message.get("role", "user"), "content": content})
        remaining -= len(content)
        if remaining <= 0:
            break
    return AssembledContext(
        messages=messages,
        compressed=True,
        report={
            "facts_kept": len(facts),
            "dropped_turns": len(older),
            "kept_recent": len(recent),
            "fact_kinds": [fact.kind for fact in facts],
        },
    )


def build_needle_thread(
    needle: str, *, filler_turns: int = 60, needle_at: int = 4
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for index in range(filler_turns):
        if index == needle_at:
            history.append({"role": "user", "content": needle})
            history.append({"role": "assistant", "content": "Noted."})
            continue
        history.append(
            {"role": "user", "content": f"Filler turn {index}: please wait while I check status."}
        )
        history.append({"role": "assistant", "content": f"Acknowledged filler {index}."})
    history.append({"role": "user", "content": "What is the vault code I told you earlier?"})
    return history
