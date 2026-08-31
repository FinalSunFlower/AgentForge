from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScheduledCall:
    tool_call_id: str
    name: str
    arguments: str
    side_effect: str = "none"


def _depends_on_prior(arguments: str, prior_ids: set[str]) -> bool:
    return any(call_id and call_id in arguments for call_id in prior_ids)


def schedule_batches(calls: list[ScheduledCall]) -> list[list[ScheduledCall]]:
    """Group independent read-only tool calls; isolate writes and data dependencies.

    A later call is dependent when its argument string mentions an earlier
    tool_call_id. Write tools always run alone so side effects stay ordered.
    """
    batches: list[list[ScheduledCall]] = []
    current: list[ScheduledCall] = []
    seen_ids: set[str] = set()
    for call in calls:
        write = call.side_effect != "none"
        dependent = _depends_on_prior(call.arguments, seen_ids)
        if write or dependent:
            if current:
                batches.append(current)
                current = []
            batches.append([call])
        else:
            current.append(call)
        seen_ids.add(call.tool_call_id)
    if current:
        batches.append(current)
    return batches


def calls_from_deltas(deltas: list[Any], side_effects: dict[str, str]) -> list[ScheduledCall]:
    return [
        ScheduledCall(
            tool_call_id=delta.tool_call_id,
            name=delta.name,
            arguments=delta.arguments or "",
            side_effect=side_effects.get(delta.name, "none"),
        )
        for delta in deltas
    ]
