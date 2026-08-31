from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import cast

from .environment import Action, Domain, State, Transition


def write_jsonl(path: Path, transitions: list[Transition]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for transition in transitions:
            record = asdict(transition)
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterator[Transition]:
    """Read normalized trajectories exported by an external environment adapter."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            yield Transition(
                State(**record["state"]),
                Action(**record["action"]),
                State(**record["next_state"]),
                float(record["reward"]),
                bool(record["done"]),
                bool(record["success"]),
            )


def alfworld_record_to_transition(record: dict) -> Transition:
    """Normalize a manually exported ALFWorld-style transition.

    The adapter requires explicit numeric progress fields; it refuses ambiguous
    records instead of inventing a state representation.
    """
    required = {
        "domain",
        "position",
        "target",
        "progress",
        "steps",
        "action",
        "next_state",
        "reward",
        "done",
        "success",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"trajectory_record_missing_fields:{sorted(missing)}")
    hidden = int(record.get("hidden", 0))
    state = State(
        cast(Domain, record["domain"]),
        int(record["position"]),
        int(record["target"]),
        int(record["progress"]),
        int(record["steps"]),
        hidden,
    )
    next_record = record["next_state"]
    next_state = State(
        cast(Domain, next_record["domain"]),
        int(next_record["position"]),
        int(next_record["target"]),
        int(next_record["progress"]),
        int(next_record["steps"]),
        int(next_record.get("hidden", hidden)),
    )
    action = Action(str(record["action"]["name"]), int(record["action"]["argument"]))
    return Transition(
        state,
        action,
        next_state,
        float(record["reward"]),
        bool(record["done"]),
        bool(record["success"]),
    )
