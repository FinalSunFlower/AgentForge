from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .agents import Agent
from .environment import DecisionWorld, Domain


@dataclass(frozen=True)
class EpisodeRecord:
    agent: str
    domain: Domain
    seed: int
    success: bool
    steps: int
    cost: float
    foresight_calls: int
    expert_calls: int
    prediction_mae: float
    terminal_brier: float
    tokens: int = 0
    wall_ms: float = 0.0


def bootstrap_ci(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        np.asarray(values, dtype=float), size=(samples, len(values)), replace=True
    ).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def evaluate(
    agent_factory: Callable[[DecisionWorld], Agent],
    world: DecisionWorld,
    agent_name: str,
    episodes: int,
    seed: int,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for index in range(episodes):
        episode_seed = seed + index
        domain = world.domains[index % len(world.domains)]
        initial = world.reset(episode_seed, domain)
        agent = agent_factory(world)
        success, stats, steps = agent.run(initial)
        records.append(
            EpisodeRecord(
                agent_name,
                domain,
                episode_seed,
                success,
                steps,
                stats.cost,
                stats.foresight_calls,
                stats.expert_calls,
                float(np.mean(stats.prediction_errors)) if stats.prediction_errors else 0.0,
                float(np.mean(np.square(stats.terminal_errors))) if stats.terminal_errors else 0.0,
                stats.tokens,
                stats.wall_ms,
            )
        )
    return records


def _row(agent: str, domain: str, items: list[EpisodeRecord], seed: int) -> dict[str, object]:
    success = [float(item.success) for item in items]
    low, high = bootstrap_ci(success, seed)
    return {
        "agent": agent,
        "domain": domain,
        "episodes": len(items),
        "success_rate": float(np.mean(success)),
        "success_ci_low": low,
        "success_ci_high": high,
        "mean_cost": float(np.mean([item.cost for item in items])),
        "mean_prediction_mae": float(np.mean([item.prediction_mae for item in items])),
        "mean_terminal_brier": float(np.mean([item.terminal_brier for item in items])),
        "mean_foresight_calls": float(np.mean([item.foresight_calls for item in items])),
        "mean_expert_calls": float(np.mean([item.expert_calls for item in items])),
        "mean_tokens": float(np.mean([item.tokens for item in items])),
        "mean_wall_ms": float(np.mean([item.wall_ms for item in items])),
    }


def summarize(records: list[EpisodeRecord], seed: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[EpisodeRecord]] = {}
    pooled: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        grouped.setdefault((record.agent, record.domain), []).append(record)
        pooled.setdefault(record.agent, []).append(record)
    rows = [_row(agent, domain, items, seed) for (agent, domain), items in sorted(grouped.items())]
    rows.extend(_row(agent, "pooled", items, seed) for agent, items in sorted(pooled.items()))
    return rows


def write_results(
    output: Path, records: list[EpisodeRecord], summaries: list[dict[str, object]]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")
    (output / "metrics.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = list(summaries[0]) if summaries else []
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_split_tables(output: Path, summaries: list[dict[str, object]]) -> None:
    ablation_agents = {
        "react",
        "preact",
        "rap_single",
        "single_selective",
        "routed_always",
        "routed_selective",
    }
    ablation = [row for row in summaries if row["agent"] in ablation_agents]
    counts = [row for row in summaries if str(row["agent"]).startswith("routed_k")]
    _write_csv(output / "ablation.csv", ablation)
    _write_csv(output / "expert_count.csv", counts)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
