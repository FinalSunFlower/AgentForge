from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .agents import PreActAgent, RAPAgent, ReActAgent, RoutedForesightAgent
from .environment import DecisionWorld, ToolWorld
from .evaluation import evaluate, summarize, write_results, write_split_tables
from .experts import TabularWorldModel, WorldModel, collect_random_trajectories
from .foresight import SelectiveForesightGate
from .household import HouseholdWorld
from .llm_engine import get_engine
from .plotting import write_figures
from .prompted import PromptedWorldModel
from .router import MoERouter


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("config_must_be_mapping")
    return config


def _costs(config: dict[str, Any]) -> dict[str, float]:
    return {
        "controller_cost": float(config.get("controller_cost", 1.0)),
        "expert_cost": float(config.get("expert_cost", 0.35)),
        "token_cost": float(config.get("token_cost", 0.0)),
    }


def make_world(config: dict[str, Any]) -> DecisionWorld:
    name = str(config.get("environment", "ToolWorld-v1"))
    max_steps = int(config["max_steps"])
    if name == "Household-v1":
        return HouseholdWorld(max_steps=max_steps)
    if name != "ToolWorld-v1":
        raise ValueError(f"unknown_environment:{name}")
    return ToolWorld(max_steps=max_steps)


def build_components(
    config: dict[str, Any], seed: int
) -> tuple[DecisionWorld, list[WorldModel], WorldModel, MoERouter, SelectiveForesightGate]:
    world = make_world(config)
    backend = str(config.get("backend", "tabular"))
    if backend == "prompted":
        engine = get_engine(config)
        experts: list[WorldModel] = [
            PromptedWorldModel(f"expert-{domain}", domain, engine) for domain in world.domains
        ]
        generic: WorldModel = PromptedWorldModel("expert-generic", "*", engine)
    elif backend == "tabular":
        trajectories = collect_random_trajectories(world, int(config["train_episodes"]), seed)
        experts = [TabularWorldModel(f"expert-{domain}", domain) for domain in world.domains]
        for expert in experts:
            expert.fit(trajectories)
        generic = TabularWorldModel("expert-generic", "*")
        generic.fit(trajectories)
    else:
        raise ValueError(f"unknown_backend:{backend}")
    router = MoERouter(experts, top_k=int(config["top_k"]))
    gate = SelectiveForesightGate(float(config["gate_quantile"]))
    calibration = collect_random_trajectories(
        world, int(config.get("calibration_episodes", 8)), seed + 100_000
    )
    gate.calibrate(router, [(item.state, item.action) for item in calibration])
    return world, experts, generic, router, gate


def run(config: dict[str, Any], output: Path) -> None:
    costs = _costs(config)
    depth = int(config.get("planning_depth", 3))
    max_steps = int(config["max_steps"])
    test_episodes = int(config["test_episodes"])
    variants = set(config.get("variants") or [])
    skip_expert_count = bool(config.get("skip_expert_count", False))
    all_records = []
    for seed in config["seeds"]:
        world, experts, generic, router, gate = build_components(config, int(seed))

        def make_react(current: DecisionWorld) -> ReActAgent:
            return ReActAgent(current, max_steps, **costs)

        def make_preact(current: DecisionWorld, model: WorldModel = generic) -> PreActAgent:
            return PreActAgent(current, model, max_steps, **costs)

        def make_rap(current: DecisionWorld, model: WorldModel = generic) -> RAPAgent:
            return RAPAgent(current, model, max_steps, planning_depth=depth, **costs)

        def make_selective(
            current: DecisionWorld,
            route: MoERouter = router,
            selective_gate: SelectiveForesightGate = gate,
        ) -> RoutedForesightAgent:
            return RoutedForesightAgent(
                current, route, selective_gate, max_steps, planning_depth=depth, **costs
            )

        def make_always(
            current: DecisionWorld,
            route: MoERouter = router,
            selective_gate: SelectiveForesightGate = gate,
        ) -> RoutedForesightAgent:
            return RoutedForesightAgent(
                current,
                route,
                selective_gate,
                max_steps,
                planning_depth=depth,
                always_predict=True,
                **costs,
            )

        single_router = MoERouter([generic], top_k=1)
        single_gate = SelectiveForesightGate(float(config["gate_quantile"]))
        single_gate.calibrate(
            single_router,
            [
                (item.state, item.action)
                for item in collect_random_trajectories(
                    world, int(config.get("calibration_episodes", 8)), int(seed) + 200_000
                )
            ],
        )

        def make_single(
            current: DecisionWorld,
            route: MoERouter = single_router,
            selective_gate: SelectiveForesightGate = single_gate,
        ) -> RoutedForesightAgent:
            return RoutedForesightAgent(
                current, route, selective_gate, max_steps, planning_depth=depth, **costs
            )

        jobs = [
            ("react", make_react),
            ("preact", make_preact),
            ("rap_single", make_rap),
            ("routed_selective", make_selective),
            ("routed_always", make_always),
            ("single_selective", make_single),
        ]
        for name, factory in jobs:
            if variants and name not in variants:
                continue
            all_records.extend(evaluate(factory, world, name, test_episodes, int(seed)))

        if skip_expert_count:
            continue
        for top_k in range(1, len(experts) + 1):
            if variants and f"routed_k{top_k}" not in variants:
                continue
            count_router = MoERouter(experts, top_k=top_k)
            count_gate = SelectiveForesightGate(float(config["gate_quantile"]))
            count_gate.calibrate(
                count_router,
                [
                    (item.state, item.action)
                    for item in collect_random_trajectories(
                        world,
                        int(config.get("calibration_episodes", 8)),
                        int(seed) + 300_000 + top_k,
                    )
                ],
            )

            def make_k(
                current: DecisionWorld,
                route: MoERouter = count_router,
                selective_gate: SelectiveForesightGate = count_gate,
            ) -> RoutedForesightAgent:
                return RoutedForesightAgent(
                    current,
                    route,
                    selective_gate,
                    max_steps,
                    planning_depth=depth,
                    always_predict=True,
                    **costs,
                )

            all_records.extend(
                evaluate(make_k, world, f"routed_k{top_k}", test_episodes, int(seed))
            )

    summaries = summarize(all_records, seed=23)
    write_results(output, all_records, summaries)
    write_split_tables(output, summaries)
    write_figures(output)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routed world-model experiments.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
