from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .environment import Action, State
from .experts import WorldModel


@dataclass(frozen=True)
class Route:
    expert: WorldModel
    score: float


class MoERouter:
    """Action-level sparse router with deterministic top-k selection.

    The router uses domain compatibility as a prior and expert uncertainty as a
    penalty. This is intentionally inspectable; a learned router can replace
    ``score`` without changing the agent/evaluation contracts.
    """

    def __init__(self, experts: Sequence[WorldModel], top_k: int = 2) -> None:
        if not experts:
            raise ValueError("at_least_one_expert_required")
        if top_k < 1:
            raise ValueError("top_k_must_be_positive")
        self.experts, self.top_k = experts, min(top_k, len(experts))

    def score(self, expert: WorldModel, state: State, action: Action) -> float:
        prediction = expert.predict(state, action)
        compatibility = 2.0 if expert.domain == state.domain else -2.0
        return compatibility - prediction.uncertainty

    def route(self, state: State, action: Action) -> list[Route]:
        routes = [Route(expert, self.score(expert, state, action)) for expert in self.experts]
        return sorted(routes, key=lambda route: (-route.score, route.expert.name))[: self.top_k]

    def mixture_prediction(self, state: State, action: Action) -> tuple[float, float, list[str]]:
        routes = self.route(state, action)
        weights = [max(0.01, route.score + 3.0) for route in routes]
        total = sum(weights)
        predictions = [route.expert.predict(state, action) for route in routes]
        reward = (
            sum(
                weight * prediction.reward_mean
                for weight, prediction in zip(weights, predictions, strict=True)
            )
            / total
        )
        terminal = (
            sum(
                weight * prediction.terminal_probability
                for weight, prediction in zip(weights, predictions, strict=True)
            )
            / total
        )
        return reward, terminal, [route.expert.name for route in routes]
