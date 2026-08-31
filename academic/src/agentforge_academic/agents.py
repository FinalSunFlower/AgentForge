from __future__ import annotations

import time
from dataclasses import dataclass, field

from .environment import Action, DecisionWorld, State
from .experts import WorldModel
from .foresight import SelectiveForesightGate, rollout
from .llm_engine import METER
from .router import MoERouter


def heuristic_action(world: DecisionWorld, state: State) -> Action:
    """Myopic observation heuristic used as the ReAct / no-foresight controller.

    It ignores hidden hazards and always prefers the larger increment, which
    overshoots exact-match retrieval and arithmetic targets.
    """
    if state.domain == "navigation":
        if state.position < state.target:
            return Action("move", 1)
        if state.position > state.target:
            return Action("move", -1)
        return Action("wait", 0)
    if state.domain == "manipulation":
        return Action("grasp", 0)
    if state.domain == "retrieval":
        return Action("retrieve", 2)
    if state.domain == "household":
        if state.progress == 0:
            return Action("take", 0)
        if state.position != 0:
            return Action("go", 0)
        return Action("put", 0)
    return Action("add", 2)


@dataclass
class AgentStats:
    controller_calls: int = 0
    expert_calls: int = 0
    foresight_calls: int = 0
    prediction_errors: list[float] = field(default_factory=list)
    terminal_errors: list[float] = field(default_factory=list)
    controller_cost: float = 1.0
    expert_cost: float = 0.35
    token_cost: float = 0.0
    tokens: int = 0
    wall_ms: float = 0.0

    @property
    def cost(self) -> float:
        return (
            self.controller_calls * self.controller_cost
            + self.expert_calls * self.expert_cost
            + self.tokens * self.token_cost
        )


class Agent:
    name = "agent"

    def __init__(
        self,
        world: DecisionWorld,
        max_steps: int,
        *,
        controller_cost: float = 1.0,
        expert_cost: float = 0.35,
        token_cost: float = 0.0,
    ) -> None:
        self.world, self.max_steps = world, max_steps
        self.controller_cost, self.expert_cost, self.token_cost = (
            controller_cost,
            expert_cost,
            token_cost,
        )

    def choose(self, state: State, stats: AgentStats) -> Action:
        raise NotImplementedError

    def predict_eval(self, state: State, action: Action) -> tuple[float, float] | None:
        """Optional one-step prediction used only after the action is chosen."""
        return None

    def run(self, state: State) -> tuple[bool, AgentStats, int]:
        stats = AgentStats(
            controller_cost=self.controller_cost,
            expert_cost=self.expert_cost,
            token_cost=self.token_cost,
        )
        started = time.perf_counter()
        METER.take()
        for _ in range(self.max_steps):
            action = self.choose(state, stats)
            forecast = self.predict_eval(state, action)
            transition = self.world.step(state, action)
            if forecast is not None:
                stats.prediction_errors.append(abs(forecast[0] - transition.reward))
                stats.terminal_errors.append(abs(forecast[1] - float(transition.done)))
            state = transition.next_state
            if transition.done:
                stats.tokens, _ = METER.take()
                stats.wall_ms = (time.perf_counter() - started) * 1000
                return transition.success, stats, state.steps
        stats.tokens, _ = METER.take()
        stats.wall_ms = (time.perf_counter() - started) * 1000
        return False, stats, self.max_steps


class ReActAgent(Agent):
    name = "react"

    def choose(self, state: State, stats: AgentStats) -> Action:
        stats.controller_calls += 1
        return heuristic_action(self.world, state)


class RAPAgent(Agent):
    name = "rap_single"

    def __init__(
        self,
        world: DecisionWorld,
        model: WorldModel,
        max_steps: int,
        planning_depth: int = 3,
        **costs: float,
    ) -> None:
        super().__init__(world, max_steps, **costs)
        self.model, self.planning_depth = model, planning_depth

    def choose(self, state: State, stats: AgentStats) -> Action:
        stats.controller_calls += 1
        stats.foresight_calls += 1
        best_action, best_value = None, float("-inf")
        for action in self.world.candidates(state):
            value, _ = rollout(self.model, self.world, state, action, self.planning_depth)
            stats.expert_calls += self.planning_depth
            if value > best_value:
                best_action, best_value = action, value
        assert best_action is not None
        return best_action

    def predict_eval(self, state: State, action: Action) -> tuple[float, float] | None:
        prediction = self.model.predict(state, action)
        return prediction.reward_mean, prediction.terminal_probability


class PreActAgent(RAPAgent):
    """Single-step predicted lookahead. Depth-1 analogue of Fu et al. PreAct."""

    name = "preact"

    def __init__(
        self, world: DecisionWorld, model: WorldModel, max_steps: int, **costs: float
    ) -> None:
        super().__init__(world, model, max_steps, planning_depth=1, **costs)


class RoutedForesightAgent(Agent):
    name = "routed_selective"

    def __init__(
        self,
        world: DecisionWorld,
        router: MoERouter,
        gate: SelectiveForesightGate,
        max_steps: int,
        planning_depth: int = 3,
        always_predict: bool = False,
        **costs: float,
    ) -> None:
        super().__init__(world, max_steps, **costs)
        self.router = router
        self.gate = gate
        self.planning_depth = planning_depth
        self.always_predict = always_predict

    def choose(self, state: State, stats: AgentStats) -> Action:
        stats.controller_calls += 1
        candidates = self.world.candidates(state)
        decisions = [
            (action, self.gate.decide(self.router, state, action)) for action in candidates
        ]
        if not self.always_predict and not any(
            decision.should_predict for _, decision in decisions
        ):
            return heuristic_action(self.world, state)
        stats.foresight_calls += 1
        planned: list[tuple[float, Action]] = []
        for action, decision in decisions:
            if not self.always_predict and not decision.should_predict:
                continue
            routes = self.router.route(state, action)
            value = 0.0
            for route in routes:
                value_i, _ = rollout(route.expert, self.world, state, action, self.planning_depth)
                value += value_i / len(routes)
                stats.expert_calls += self.planning_depth
            planned.append((value, action))
        if not planned:
            return heuristic_action(self.world, state)
        return max(planned, key=lambda item: item[0])[1]

    def predict_eval(self, state: State, action: Action) -> tuple[float, float] | None:
        reward, terminal, _ = self.router.mixture_prediction(state, action)
        return reward, terminal
