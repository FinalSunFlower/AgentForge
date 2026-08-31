from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

Domain = Literal["navigation", "manipulation", "retrieval", "arithmetic", "household"]


@dataclass(frozen=True)
class State:
    domain: Domain
    position: int
    target: int
    progress: int = 0
    steps: int = 0
    hidden: int = 0


@dataclass(frozen=True)
class Action:
    name: str
    argument: int


@dataclass(frozen=True)
class Transition:
    state: State
    action: Action
    next_state: State
    reward: float
    done: bool
    success: bool


class DecisionWorld(Protocol):
    """Shared surface for ToolWorld-v1 and Household-v1."""

    domains: tuple[str, ...]
    max_steps: int

    def reset(self, seed: int, domain: Domain | None = None) -> State: ...
    def actions(self, state: State) -> tuple[Action, ...]: ...
    def candidates(self, state: State) -> list[Action]: ...
    def step(self, state: State, action: Action) -> Transition: ...


class ToolWorld:
    """Four-domain deterministic tool world (ToolWorld-v1).

    Domains match the research brief: navigation/movement, object manipulation,
    query/API retrieval, and numerical calculation. Hidden fields create
    transitions a myopic ReAct heuristic cannot see, so world-model foresight
    is testable. This is a development benchmark, not ALFWorld.
    """

    domains: tuple[Domain, ...] = ("navigation", "manipulation", "retrieval", "arithmetic")

    def __init__(self, *, max_steps: int = 8) -> None:
        self.max_steps = max_steps

    def reset(self, seed: int, domain: Domain | None = None) -> State:
        rng = np.random.default_rng(seed)
        selected = domain or self.domains[int(rng.integers(0, len(self.domains)))]
        if selected == "navigation":
            position, target = int(rng.integers(0, 5)), int(rng.integers(0, 5))
            if position == target:
                target = (target + 2) % 5
            hazard = int(rng.integers(0, 5))
            while hazard in {position, target}:
                hazard = (hazard + 1) % 5
            return State(selected, position, target, hidden=hazard)
        if selected == "manipulation":
            return State(selected, int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        if selected == "retrieval":
            return State(selected, int(rng.integers(0, 3)), int(rng.integers(1, 5)))
        return State(selected, int(rng.integers(0, 4)), int(rng.integers(2, 7)))

    def actions(self, state: State) -> tuple[Action, ...]:
        if state.domain == "navigation":
            return (
                Action("move", -1),
                Action("move", 1),
                Action("jump", 2),
                Action("jump", -2),
                Action("wait", 0),
            )
        if state.domain == "manipulation":
            return (Action("grasp", 0), Action("grasp", 1), Action("grasp", 2))
        if state.domain == "retrieval":
            return (Action("retrieve", 0), Action("retrieve", 1), Action("retrieve", 2))
        return (Action("add", 1), Action("add", 2), Action("reset", 0))

    def step(self, state: State, action: Action) -> Transition:
        if action not in self.actions(state):
            raise ValueError("action_not_in_candidate_set")
        if state.domain == "navigation":
            transition = self._step_navigation(state, action)
        elif state.domain == "manipulation":
            transition = self._step_manipulation(state, action)
        elif state.domain == "retrieval":
            transition = self._step_retrieval(state, action)
        else:
            transition = self._step_arithmetic(state, action)
        done = transition.done or transition.next_state.steps >= self.max_steps
        return Transition(
            transition.state,
            transition.action,
            transition.next_state,
            transition.reward,
            done,
            transition.success,
        )

    def candidates(self, state: State) -> list[Action]:
        return list(self.actions(state))

    def _step_navigation(self, state: State, action: Action) -> Transition:
        position = state.position
        reward = -0.04
        if action.name == "move":
            proposed = min(4, max(0, state.position + action.argument))
            if proposed == state.hidden:
                reward = -0.35
            else:
                position = proposed
        elif action.name == "jump":
            proposed = state.position + action.argument
            if proposed < 0 or proposed > 4 or proposed == state.hidden:
                reward = -0.25
            else:
                position = proposed
                reward = -0.08
        success = position == state.target
        if success:
            reward = 1.0
        next_state = State(
            state.domain, position, state.target, state.progress, state.steps + 1, state.hidden
        )
        return Transition(state, action, next_state, reward, success, success)

    def _step_manipulation(self, state: State, action: Action) -> Transition:
        success = action.name == "grasp" and action.argument == state.target
        reward = 1.0 if success else -0.18
        next_state = State(
            state.domain, state.position, state.target, int(success), state.steps + 1, state.hidden
        )
        return Transition(state, action, next_state, reward, success, success)

    def _step_retrieval(self, state: State, action: Action) -> Transition:
        if action.argument == 0:
            progress, reward, success = state.progress, -0.2, False
        else:
            progress = state.progress + action.argument
            if progress == state.target:
                reward, success = 1.0, True
            elif progress > state.target:
                reward, success = -0.4, False
                progress = state.progress
            else:
                reward, success = -0.02, False
        next_state = State(
            state.domain, state.position, state.target, progress, state.steps + 1, state.hidden
        )
        done = success or (action.argument != 0 and state.progress + action.argument > state.target)
        return Transition(state, action, next_state, reward, done, success)

    def _step_arithmetic(self, state: State, action: Action) -> Transition:
        if action.name == "reset":
            progress, reward, success, fail = 0, -0.05, False, False
        else:
            progress = state.progress + action.argument
            if progress == state.target:
                reward, success, fail = 1.0, True, False
            elif progress > state.target:
                reward, success, fail = -0.4, False, True
                progress = state.progress
            else:
                reward, success, fail = -0.03, False, False
        next_state = State(
            state.domain, state.position, state.target, progress, state.steps + 1, state.hidden
        )
        return Transition(state, action, next_state, reward, success or fail, success)
