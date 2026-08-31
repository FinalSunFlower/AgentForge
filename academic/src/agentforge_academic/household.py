from __future__ import annotations

import numpy as np

from .environment import Action, State, Transition


class HouseholdWorld:
    """ALFWorld-style text household (Household-v1).

    Three rooms, one object, pick-and-place. The object room is hidden from the
    myopic heuristic. This is a second environment for generalization checks.
    It is not the official ALFWorld release.
    """

    domains = ("household",)

    def __init__(self, *, max_steps: int = 10) -> None:
        self.max_steps = max_steps

    def reset(self, seed: int, domain: str | None = None) -> State:
        del domain
        rng = np.random.default_rng(seed)
        room = int(rng.integers(0, 3))
        object_room = int(rng.integers(0, 3))
        goal = int(rng.integers(0, 2))
        return State("household", room, goal, 0, 0, object_room)

    def actions(self, state: State) -> tuple[Action, ...]:
        del state
        return (
            Action("go", 0),
            Action("go", 1),
            Action("go", 2),
            Action("take", 0),
            Action("put", 0),
        )

    def candidates(self, state: State) -> list[Action]:
        return list(self.actions(state))

    def step(self, state: State, action: Action) -> Transition:
        if action not in self.actions(state):
            raise ValueError("action_not_in_candidate_set")
        room, holding, object_room = state.position, state.progress, state.hidden
        reward, success = -0.05, False
        if action.name == "go":
            room = action.argument
        elif action.name == "take":
            if holding == 0 and room == object_room:
                holding = 1
                reward = 1.0 if state.target == 0 else 0.25
                success = state.target == 0
            else:
                reward = -0.2
        elif action.name == "put":
            if holding == 1 and room == 0:
                holding = 0
                object_room = 0
                reward = 1.0 if state.target == 1 else 0.15
                success = state.target == 1
            else:
                reward = -0.2
        next_state = State("household", room, state.target, holding, state.steps + 1, object_room)
        done = success or next_state.steps >= self.max_steps
        return Transition(state, action, next_state, reward, done, success)
