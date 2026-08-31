from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .environment import Action, DecisionWorld, Domain, State, Transition


@dataclass(frozen=True)
class Prediction:
    next_state: State
    reward_mean: float
    terminal_probability: float
    uncertainty: float


class WorldModel(Protocol):
    name: str
    domain: str

    def predict(self, state: State, action: Action) -> Prediction: ...


class TabularWorldModel:
    """Finite-count probabilistic dynamics model with Laplace smoothing.

    Each specialist trains only on its declared domain. The generic ``*``
    expert sees every domain. Unseen keys return maximum uncertainty rather
    than a fabricated deterministic next state.
    """

    def __init__(self, name: str, domain: str, *, alpha: float = 1.0) -> None:
        self.name, self.domain, self.alpha = name, domain, alpha
        self._next_counts: dict[tuple[State, Action], dict[State, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._reward_sum: dict[tuple[State, Action], float] = defaultdict(float)
        self._done_sum: dict[tuple[State, Action], float] = defaultdict(float)
        self._counts: dict[tuple[State, Action], float] = defaultdict(float)

    def fit(self, transitions: list[Transition]) -> None:
        for item in transitions:
            if self.domain != "*" and item.state.domain != self.domain:
                continue
            key = (item.state, item.action)
            self._next_counts[key][item.next_state] += 1.0
            self._reward_sum[key] += item.reward
            self._done_sum[key] += float(item.done)
            self._counts[key] += 1.0

    def predict(self, state: State, action: Action) -> Prediction:
        key = (state, action)
        count = self._counts[key]
        if not count:
            return Prediction(state, 0.0, 0.5, 1.0)
        candidates = self._next_counts[key]
        total = count + self.alpha * max(1, len(candidates))
        next_state = max(candidates.items(), key=lambda item: item[1])[0]
        posterior = (candidates[next_state] + self.alpha) / total
        reward = self._reward_sum[key] / count
        terminal = self._done_sum[key] / count
        return Prediction(next_state, reward, terminal, float(1.0 - posterior))


class OracleWorldModel(TabularWorldModel):
    """Reference model trained on the exact transition function for sanity checks."""

    def __init__(self, world: DecisionWorld, domain: str) -> None:
        super().__init__(f"oracle-{domain}", domain)
        self.world = world

    def predict(self, state: State, action: Action) -> Prediction:
        transition = self.world.step(state, action)
        return Prediction(transition.next_state, transition.reward, float(transition.done), 0.0)


def collect_random_trajectories(world: DecisionWorld, episodes: int, seed: int) -> list[Transition]:
    rng = np.random.default_rng(seed)
    records: list[Transition] = []
    for episode in range(episodes):
        domain: Domain = world.domains[episode % len(world.domains)]
        state = world.reset(seed + episode, domain)
        while state.steps < world.max_steps:
            actions = world.candidates(state)
            transition = world.step(state, actions[int(rng.integers(0, len(actions)))])
            records.append(transition)
            state = transition.next_state
            if transition.done:
                break
    # One-step cover so tabular experts see each reset state's action set.
    for offset, domain in enumerate(world.domains):
        for episode in range(max(1, episodes // len(world.domains))):
            state = world.reset(seed + 50_000 + offset * 10_000 + episode, domain)
            for action in world.candidates(state):
                records.append(world.step(state, action))
    return records
