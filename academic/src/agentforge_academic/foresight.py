from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import Action, State
from .experts import WorldModel
from .router import MoERouter


@dataclass(frozen=True)
class GateDecision:
    should_predict: bool
    confidence: float
    uncertainty: float
    threshold: float
    reason: str


class SelectiveForesightGate:
    """Calibrated selective prediction gate.

    Calibration stores a held-out quantile of ensemble disagreement. At test time
    foresight is requested only when disagreement exceeds that threshold, making
    the cost/coverage trade-off explicit and reproducible.
    """

    def __init__(self, quantile: float = 0.8) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile_must_be_between_zero_and_one")
        self.quantile = quantile
        self.threshold: float | None = None

    def disagreement(self, router: MoERouter, state: State, action: Action) -> float:
        predictions = [route.expert.predict(state, action) for route in router.route(state, action)]
        if len(predictions) < 2:
            return predictions[0].uncertainty if predictions else 1.0
        rewards = np.asarray([prediction.reward_mean for prediction in predictions], dtype=float)
        terminals = np.asarray(
            [prediction.terminal_probability for prediction in predictions], dtype=float
        )
        return float(np.sqrt(np.var(rewards) + np.var(terminals)))

    def calibrate(self, router: MoERouter, samples: list[tuple[State, Action]]) -> float:
        values = [self.disagreement(router, state, action) for state, action in samples]
        if not values:
            raise ValueError("calibration_requires_samples")
        self.threshold = float(np.quantile(np.asarray(values), self.quantile, method="higher"))
        return self.threshold

    def decide(self, router: MoERouter, state: State, action: Action) -> GateDecision:
        uncertainty = self.disagreement(router, state, action)
        threshold = self.threshold if self.threshold is not None else 0.0
        should_predict = uncertainty >= threshold
        return GateDecision(
            should_predict,
            1.0 / (1.0 + uncertainty),
            uncertainty,
            threshold,
            "calibrated_disagreement",
        )


def rollout(
    model: WorldModel, world, state: State, action: Action, depth: int
) -> tuple[float, float]:
    """Evaluate a candidate using model-predicted transitions and greedy continuation."""
    total, discount, current, current_action = 0.0, 1.0, state, action
    terminal_probability = 0.0
    for _ in range(depth):
        prediction = model.predict(current, current_action)
        total += discount * prediction.reward_mean
        terminal_probability = max(terminal_probability, prediction.terminal_probability)
        current = prediction.next_state
        if prediction.terminal_probability > 0.5:
            break
        actions = world.candidates(current)
        current_action = max(
            actions, key=lambda candidate: model.predict(current, candidate).reward_mean
        )
        discount *= 0.9
    return total, terminal_probability
