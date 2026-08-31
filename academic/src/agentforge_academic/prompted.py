from __future__ import annotations

from .environment import Action, State
from .experts import Prediction
from .llm_engine import LanguageEngine, parse_prediction_json

DOMAIN_PROMPTS = {
    "navigation": (
        "You are a frozen navigation world model. Predict move/jump/wait outcomes. "
        "Hazards block move and can be skipped by jump. Reply with JSON only."
    ),
    "manipulation": (
        "You are a frozen object-manipulation world model. Predict grasp outcomes. "
        "Only grasping the target object succeeds. Reply with JSON only."
    ),
    "retrieval": (
        "You are a frozen retrieval world model. Predict retrieve(0/1/2) outcomes. "
        "Success requires exact progress == target. Overshoot fails. Reply with JSON only."
    ),
    "arithmetic": (
        "You are a frozen arithmetic world model. Predict add/reset outcomes. "
        "Success requires exact progress == target. Overshoot fails. Reply with JSON only."
    ),
    "household": (
        "You are a frozen household world model. Rooms: 0 kitchen, 1 living, 2 bedroom. "
        "hidden is the object room. progress 1 means holding. target 0=take object, 1=put in kitchen. "
        "take succeeds only if position==hidden and progress==0. "
        "put succeeds only if progress==1 and position==0. "
        "Reply with one JSON object only."
    ),
    "*": (
        "You are a frozen generic world model for tool-using agents. "
        "Predict the next reward and whether the episode ends. Reply with JSON only."
    ),
}


def _user_prompt(state: State, action: Action) -> str:
    return (
        f"domain={state.domain} position={state.position} target={state.target} "
        f"progress={state.progress} steps={state.steps} hidden={state.hidden} "
        f"action={action.name}({action.argument})\n"
        "Return only JSON like "
        '{"reward": 1.0, "done": 1, "confidence": 0.8, "next": {"position": 0, "progress": 1}}'
    )


class PromptedWorldModel:
    """Prompt-specialized expert on one shared frozen LM. Weights are never updated."""

    def __init__(self, name: str, domain: str, engine: LanguageEngine) -> None:
        self.name, self.domain, self.engine = name, domain, engine
        self._cache: dict[tuple[State, Action], Prediction] = {}

    def fit(self, transitions: list) -> None:
        del transitions

    def predict(self, state: State, action: Action) -> Prediction:
        key = (state, action)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        system = DOMAIN_PROMPTS.get(self.domain, DOMAIN_PROMPTS["*"])
        result = self.engine.complete(system, _user_prompt(state, action))
        payload = parse_prediction_json(result.text)
        if payload is None:
            prediction = Prediction(state, 0.0, 0.5, 1.0)
        else:
            nxt = payload.get("next") if isinstance(payload.get("next"), dict) else {}
            next_state = State(
                state.domain,
                int(nxt.get("position", state.position)),
                state.target,
                int(nxt.get("progress", state.progress)),
                state.steps + 1,
                state.hidden,
            )
            reward = float(payload.get("reward", 0.0))
            done = min(1.0, max(0.0, float(payload.get("done", 0.5))))
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.5))))
            prediction = Prediction(next_state, reward, done, 1.0 - confidence)
        self._cache[key] = prediction
        return prediction
