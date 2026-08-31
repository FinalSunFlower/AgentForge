from pathlib import Path

from agentforge_academic.environment import Action, State
from agentforge_academic.experiment import run
from agentforge_academic.llm_engine import FakeEngine, get_engine, parse_prediction_json
from agentforge_academic.prompted import PromptedWorldModel


class CountingEngine(FakeEngine):
    calls = 0

    def complete(self, system: str, user: str):
        type(self).calls += 1
        return super().complete(system, user)


def test_parse_prediction_json_reads_embedded_object() -> None:
    payload = parse_prediction_json('noise {"reward": 0.5, "done": 0.1, "confidence": 0.8} tail')
    assert payload is not None
    assert payload["reward"] == 0.5


def test_parse_prediction_json_reads_fenced_block() -> None:
    payload = parse_prediction_json('```json\n{"reward": 1.0, "done": 1, "confidence": 0.9}\n```')
    assert payload is not None
    assert payload["reward"] == 1.0


def test_parse_prediction_json_rejects_invalid() -> None:
    assert parse_prediction_json("no json here") is None


def test_get_engine_fake_never_loads_weights() -> None:
    engine = get_engine({"engine": "fake"})
    assert isinstance(engine, FakeEngine)


def test_prompted_world_model_caches_identical_queries() -> None:
    CountingEngine.calls = 0
    model = PromptedWorldModel("expert-household", "household", CountingEngine())
    state = State("household", 0, 0, 0, 0, 1)
    action = Action("take", 0)
    first = model.predict(state, action)
    second = model.predict(state, action)
    assert first == second
    assert CountingEngine.calls == 1


def test_prompted_fit_is_a_noop() -> None:
    model = PromptedWorldModel("expert-generic", "*", FakeEngine())
    model.fit([])
    prediction = model.predict(State("arithmetic", 0, 2), Action("add", 1))
    assert 0.0 <= prediction.terminal_probability <= 1.0


def test_experiment_prompted_fake_household(tmp_path: Path) -> None:
    config = {
        "environment": "Household-v1",
        "backend": "prompted",
        "engine": "fake",
        "train_episodes": 0,
        "calibration_episodes": 3,
        "test_episodes": 4,
        "max_steps": 4,
        "seeds": [7],
        "top_k": 1,
        "gate_quantile": 0.7,
        "controller_cost": 1.0,
        "expert_cost": 0.35,
        "planning_depth": 1,
        "variants": ["react", "preact"],
        "skip_expert_count": True,
    }
    run(config, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "cost_quality.pdf").exists()
    assert (tmp_path / "ablation.pdf").exists()
