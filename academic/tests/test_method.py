from pathlib import Path

import pytest

from agentforge_academic.agents import ReActAgent, heuristic_action
from agentforge_academic.environment import Action, ToolWorld
from agentforge_academic.experiment import run
from agentforge_academic.experts import TabularWorldModel, collect_random_trajectories
from agentforge_academic.foresight import SelectiveForesightGate
from agentforge_academic.router import MoERouter
from agentforge_academic.trajectory import alfworld_record_to_transition


@pytest.fixture
def components():
    world = ToolWorld(max_steps=8)
    records = collect_random_trajectories(world, 80, 17)
    experts = [TabularWorldModel(f"expert-{domain}", domain) for domain in world.domains]
    for expert in experts:
        expert.fit(records)
    router = MoERouter(experts, top_k=2)
    gate = SelectiveForesightGate(0.8)
    gate.calibrate(router, [(item.state, item.action) for item in records[:100]])
    return world, router, gate


def test_router_selects_domain_expert(components) -> None:
    world, router, _ = components
    state = world.reset(2, "retrieval")
    routes = router.route(state, Action("retrieve", 1))
    assert routes[0].expert.domain == "retrieval"
    assert len(routes) == 2


def test_router_selects_manipulation_expert(components) -> None:
    world, router, _ = components
    state = world.reset(8, "manipulation")
    routes = router.route(state, Action("grasp", state.target))
    assert routes[0].expert.domain == "manipulation"


def test_gate_calibration_is_deterministic(components) -> None:
    world, router, gate = components
    state = world.reset(4, "navigation")
    decision = gate.decide(router, state, Action("move", 1))
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.threshold >= 0.0


def test_gate_rejects_empty_calibration() -> None:
    gate = SelectiveForesightGate()
    router = MoERouter([TabularWorldModel("x", "navigation")])
    with pytest.raises(ValueError, match="calibration_requires_samples"):
        gate.calibrate(router, [])


def test_react_heuristic_ignores_hazard() -> None:
    world = ToolWorld()
    state = world.reset(11, "navigation")
    action = heuristic_action(world, state)
    if state.position < state.target:
        assert action == Action("move", 1)
    elif state.position > state.target:
        assert action == Action("move", -1)


def test_react_run_records_no_prediction_error() -> None:
    world = ToolWorld(max_steps=4)
    agent = ReActAgent(world, 4)
    _, stats, _ = agent.run(world.reset(3, "arithmetic"))
    assert stats.prediction_errors == []
    assert stats.foresight_calls == 0


def test_trajectory_adapter_accepts_hidden_default() -> None:
    record = {
        "domain": "navigation",
        "position": 0,
        "target": 2,
        "progress": 0,
        "steps": 0,
        "action": {"name": "move", "argument": 1},
        "next_state": {
            "domain": "navigation",
            "position": 1,
            "target": 2,
            "progress": 0,
            "steps": 1,
        },
        "reward": -0.04,
        "done": False,
        "success": False,
    }
    transition = alfworld_record_to_transition(record)
    assert transition.state.hidden == 0
    assert transition.next_state.position == 1


def test_experiment_smoke(tmp_path: Path) -> None:
    config = {
        "environment": "ToolWorld-v1",
        "train_episodes": 24,
        "calibration_episodes": 12,
        "test_episodes": 8,
        "max_steps": 5,
        "seeds": [7],
        "top_k": 2,
        "gate_quantile": 0.7,
        "controller_cost": 1.0,
        "expert_cost": 0.35,
        "planning_depth": 2,
    }
    run(config, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "ablation.csv").exists()
    assert (tmp_path / "expert_count.csv").exists()
    assert (tmp_path / "cost_quality.pdf").exists()
    assert (tmp_path / "ablation.pdf").exists()
    assert (tmp_path / "expert_count.pdf").exists()
    assert (tmp_path / "config.json").exists()
