from agentforge_academic.environment import Action, ToolWorld
from agentforge_academic.experts import TabularWorldModel, collect_random_trajectories


def test_tool_world_is_deterministic_for_seed() -> None:
    world = ToolWorld(max_steps=6)
    assert world.reset(11, "navigation") == world.reset(11, "navigation")


def test_four_domains_are_declared() -> None:
    assert ToolWorld.domains == ("navigation", "manipulation", "retrieval", "arithmetic")


def test_expert_only_learns_declared_domain() -> None:
    world = ToolWorld()
    records = collect_random_trajectories(world, 20, 3)
    expert = TabularWorldModel("navigation", "navigation")
    expert.fit(records)
    assert expert.predict(world.reset(9, "arithmetic"), Action("add", 1)).uncertainty == 1.0


def test_invalid_domain_action_is_rejected() -> None:
    import pytest

    world = ToolWorld()
    state = world.reset(1, "navigation")
    with pytest.raises(ValueError, match="action_not_in_candidate_set"):
        world.step(state, Action("unknown", 0))


def test_navigation_hazard_blocks_direct_move() -> None:
    world = ToolWorld()
    state = world.reset(11, "navigation")
    direction = 1 if state.target > state.position else -1
    proposed = min(4, max(0, state.position + direction))
    if proposed != state.hidden:
        state = state.__class__(
            state.domain, state.position, state.target, state.progress, state.steps, proposed
        )
    transition = world.step(state, Action("move", direction))
    assert transition.next_state.position == state.position
    assert transition.reward < 0


def test_arithmetic_overshoot_is_terminal_failure() -> None:
    world = ToolWorld()
    state = world.reset(4, "arithmetic")
    state = state.__class__(state.domain, state.position, 3, 2, 0, 0)
    transition = world.step(state, Action("add", 2))
    assert transition.done is True
    assert transition.success is False


def test_jump_can_skip_a_hazard() -> None:
    from dataclasses import replace

    world = ToolWorld()
    state = replace(world.reset(2, "navigation"), position=0, target=3, hidden=1)
    transition = world.step(state, Action("jump", 2))
    assert transition.next_state.position == 2
    assert transition.success is False


def test_wrong_grasp_does_not_succeed() -> None:
    world = ToolWorld()
    state = world.reset(5, "manipulation")
    wrong = 0 if state.target != 0 else 1
    transition = world.step(state, Action("grasp", wrong))
    assert transition.success is False
