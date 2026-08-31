from dataclasses import replace

from agentforge_academic.agents import heuristic_action
from agentforge_academic.environment import Action
from agentforge_academic.experts import collect_random_trajectories
from agentforge_academic.household import HouseholdWorld


def test_household_reset_is_deterministic() -> None:
    world = HouseholdWorld(max_steps=6)
    assert world.reset(11) == world.reset(11)


def test_household_take_succeeds_when_colocated() -> None:
    world = HouseholdWorld(max_steps=6)
    state = replace(world.reset(0), position=1, hidden=1, target=0, progress=0)
    transition = world.step(state, Action("take", 0))
    assert transition.success is True
    assert transition.next_state.progress == 1


def test_household_put_requires_kitchen_and_holding() -> None:
    world = HouseholdWorld(max_steps=6)
    state = replace(world.reset(0), position=0, hidden=2, target=1, progress=1)
    transition = world.step(state, Action("put", 0))
    assert transition.success is True


def test_household_heuristic_does_not_search_rooms() -> None:
    world = HouseholdWorld()
    empty = replace(world.reset(3), position=0, hidden=2, target=0, progress=0)
    assert heuristic_action(world, empty) == Action("take", 0)
    holding = replace(empty, progress=1, position=2)
    assert heuristic_action(world, holding) == Action("go", 0)


def test_household_random_trajectories_stay_in_domain() -> None:
    world = HouseholdWorld(max_steps=4)
    records = collect_random_trajectories(world, 4, 1)
    assert records
    assert all(item.state.domain == "household" for item in records)
