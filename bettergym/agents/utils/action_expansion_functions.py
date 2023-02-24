import random
from typing import Any, Callable

from gymnasium import Space

from bettergym.agents.planner import Planner


def uniform(current_state: Any, planner: Planner):
    available_actions: Space = planner.environment.get_actions(current_state)
    return available_actions.sample()


def epsilon_greedy(eps: float, other_func: Callable, current_state: Any, planner: Planner):
    """
    :param eps: defines the probability of acting according to other_func
    :param other_func:
    :param current_state:
    :param planner:
    :return:
    """
    prob = random.random()
    if prob <= eps:
        other_func(current_state, planner)
    else:
        uniform(current_state, planner)


def voo(current_state: Any, planner: Planner):
    pass
