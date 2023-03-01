import random
from typing import Any, Callable

import numpy as np
from numpy import arctan2

from bettergym.agents.planner import Planner


def uniform(current_state: Any, planner: Planner):
    available_actions = planner.environment.get_actions(current_state)
    return available_actions.sample()


def towards_goal(current_state, planner: Planner):
    goal = current_state.goal
    x = current_state.x
    config = planner.environment.config
    y = goal[1] - x[1]
    x = goal[0] - x[0]
    mean_angle = arctan2(y, x)
    std_angle = config.max_yaw_rate/10
    angular_velocity = np.random.normal(mean_angle, std_angle)
    linear_velocity = np.random.uniform(
        low=config.min_speed,
        high=config.max_speed
    )
    return np.array([linear_velocity, angular_velocity])


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
