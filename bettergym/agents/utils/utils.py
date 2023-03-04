import random
from typing import Any, Callable

import numpy as np
from numpy import arctan2

from bettergym.agents.planner import Planner


def uniform(node: Any, planner: Planner):
    current_state = node.state
    available_actions = planner.environment.get_actions(current_state)
    return available_actions.sample()


def uniform_discrete(node: Any, planner: Planner):
    current_state = node.state
    actions = planner.environment.get_actions(current_state)
    return actions[np.random.choice(len(actions))]


def towards_goal(node: Any, planner: Planner):
    current_state = node.state
    goal = current_state.goal
    x = current_state.x
    config = planner.environment.config
    mean_angle = arctan2(goal[1] - x[1], goal[0] - x[0])
    var_angle = config.max_yaw_rate / 10
    angular_velocity = np.random.normal(mean_angle, var_angle)
    linear_velocity = np.random.uniform(
        low=config.min_speed,
        high=config.max_speed
    )
    return np.array([linear_velocity, angular_velocity])


def epsilon_greedy(eps: float, other_func: Callable, node: Any, planner: Planner):
    """
    :param node:
    :param eps: defines the probability of acting according to other_func
    :param other_func:
    :param planner:
    :return:
    """
    prob = random.random()
    if prob <= 1-eps:
        other_func(node, planner)
    else:
        uniform(node, planner)


def binary_policy(node: Any, planner: Planner):
    if len(node.actions) == 1:
        return uniform(node, planner)
    else:
        sorted_actions = [a for _, a in sorted(zip(node.a_values, node.actions))]
        return np.mean([sorted_actions[0], sorted_actions[1]], axis=0)




def voo(current_state: Any, planner: Planner):
    pass
