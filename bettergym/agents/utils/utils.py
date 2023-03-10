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
    mean_angle_vel = (arctan2(goal[1] - x[1], goal[0] - x[0]) - x[2]) / config.dt
    var_angle_vel = config.max_yaw_rate / 4
    angular_velocity = np.random.normal(mean_angle_vel, var_angle_vel)
    linear_velocity = np.random.uniform(
        low=config.min_speed,
        high=config.max_speed
    )
    angular_velocity = np.clip(
        a=angular_velocity,
        a_min=-config.max_yaw_rate,
        a_max=config.max_yaw_rate
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
    if prob <= 1 - eps:
        return other_func(node, planner)
    else:
        return uniform(node, planner)


def binary_policy(node: Any, planner: Planner):
    if len(node.actions) == 1:
        return uniform(node, planner)
    else:
        sorted_actions = [a for _, a in sorted(zip(node.a_values, node.actions), key=lambda pair: pair[0])]
        return np.mean([sorted_actions[0].action, sorted_actions[1].action], axis=0)


def voo(current_state: Any, planner: Planner):
    pass
