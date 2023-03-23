import random
from typing import Any, Callable

import numpy as np
from numba import njit
from numpy import arctan2

from bettergym.agents.planner import Planner


def uniform(node: Any, planner: Planner):
    current_state = node.state
    available_actions = planner.environment.get_actions(current_state)
    return available_actions.sample()


def uniform_discrete(node: Any, planner: Planner):
    current_state = node.state
    actions = planner.environment.get_actions(current_state)
    return random.choice(actions)


@njit
def compute_towards_goal_jit(x: np.ndarray, goal: np.ndarray, max_yaw_rate: float, dt: float, min_speed: float,
                             max_speed: float):
    mean_angle_vel = (np.arctan2(goal[1] - x[1], goal[0] - x[0]) - x[2]) / dt
    var_angle_vel = max_yaw_rate / 10
    angular_velocity = np.random.normal(mean_angle_vel, var_angle_vel)
    linear_velocity = np.random.uniform(
        a=min_speed,
        b=max_speed
    )
    if angular_velocity > max_yaw_rate:
        angular_velocity = max_yaw_rate
    elif angular_velocity < -max_yaw_rate:
        angular_velocity = -max_yaw_rate

    return np.array([linear_velocity, angular_velocity])


def towards_goal(node: Any, planner: Planner):
    config = planner.environment.config
    return compute_towards_goal_jit(node.state.x, node.state.goal, config.max_yaw_rate, config.dt, config.min_speed,
                                    config.max_speed)

# def towards_goal(node: Any, planner: Planner):
#     current_state = node.state
#     goal = current_state.goal
#     x = current_state.x
#     config = planner.environment.config
#     mean_angle_vel = (arctan2(goal[1] - x[1], goal[0] - x[0]) - x[2]) / config.dt
#     var_angle_vel = config.max_yaw_rate / 4
#     # angular_velocity = random.gauss(mean_angle_vel, var_angle_vel)
#     # linear_velocity = random.uniform(
#     #     config.min_speed,
#     #     config.max_speed
#     # )
#     angular_velocity = np.random.normal(mean_angle_vel, var_angle_vel)
#     linear_velocity = np.random.uniform(
#         low=config.min_speed,
#         high=config.max_speed
#     )
#     if angular_velocity > config.max_yaw_rate:
#         angular_velocity = config.max_yaw_rate
#     elif angular_velocity < -config.max_yaw_rate:
#         angular_velocity = -config.max_yaw_rate
#
#     return np.array([linear_velocity, angular_velocity])
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
