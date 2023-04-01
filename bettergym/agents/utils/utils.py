import random
from typing import Any, Callable

import numpy as np
from numba import njit

from bettergym.agents.planner import Planner
from mcts_utils import velocity_obstacle_nearest


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
    var_angle_vel = max_yaw_rate ** 2
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


def voronoi(actions: np.ndarray, q_vals: np.ndarray, sample_centered: Callable):
    best_action_index = np.argmax(q_vals)
    best_action = actions[best_action_index]

    closest = False
    point = None
    n_sampled = 1
    while not closest and n_sampled <= 200:
        point = sample_centered(best_action)
        n_sampled += 1
        dists = np.linalg.norm(
            point - actions,
            axis=1
        )
        best_action_distance = dists[best_action_index]
        dists = np.delete(dists, best_action_index)
        closest = np.all(dists >= best_action_distance)
    return point


def voo(eps: float, sample_centered: Callable, node: Any, planner: Planner):
    prob = random.random()
    if prob <= 1 - eps:
        return voronoi(
            np.array([node.action for node in node.actions]),
            node.a_values,
            sample_centered
        )
    else:
        return uniform(node, planner)


def voo_vo(eps: float, sample_centered: Callable, node: Any, planner: Planner):
    prob = random.random()
    if prob <= 1 - eps:
        return voronoi_vo(
            actions=np.array([node.action for node in node.actions]),
            q_vals=node.a_values,
            sample_centered=sample_centered,
            x=node.state.x,
            obs=planner.environment.config.ob,
            dt=planner.environment.config.dt,
            ROBOT_RADIUS=planner.environment.config.robot_radius,
            OBS_RADIUS=planner.environment.config.obs_size
        )
    else:
        return uniform(node, planner)


def in_range(p: np.ndarray, rng: list):
    return rng[0] <= p[1] <= rng[1]


def voronoi_vo(actions, q_vals, sample_centered, x, obs, dt, ROBOT_RADIUS, OBS_RADIUS):
    best_action_index = np.argmax(q_vals)
    best_action = actions[best_action_index]
    forbidden_angular_vel = velocity_obstacle_nearest(x, obs, dt, ROBOT_RADIUS, OBS_RADIUS)

    closest = False
    point = None
    valid = False
    n_sampled = 1
    invalid_sample = -1
    while not closest and n_sampled <= 200:
        while not valid:
            invalid_sample += 1
            point = sample_centered(best_action)
            valid = in_range(point, forbidden_angular_vel)
        n_sampled += 1
        dists = np.linalg.norm(
            point - actions,
            axis=1
        )
        best_action_distance = dists[best_action_index]
        dists = np.delete(dists, best_action_index)
        closest = np.all(dists >= best_action_distance)
        print(f"INVALID SAMPLE: {invalid_sample}")
    return point
