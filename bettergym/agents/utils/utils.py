import math
import random
from typing import Any, Callable

import numpy as np
from numba import njit

from bettergym.agents.planner import Planner
from mcts_utils import get_intersections


def uniform(node: Any, planner: Planner):
    current_state = node.state
    available_actions = planner.environment.get_actions(current_state)
    return available_actions.sample()


def uniform_discrete(node: Any, planner: Planner):
    current_state = node.state
    actions = planner.environment.get_actions(current_state)
    return random.choice(actions)


@njit
def compute_towards_goal_jit(x: np.ndarray, goal: np.ndarray, max_angle_change: float, var_angle: float,
                             min_speed: float,
                             max_speed: float):
    mean_angle = np.arctan2(goal[1] - x[1], goal[0] - x[0])
    # Make sure angle is within range of -π to π
    # var_angle = max_angle_change ** 3
    angle = np.random.normal(mean_angle, var_angle)
    linear_velocity = np.random.uniform(
        a=min_speed,
        b=max_speed
    )
    max_angle = x[2] + max_angle_change
    min_angle = x[2] - max_angle_change
    if angle > max_angle:
        angle = max_angle
    elif angle < min_angle:
        angle = min_angle

    return np.array([linear_velocity, angle])


def towards_goal(node: Any, planner: Planner, var_angle: float):
    config = planner.environment.config
    return compute_towards_goal_jit(node.state.x, node.state.goal, config.max_angle_change, var_angle, config.min_speed,
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
    N_SAMPLE = 200
    valid = False

    while not valid:
        # find the index of the action with the highest Q-value
        best_action_index = np.argmax(q_vals)

        # get the action with the highest Q-value
        best_action = actions[best_action_index]

        # generate 200 random points centered around the best action
        points = sample_centered(best_action, N_SAMPLE)

        # compute the Euclidean distances between each point and each action
        dists = np.linalg.norm(points[:, np.newaxis, :] - actions, axis=2)

        # find the distances between each point and the best action
        best_action_distances = dists[:, best_action_index]

        # repeat the distances for each action except the best action
        best_action_distances_rep = np.tile(best_action_distances, (dists.shape[1] - 1, 1)).T

        # remove the column for the best action from the distance matrix
        dists = np.hstack((dists[:, :best_action_index], dists[:, best_action_index + 1:]))

        # find the closest action to each point
        closest = best_action_distances_rep <= dists

        # find the rows where all distances to other actions are greater than the distance to the best action
        all_true_rows = np.where(np.all(closest, axis=1))[0]

        # find the index of the point closest to the best action among the valid rows
        valid_points = best_action_distances[all_true_rows]
        if len(valid_points >= 0):
            closest_point_idx = np.argmin(valid_points)
            # return the closest point to the best action
            return points[closest_point_idx]


@njit
def clip_act(chosen, angle_change, min_speed, max_speed, x):
    chosen[0] = max(min_speed, min(chosen[0], max_speed))
    chosen[1] = max(x[2] - angle_change, min(chosen[1], x[2] + angle_change))
    return chosen


def voo(eps: float, sample_centered: Callable, node: Any, planner: Planner):
    prob = random.random()
    if prob <= 1 - eps:
        chosen = voronoi(
            np.array([node.action for node in node.actions]),
            node.a_values,
            sample_centered
        )
    else:
        chosen = uniform(node, planner)
    config = planner.environment.gym_env.config
    chosen = clip_act(
        chosen=chosen,
        angle_change=config.max_angle_change,
        min_speed=config.min_speed,
        max_speed=config.max_speed,
        x=node.state.x
    )
    # chosen[0] = max(-0.1, min(chosen[0], 0.3))
    # chosen[1] = max(node.state.x[2] - config.max_angle_change, min(chosen[1], node.state.x[2] + config.max_angle_change))
    return chosen


def voo_vo(eps: float, sample_centered: Callable, node: Any, planner: Planner):
    prob = random.random()
    if prob <= 1 - eps:
        obs_pos = []
        obs_rad = []
        for ob in node.state.obstacles:
            obs_pos.append(ob.x[:2])
            obs_rad.append(ob.radius)
        return voronoi_vo(
            actions=np.array([node.action for node in node.actions]),
            q_vals=node.a_values,
            sample_centered=sample_centered,
            x=node.state.x,
            obs=np.array(obs_pos),
            dt=planner.environment.config.dt,
            ROBOT_RADIUS=planner.environment.config.robot_radius,
            OBS_RADIUS=np.array(obs_rad)
        )
    else:
        return uniform(node, planner)


def in_range(p: np.ndarray, rng: list):
    return rng[0] <= p[1] <= rng[1]


def voronoi_vo(actions, q_vals, sample_centered, x, obs, dt, ROBOT_RADIUS, OBS_RADIUS):
    VMAX = 0.3
    # 0 is the velocity of the obstacle, if its moving then change
    r0 = VMAX * dt + 0 * dt
    r1 = ROBOT_RADIUS + OBS_RADIUS
    intersection_points = [get_intersections(x[:2], obs[i], r0, r1[i]) for i in range(len(obs))]
    # check if the list contains only None
    if not any(intersection_points):
        return voronoi(
            actions,
            q_vals,
            sample_centered
        )
    else:
        not_none = [point for point in intersection_points if point is not None]
        min_angle = np.inf
        max_angle = -np.inf
        for t in not_none:
            for p in t:
                angle = math.atan2(p[1] - x[1], p[0] - x[0])
                max_angle = max(angle, max_angle)
                min_angle = min(angle, min_angle)
        print("VELOCITY OBSTACLE")
