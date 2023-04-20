import math
import random
from functools import partial
from typing import Any, Callable

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist
from scipy.special import softmax

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
    N_SAMPLE = 500
    valid = False

    while not valid:
        # find the index of the action with the highest Q-value
        best_action_index = np.argmax(q_vals)

        # get the action with the highest Q-value
        best_action = actions[best_action_index]

        # generate 200 random points centered around the best action
        points = sample_centered(best_action, N_SAMPLE)

        # compute the Euclidean distances between each point and each action
        # column -> actions
        # rows -> points
        dists = cdist(points, actions, 'euclidean')

        # find the distances between each point and the best action
        best_action_distances = dists[:, best_action_index]

        # repeat the distances for each action except the best action (necessary for doing `<=` later)
        best_action_distances_rep = np.tile(best_action_distances, (dists.shape[1] - 1, 1)).T

        # remove the column for the best action from the distance matrix
        # dists = np.hstack((dists[:, :best_action_index], dists[:, best_action_index + 1:]))
        dists = np.delete(dists, best_action_index, axis=1)

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
    if prob <= 1 - eps and len(node.actions) != 0:
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
    return chosen


def voo_vo(eps: float, sample_centered: Callable, node: Any, planner: Planner):
    obs_x = []
    obs_rad = []
    for ob in node.state.obstacles:
        obs_x.append(ob.x)
        obs_rad.append(ob.radius)
    x = node.state.x
    obs = np.array(obs_x)
    dt = planner.environment.config.dt
    ROBOT_RADIUS = planner.environment.config.robot_radius
    OBS_RADIUS = np.array(obs_rad)

    VMAX = 0.3
    # 0 is the velocity of the obstacle, if its moving then change
    r0 = VMAX * dt + obs[:, 3] * dt
    r1 = ROBOT_RADIUS + OBS_RADIUS + 0.1
    intersection_points = [get_intersections(x[:2], obs[i][:2], r0[i], r1[i]) for i in range(len(obs))]
    config = planner.environment.gym_env.config
    chosen = voronoi_vo(
        actions=np.array([node.action for node in node.actions]),
        q_vals=node.a_values,
        sample_centered=sample_centered,
        x=x,
        intersection_points=intersection_points,
        config=config,
        eps=eps
    )

    chosen = clip_act(
        chosen=chosen,
        angle_change=config.max_angle_change,
        min_speed=config.min_speed,
        max_speed=config.max_speed,
        x=node.state.x
    )
    return chosen


def in_range(p: np.ndarray, rng: list):
    return rng[0] <= p[1] <= rng[1]


def compute_angle_space(intersection_points, config, x):
    # convert points into angles and define the forbidden angles space
    min_angle = np.inf
    max_angle = -np.inf
    for point in intersection_points:
        if point is not None:
            p1, p2 = point
            angle1 = math.atan2(p1[1] - x[1], p1[0] - x[0])
            min_angle = min(angle1, min_angle)
            angle2 = math.atan2(p2[1] - x[1], p2[0] - x[0])
            max_angle = max(angle2, max_angle)
    robot_angles = [x[2] - config.max_angle_change, x[2] + config.max_angle_change]
    forbidden_angles = [min_angle, max_angle]

    # CASE 3: the forbidden angle range is inside the angles available to the robot
    if forbidden_angles[0] > robot_angles[0] and forbidden_angles[1] < robot_angles[1]:
        angle_space = [
            [robot_angles[0], forbidden_angles[0]],
            [forbidden_angles[1], robot_angles[1]]
        ]
    # CASE 4: the forbidden angle range is bigger than the range of angles available to the robot
    elif forbidden_angles[0] < robot_angles[0] and forbidden_angles[1] > robot_angles[1]:
        angle_space = None
    # CASE 1: the forbidden angle range starts before the range of angles available to the robot
    elif forbidden_angles[0] < robot_angles[0]:
        angle_space = [
            [robot_angles[0], forbidden_angles[0]]
        ]
    # CASE 2: the forbidden angle range ends after the range of angles available to the robot
    elif forbidden_angles[1] > robot_angles[1]:
        angle_space = [
            [robot_angles[0], forbidden_angles[0]]
        ]
    else:
        raise Exception(f"The provided forbidden angles: {forbidden_angles} does not match any case with "
                        f"the following angles available to the robot: {robot_angles}")

    return angle_space


def voronoi_vo(actions, q_vals, sample_centered, x, intersection_points, config, eps):
    prob = random.random()
    # check if the list contains only None
    if not any(intersection_points):
        if prob <= 1 - eps and len(actions) != 0:
            chosen = voronoi(
                actions,
                q_vals,
                sample_centered
            )
        else:
            points1 = np.random.uniform(low=config.min_speed, high=config.max_speed)
            points2 = np.random.uniform(low=x[2] - config.max_angle_change, high=x[2] + config.max_angle_change)
            return np.vstack([points1, points2]).T[0]
    else:
        def sample_multiple_spaces(center, space, number):
            length0 = np.linalg.norm(space[0])
            length1 = np.linalg.norm(space[1])

            percentages = softmax([length0, length1])
            pct = random.random()
            if pct <= percentages[0]:
                return np.vstack([
                    np.random.uniform(low=config.min_speed, high=config.max_speed, size=number),
                    np.random.uniform(low=space[0][0], high=space[0][1], size=number)
                ]).T
            else:
                return np.vstack([
                    np.random.uniform(low=config.min_speed, high=config.max_speed, size=number),
                    np.random.uniform(low=space[1][0], high=space[1][1], size=number)
                ]).T

        def sample_single_space(center, space, number):
            return np.vstack([
                np.random.uniform(low=config.min_speed, high=config.max_speed, size=number),
                np.random.uniform(low=space[0], high=space[1], size=number)
            ]).T

        angle_space = compute_angle_space(intersection_points=intersection_points, config=config, x=x)
        if angle_space is None:
            return np.array([config.min_speed, x[2]])
        elif len(angle_space) == 1:
            sample = sample_single_space
        else:
            sample = sample_multiple_spaces

        if prob <= 1 - eps and len(actions) != 0:
            chosen = voronoi(
                actions,
                q_vals,
                partial(sample, space=angle_space)
            )
        else:
            return sample(None, angle_space, 1)[0]

    chosen = clip_act(
        chosen=chosen,
        angle_change=config.max_angle_change,
        min_speed=config.min_speed,
        max_speed=config.max_speed,
        x=x
    )
    return chosen
