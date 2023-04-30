import math
import random
from functools import partial
from typing import Callable, Any

import numpy as np
from scipy.special import softmax

from bettergym.agents.planner import Planner
from bettergym.agents.utils.utils import voronoi, clip_act
from mcts_utils import get_intersections


def sample_multiple_spaces(center, a_space, number, v_space):
    length0 = np.linalg.norm(a_space[0])
    length1 = np.linalg.norm(a_space[1])
    percentages = softmax([length0, length1])
    pct = random.random()
    if pct <= percentages[0]:
        return np.vstack([
            np.random.uniform(low=v_space[0], high=v_space[-1], size=number),
            np.random.uniform(low=a_space[0][0], high=a_space[0][1], size=number)
        ]).T
    else:
        return np.vstack([
            np.random.uniform(low=v_space[0], high=v_space[-1], size=number),
            np.random.uniform(low=a_space[1][0], high=a_space[1][1], size=number)
        ]).T


def sample_single_space(center, a_space, number, v_space):
    return np.vstack([
        np.random.uniform(low=v_space[0], high=v_space[-1], size=number),
        np.random.uniform(low=a_space[0], high=a_space[1], size=number)
    ]).T


def sample(center, a_space, v_space, number):
    if len(a_space) == 1:
        return sample_single_space(center, a_space[0], number, v_space)
    else:
        return sample_multiple_spaces(center, a_space, number, v_space)


def get_spaces(intersection_points, x, obs, r1, config):
    angle_space = compute_angle_space(intersection_points=intersection_points,
                                      max_angle_change=config.max_angle_change, x=x)
    velocity_space = [config.min_speed, config.max_speed]

    if angle_space is None:
        retro_available, angle_space = vo_negative_speed(obs, x, r1, config)
        if retro_available:
            # If VO with negative speed is possible, use it
            velocity_space = [config.min_speed, 0.0]
        else:
            velocity_space = [0.0]

    return angle_space, velocity_space


def compute_angle_space(intersection_points, max_angle_change, x):
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
    robot_angles = [x[2] - max_angle_change, x[2] + max_angle_change]
    forbidden_angles = [min_angle, max_angle]

    # Check different cases for angle spaces
    if forbidden_angles[0] > robot_angles[0] and forbidden_angles[1] < robot_angles[1]:
        angle_space = [
            [robot_angles[0], forbidden_angles[0]],
            [forbidden_angles[1], robot_angles[1]]
        ]
    elif forbidden_angles[0] < robot_angles[0] and forbidden_angles[1] > robot_angles[1]:
        angle_space = None
    elif forbidden_angles[0] < robot_angles[0]:
        angle_space = [
            [forbidden_angles[1], robot_angles[1]]
        ]
    elif forbidden_angles[1] > robot_angles[1]:
        angle_space = [
            [robot_angles[0], forbidden_angles[0]]
        ]
    else:
        raise Exception(f"The provided forbidden angles: {forbidden_angles} does not match any case with "
                        f"the following angles available to the robot: {robot_angles}")

    return angle_space


def vo_negative_speed(obs, x, r1, config):
    VELOCITY = 0.1
    v = get_relative_velocity(VELOCITY, obs, x)
    r0 = np.linalg.norm(v, axis=1) * config.dt
    intersection_points = [get_intersections(x[:2], obs[i][:2], r0[i], r1[i]) for i in range(len(obs))]

    # check if there are any intersections
    if not any(intersection_points):
        # return a list of angles to explore
        angle_space = [[x[2] + math.pi - config.max_angle_change, x[2] + math.pi + config.max_angle_change]]
        return True, angle_space

    # create a copy of the current state
    x_copy = x.copy()
    x_copy[2] += math.pi
    angle_space = compute_angle_space(intersection_points=intersection_points,
                                      max_angle_change=config.max_angle_change, x=x_copy)
    return angle_space is not None, angle_space


def voronoi_vo(actions, q_vals, sample_centered, x, intersection_points, config, obs, eps, r1):
    prob = random.random()

    # If there are no intersection points
    if not any(intersection_points):
        if prob <= 1 - eps and len(actions) != 0:
            chosen = voronoi(actions, q_vals, sample_centered)
            return clip_act(chosen=chosen, angle_change=config.max_angle_change,
                            min_speed=config.min_speed, max_speed=config.max_speed, x=x)
        else:
            # Generate random actions
            velocity_space = [config.min_speed, config.max_speed]
            angle_space = [[x[2] - config.max_angle_change, x[2] + config.max_angle_change]]
            return sample(center=None, a_space=angle_space, v_space=velocity_space, number=1)[0]
    # If there are intersection points
    else:
        angle_space, velocity_space = get_spaces(intersection_points, x, obs, r1, config)

        # Use Voronoi with probability 1-eps, otherwise sample random actions
        if prob <= 1 - eps and len(actions) != 0:
            chosen = voronoi(actions, q_vals, partial(sample, a_space=angle_space, v_space=velocity_space))
            if chosen[0] <= 0:
                if chosen[0] == 0:
                    return chosen
                else:
                    x_copy = x.copy()
                    x_copy[2] += math.pi
                    return clip_act(chosen=chosen, angle_change=config.max_angle_change,
                                    min_speed=config.min_speed, max_speed=config.max_speed, x=x_copy)
            else:
                return clip_act(chosen=chosen, angle_change=config.max_angle_change,
                                min_speed=config.min_speed, max_speed=config.max_speed, x=x)
        else:
            return sample(center=None, a_space=angle_space, v_space=velocity_space, number=1)[0]


def get_relative_velocity(velocity: float, obs_x: np.ndarray, x: np.ndarray):
    conjunction_angle = np.arctan2(obs_x[:, 1] - x[1], obs_x[:, 0] - x[0])
    v1_vec = velocity * np.column_stack((np.cos(conjunction_angle), np.sin(conjunction_angle)))
    v2_vec = obs_x[:, 3][:, np.newaxis] * np.column_stack((np.cos(obs_x[:, 2]), np.sin(obs_x[:, 2])))
    return v1_vec - v2_vec


def voo_vo(eps: float, sample_centered: Callable, node: Any, planner: Planner):
    # Extract obstacle information
    obstacles = node.state.obstacles
    obs_x = np.array([ob.x for ob in obstacles])
    obs_rad = np.array([ob.radius for ob in obstacles])

    # Extract robot information
    x = node.state.x
    dt = planner.environment.config.dt
    ROBOT_RADIUS = planner.environment.config.robot_radius
    VMAX = 0.3

    # Calculate velocities
    v = get_relative_velocity(VMAX, obs_x, x)

    # Calculate radii
    r0 = np.linalg.norm(v, axis=1) * dt
    r1 = ROBOT_RADIUS + obs_rad
    # increment by ten percent radius 1
    r1 = r1 * 1.1

    # Calculate intersection points
    intersection_points = [get_intersections(x[:2], obs_x[i][:2], r0[i], r1[i]) for i in range(len(obs_x))]

    # Choose the best action using Voronoi VO
    actions = np.array([n.action for n in node.actions])
    q_vals = node.a_values
    config = planner.environment.gym_env.config
    chosen = voronoi_vo(actions, q_vals, sample_centered, x, intersection_points, config, obs_x, eps, r1)

    return chosen
