import math
import random
from functools import partial
from typing import Callable, Any

import numpy as np
from scipy.special import softmax

from bettergym.agents.planner import Planner
from bettergym.agents.utils.utils import voronoi, clip_act
from mcts_utils import get_intersections


def sample_centered_robot_arena(center: np.ndarray, number):
    chosen = np.random.multivariate_normal(
        mean=center,
        cov=np.diag([0.3 / 2, 0.38 * 2]),
        size=number
    )
    # Make sure angle is within range of -π to π
    chosen[:, 1] = (chosen[:, 1] + math.pi) % (2 * math.pi) - math.pi
    return chosen


def sample_multiple_spaces(center, a_space, number, v_space):
    lengths = np.linalg.norm(a_space, axis=0)
    percentages = np.cumsum(softmax(lengths))
    pct = random.random()
    idx_space = np.flatnonzero(pct <= percentages)[0]
    return np.vstack([
        np.random.uniform(low=v_space[0], high=v_space[-1], size=number),
        np.random.uniform(low=a_space[idx_space][0], high=a_space[idx_space][1], size=number)
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
    angle_space = compute_safe_angle_space(intersection_points=intersection_points,
                                           max_angle_change=config.max_angle_change, x=x)
    velocity_space = [0.0, config.max_speed]

    # No angle at positive velocity is safe
    if angle_space is None:
        retro_available, angle_space = vo_negative_speed(obs, x, r1, config)
        if retro_available:
            # If VO with negative speed is possible, use it
            velocity_space = [config.min_speed, 0.0]
        else:
            velocity_space = [0.0]
            min_angle = ((x[2] - config.max_angle_change) + math.pi) % (2 * math.pi) - math.pi
            max_angle = ((x[2] + config.max_angle_change) + math.pi) % (2 * math.pi) - math.pi
            if min_angle > max_angle:
                angle_space = [[min_angle, math.pi], [-math.pi, max_angle]]
            else:
                angle_space = [[min_angle, max_angle]]

    return angle_space, velocity_space


def compute_safe_angle_space(intersection_points, max_angle_change, x):
    # convert points into angles and define the forbidden angles space
    forbidden_ranges = []
    for point in intersection_points:
        if point is not None:
            p1, p2 = point
            angle1 = math.atan2(p1[1] - x[1], p1[0] - x[0])
            angle2 = math.atan2(p2[1] - x[1], p2[0] - x[0])
            if angle1 > angle2:
                forbidden_ranges.extend([[angle1, math.pi], [-math.pi, angle2]])
            else:
                forbidden_ranges.append([angle1, angle2])
    robot_angles = [x[2] - max_angle_change, x[2] + max_angle_change]
    robot_angles = np.array(robot_angles)
    # Make sure angle is within range of -π to π
    robot_angles = (robot_angles + np.pi) % (2 * np.pi) - np.pi
    if type(robot_angles[0]) is np.float64:
        robot_angles = [robot_angles]
    new_robot_angles = []
    for a in robot_angles:
        if a[0] > a[1]:
            new_robot_angles.extend([[a[0], math.pi], [-math.pi, a[1]]])
        else:
            new_robot_angles.append(a)
    angle_spaces = []
    all_safe = True
    for rr in new_robot_angles:
        for fr in forbidden_ranges:
            # CASE 1 the forbidden range is inside the robot angles
            if rr[0] <= fr[0] <= rr[1] and rr[0] <= fr[1] <= rr[1]:
                all_safe = False
                angle_space = [
                    [rr[0], fr[0]],
                    [fr[1], rr[1]]
                ]
            # CASE 2 the forbidden range all the robot angles
            elif fr[0] <= rr[0] and fr[1] >= rr[1]:
                all_safe = False
                # all angles collide
                angle_space = [None]
            # CASE 3 the forbidden range starts before the robot angles and ends inside
            elif fr[0] <= rr[0] <= fr[1] <= rr[1]:
                all_safe = False
                angle_space = [
                    [fr[1], rr[1]]
                ]
            # CASE 4 the forbidden range starts in the robot angles and ends after
            elif fr[1] >= rr[1] >= fr[0] >= rr[0]:
                all_safe = False
                angle_space = [
                    [rr[0], fr[0]]
                ]
            # CASE 5 no overlap
            elif (fr[0] >= rr[1] and fr[1] >= rr[1]) or (fr[0] <= rr[0] and fr[0] <= rr[0]):
                continue
            else:
                raise Exception(f"The provided forbidden angles: {fr} does not match any case with "
                                f"the following angles available to the robot: {rr}")
            angle_spaces.extend(angle_space)
        # need to check all forbidden ranges before saying its safe
        if all_safe:
            angle_spaces.append(rr)
    angle_spaces = [a for a in angle_spaces if a is not None]
    if len(angle_spaces) == 0:
        return None
    else:
        return angle_spaces


def vo_negative_speed(obs, x, r1, config):
    VELOCITY = np.abs(config.min_speed)
    v = get_relative_velocity(VELOCITY, obs, x)
    r0 = np.linalg.norm(v, axis=1) * config.dt
    intersection_points = [get_intersections(x[:2], obs[i][:2], r0[i], r1[i]) for i in range(len(obs))]

    # check if there are any intersections
    if not any(intersection_points):
        # return a list of angles to explore
        angle_space = [x[2] - config.max_angle_change, x[2] + config.max_angle_change]
        # check if the angles are in the range -pi, pi
        angle_space[0] = (angle_space[0] + math.pi) % (2 * math.pi) - math.pi
        angle_space[1] = (angle_space[1] + math.pi) % (2 * math.pi) - math.pi
        if angle_space[0] > angle_space[1]:
            angle_space = [[angle_space[0], math.pi], [-math.pi, angle_space[1]]]
        else:
            angle_space = [angle_space]
        return True, angle_space
    else:
        # create a copy of the current state
        x_copy = x.copy()
        x_copy[2] += math.pi
        x_copy[2] = (x_copy[2] + math.pi) % (2 * math.pi) - math.pi
        angle_space = compute_safe_angle_space(intersection_points=intersection_points,
                                               max_angle_change=config.max_angle_change, x=x_copy)

        if angle_space is not None:
            # since angle_space was computed using the flipped angle
            # we need to flip it so that we'll have a range compatible with current robot direction
            angle_space = np.array(angle_space) + np.pi
            # Make sure angle is within range of -π to π
            angle_space = (angle_space + np.pi) % (2 * np.pi) - np.pi
            new_angle_space = []
            for a in angle_space:
                if a[0] > a[1]:
                    new_angle_space.extend([[a[0], math.pi], [-math.pi, a[1]]])
                else:
                    new_angle_space.append(a)
            return True, new_angle_space
        else:
            return False, angle_space


def voronoi_vo(actions, q_vals, sample_centered, x, intersection_points, config, obs, eps, r1):
    prob = random.random()

    # If there are no intersection points
    if not any(intersection_points):
        if prob <= 1 - eps and len(actions) != 0:
            chosen = voronoi(actions, q_vals, sample_centered)
            return clip_act(chosen=chosen, angle_change=config.max_angle_change,
                            min_speed=config.min_speed, max_speed=config.max_speed, x=x)
            # return voronoi(actions, q_vals, partial(sample, a_space=angle_space, v_space=velocity_space))
        else:
            velocity_space = [0.0, config.max_speed]
            min_angle = (x[2] - config.max_angle_change + math.pi) % (2 * math.pi) - math.pi
            max_angle = (x[2] + config.max_angle_change + math.pi) % (2 * math.pi) - math.pi
            if min_angle > max_angle:
                angle_space = [[min_angle, math.pi], [-math.pi, max_angle]]
            else:
                angle_space = [[min_angle, max_angle]]
            # Generate random actions
            return sample(center=None, a_space=angle_space, v_space=velocity_space, number=1)[0]
    # If there are intersection points
    else:
        # convert intersection points into ranges of available velocities/angles
        angle_space, velocity_space = get_spaces(intersection_points, x, obs, r1, config)

        # Use Voronoi with probability 1-eps, otherwise sample random actions
        if prob <= 1 - eps and len(actions) != 0:
            chosen = voronoi(actions, q_vals, partial(sample, a_space=angle_space, v_space=velocity_space))
            return chosen
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
    r1 *= 1.1

    # Calculate intersection points
    intersection_points = [get_intersections(x[:2], obs_x[i][:2], r0[i], r1[i]) for i in range(len(obs_x))]

    # Choose the best action using Voronoi VO
    actions = np.array([n.action for n in node.actions])
    q_vals = node.a_values
    config = planner.environment.gym_env.config
    chosen = voronoi_vo(actions, q_vals, sample_centered, x, intersection_points, config, obs_x, eps, r1)

    return chosen
