import math
import random
from copy import deepcopy
from functools import partial
from typing import Callable, Any

import numpy as np
import portion as P

from bettergym.agents.planner import Planner
from bettergym.agents.utils.utils import (
    voronoi,
    clip_act,
    compute_towards_goal_jit,
    get_robot_angles, compute_uniform_towards_goal_jit,
)
from mcts_utils import get_intersections, uniform_random


def towards_goal_vo(node: Any, planner: Planner, std_angle_rollout: float):
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
    r1 *= 1.05

    # Calculate intersection points
    intersection_points = [
        get_intersections(x[:2], obs_x[i][:2], r0[i], r1[i]) for i in range(len(obs_x))
    ]
    config = planner.environment.gym_env.config
    # If there are no intersection points
    if not any(intersection_points):
        return compute_towards_goal_jit(
            x,
            node.state.goal,
            config.max_angle_change,
            std_angle_rollout,
            config.min_speed,
            config.max_speed,
        )
    else:
        # convert intersection points into ranges of available velocities/angles
        angle_space, velocity_space = get_spaces(
            intersection_points, x, obs_x, r1, config
        )
        return sample(
            center=None, a_space=angle_space, v_space=velocity_space, number=1
        )[0]


def uniform_towards_goal_vo(node: Any, planner: Planner, std_angle_rollout: float):
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
    r1 *= 1.05

    # Calculate intersection points
    intersection_points = [
        get_intersections(x[:2], obs_x[i][:2], r0[i], r1[i]) for i in range(len(obs_x))
    ]
    config = planner.environment.gym_env.config
    # If there are no intersection points
    if not any(intersection_points):
        return compute_uniform_towards_goal_jit(
            x,
            node.state.goal,
            config.max_angle_change,
            std_angle_rollout,
            config.min_speed,
            config.max_speed,
        )
    else:
        # convert intersection points into ranges of available velocities/angles
        angle_space, velocity_space = get_spaces(
            intersection_points, x, obs_x, r1, config
        )
        return sample(
            center=None, a_space=angle_space, v_space=velocity_space, number=1
        )[0]


def sample_centered_robot_arena(
        center: np.ndarray, number: int, clip_fn: Callable, std_angle: float
):
    chosen = np.random.multivariate_normal(
        mean=center, cov=np.diag([0.3 / 2, std_angle]), size=number
    )
    chosen = clip_fn(chosen=chosen)
    return chosen


def sample_multiple_spaces(center, a_space, number, v_space):
    lengths = np.linalg.norm(a_space, axis=1)
    # percentages = np.cumsum(softmax(lengths))
    percentages = np.cumsum(lengths / np.sum(lengths))
    pct = random.random()
    idx_space = np.flatnonzero(pct <= percentages)[0]
    return np.vstack(
        [
            np.random.uniform(low=v_space[0], high=v_space[-1], size=number),
            np.random.uniform(
                low=a_space[idx_space][0], high=a_space[idx_space][1], size=number
            ),
        ]
    ).T


def sample_single_space(center, a_space, number, v_space):
    return np.vstack(
        [
            np.random.uniform(low=v_space[0], high=v_space[-1], size=number),
            np.random.uniform(low=a_space[0], high=a_space[1], size=number),
        ]
    ).T


def sample(center, a_space, v_space, number):
    if len(a_space) == 1:
        return sample_single_space(center, a_space[0], number, v_space)
    else:
        return sample_multiple_spaces(center, a_space, number, v_space)


def get_spaces(intersection_points, x, obs, r1, config):
    angle_space = compute_safe_angle_space(
        intersection_points=intersection_points,
        max_angle_change=config.max_angle_change,
        x=x,
    )
    velocity_space = [0.0, config.max_speed]

    # No angle at positive velocity is safe
    if angle_space is None:
        retro_available, angle_space = vo_negative_speed(obs, x, r1, config)
        if retro_available:
            # If VO with negative speed is possible, use it
            velocity_space = [config.min_speed, 0.0]
        else:
            velocity_space = [0.0]
            angle_space = get_robot_angles(x, config.max_angle_change)

    return angle_space, velocity_space


def range_difference(rr, fr):
    # # CASE 2 the forbidden range is all the robot angles
    # if fr[0] <= rr[0] and fr[1] >= rr[1]:
    #     # all angles collide
    #     angle_space = None
    # # CASE 4 the forbidden range starts in the robot angles and ends after
    # elif rr[0] < fr[0] < rr[1] <= fr[1]:
    #     angle_space = [
    #         [rr[0], fr[0]]
    #     ]
    # # CASE 1 the forbidden range is inside the robot angles
    # elif rr[0] < fr[0] < rr[1] and rr[0] < fr[1] < rr[1]:
    #     angle_space = [
    #         [rr[0], fr[0]],
    #         [fr[1], rr[1]]
    #     ]
    # # CASE 3 the forbidden range starts before the robot angles and ends inside
    # elif fr[0] <= rr[0] < fr[1] < rr[1]:
    #     angle_space = [
    #         [fr[1], rr[1]]
    #     ]
    # # CASE 5 no overlap
    # elif (fr[0] >= rr[1] and fr[1] >= rr[1]) or (fr[0] <= rr[0] and fr[0] <= rr[0]):
    #     angle_space = [rr]
    # else:
    #     raise Exception(f"The provided forbidden angles: {fr} does not match any case with "
    #                     f"the following angles available to the robot: {rr}")

    rr = P.closed(rr[0], rr[1])
    fr = P.closed(fr[0], fr[1])
    angle_space = [list(r[1:3]) for r in P.to_data(rr - fr)]
    return angle_space if angle_space is not [()] else None


def compute_safe_angle_space(intersection_points, max_angle_change, x):
    robot_angles = get_robot_angles(x, max_angle_change)

    # convert points into angles and define the forbidden angles space
    forbidden_ranges = []
    for point in intersection_points:
        if point is np.inf:
            forbidden_ranges.extend(robot_angles)
        elif point is not None:
            p1, p2 = point
            angle1 = math.atan2(p1[1] - x[1], p1[0] - x[0])
            angle2 = math.atan2(p2[1] - x[1], p2[0] - x[0])
            if angle1 > angle2:
                forbidden_ranges.extend([[angle1, math.pi], [-math.pi, angle2]])
            else:
                forbidden_ranges.append([angle1, angle2])

    new_ranges = []
    for fr in forbidden_ranges:
        for rr in robot_angles:
            output = range_difference(rr, fr)
            if output is not None:
                new_ranges.extend(output)
        robot_angles = deepcopy(new_ranges)
        new_ranges = []

    if len(robot_angles) == 0:
        return None
    else:
        return robot_angles


def vo_negative_speed(obs, x, r1, config):
    VELOCITY = np.abs(config.min_speed)
    v = get_relative_velocity(VELOCITY, obs, x)
    r0 = np.linalg.norm(v, axis=1) * config.dt
    intersection_points = [
        get_intersections(x[:2], obs[i][:2], r0[i], r1[i]) for i in range(len(obs))
    ]

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
        angle_space = compute_safe_angle_space(
            intersection_points=intersection_points,
            max_angle_change=config.max_angle_change,
            x=x_copy,
        )

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


def voronoi_vo(
        actions, q_vals, sample_centered, x, intersection_points, config, obs, eps, r1
):
    prob = random.random()

    # If there are no intersection points
    if not any(intersection_points):
        if prob <= 1 - eps and len(actions) != 0:
            return voronoi(
                actions,
                q_vals,
                partial(
                    sample_centered,
                    clip_fn=partial(
                        clip_act,
                        max_angle_change=config.max_angle_change,
                        x=x,
                        allow_negative=False,
                    ),
                ),
            )
        else:
            velocity_space = [0.0, config.max_speed]
            angle_space = get_robot_angles(x, config.max_angle_change)
            # Generate random actions
            return sample(
                center=None, a_space=angle_space, v_space=velocity_space, number=1
            )[0]
    # If there are intersection points
    else:
        # convert intersection points into ranges of available velocities/angles
        angle_space, velocity_space = get_spaces(
            intersection_points, x, obs, r1, config
        )

        # Use Voronoi with probability 1-eps, otherwise sample random actions
        if prob <= 1 - eps and len(actions) != 0:
            chosen = voronoi(
                actions,
                q_vals,
                partial(sample, a_space=angle_space, v_space=velocity_space),
            )
            return chosen
        else:
            return sample(
                center=None, a_space=angle_space, v_space=velocity_space, number=1
            )[0]


def get_relative_velocity(velocity: float, obs_x: np.ndarray, x: np.ndarray):
    conjunction_angle = np.arctan2(obs_x[:, 1] - x[1], obs_x[:, 0] - x[0])
    v1_vec = velocity * np.column_stack(
        (np.cos(conjunction_angle), np.sin(conjunction_angle))
    )
    v2_vec = obs_x[:, 3][:, np.newaxis] * np.column_stack(
        (np.cos(obs_x[:, 2]), np.sin(obs_x[:, 2]))
    )
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
    r1 *= 1.05

    # Calculate intersection points
    intersection_points = [
        get_intersections(x[:2], obs_x[i][:2], r0[i], r1[i]) for i in range(len(obs_x))
    ]

    # Choose the best action using Voronoi VO
    actions = np.array([n.action for n in node.actions])
    q_vals = node.a_values
    config = planner.environment.gym_env.config
    chosen = voronoi_vo(
        actions, q_vals, sample_centered, x, intersection_points, config, obs_x, eps, r1
    )

    return chosen


def uniform_random_vo(node, planner):
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
    r1 *= 1.05

    # Calculate intersection points
    intersection_points = [
        get_intersections(x[:2], obs_x[i][:2], r0[i], r1[i]) for i in range(len(obs_x))
    ]
    config = planner.environment.gym_env.config
    # If there are no intersection points
    if not any(intersection_points):
        return uniform_random(node, planner)
    else:
        # convert intersection points into ranges of available velocities/angles
        angle_space, velocity_space = get_spaces(
            intersection_points, x, obs_x, r1, config
        )
        return sample(
            center=None, a_space=angle_space, v_space=velocity_space, number=1
        )[0]


def epsilon_normal_uniform_vo(
        node: Any, planner: Planner, std_angle_rollout: float, eps=0.1
):
    prob = random.random()
    if prob <= 1 - eps:
        return towards_goal_vo(node, planner, std_angle_rollout)
    else:
        return uniform_random_vo(node, planner)


def epsilon_uniform_uniform_vo(
        node: Any, planner: Planner, std_angle_rollout: float, eps=0.1
):
    prob = random.random()
    if prob <= 1 - eps:
        return uniform_towards_goal_vo(node, planner, std_angle_rollout)
    else:
        return uniform_random_vo(node, planner)
