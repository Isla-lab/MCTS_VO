import math

import numpy as np


def uniform_random(node, planner):
    state = node.state
    config = planner.environment.gym_env.config
    action = np.random.uniform(
        low=np.array(
            [config.min_speed, state.x[2] - config.max_angle_change], dtype=np.float64
        ),
        high=np.array(
            [config.max_speed, state.x[2] + config.max_angle_change], dtype=np.float64
        ),
    )
    action[1] = (action[1] + math.pi) % (2 * math.pi) - math.pi
    return action


def compute_int_vectorized(r0, r1, d, x0, x1, y0, y1):
    """
    Vectorized computation of intersection points between two circles
    :param r0: radius of the first circle
    :param r1: radius of the second circle
    :param d: distance between the two circle centers
    :param x0: x position of the first circle center
    :param x1: x position of the second circle center
    :param y0: y position of the first circle center
    :param y1: y position of the second circle center
    :return: array of coordinates of the two intersection points
    """
    # https://stackoverflow.com/a/55817881
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r0 ** 2 - a ** 2)
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d
    x3 = x2 + h * (y1 - y0) / d
    y3 = y2 - h * (x1 - x0) / d
    x4 = x2 - h * (y1 - y0) / d
    y4 = y2 + h * (x1 - x0) / d

    return np.array([(x3, y3), (x4, y4)])


def get_intersections_vectorized(x, obs_x, r0, r1):
    x_exp = np.expand_dims(x, 1)
    d = np.hypot(obs_x[:, 0] - x_exp[0, :], obs_x[:, 1]-x_exp[1, :])

    # Non-intersecting
    no_intersection = d > r0 + r1

    # One circle within the other
    one_within_other = d < np.abs(r0 - r1)

    # Coincident circles
    coincident = np.logical_and(d == 0, r0 == r1)

    intersecting = np.logical_not(np.logical_or.reduce((no_intersection, one_within_other, coincident)))

    # Compute intersection points
    intersection_points = compute_int_vectorized(
        r0[intersecting],
        r1[intersecting],
        d[intersecting],
        x_exp[0, :],
        obs_x[intersecting, 0],
        x_exp[1, :],
        obs_x[intersecting, 1],
    )

    output_vec = np.empty((2, 2, len(d)))
    output_vec[:] = None
    output_vec[:, :, np.logical_or(one_within_other, coincident)] = np.inf
    output_vec[:, :, intersecting] = intersection_points
    return output_vec
