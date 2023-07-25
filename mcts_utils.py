import math

import numpy as np


def uniform_random(node, planner):
    state = node.state
    config = planner.environment.gym_env.config
    action = np.random.uniform(
        low=np.array([config.min_speed, state.x[2] - config.max_angle_change], dtype=np.float64),
        high=np.array([config.max_speed, state.x[2] + config.max_angle_change], dtype=np.float64)
    )
    action[1] = (action[1] + math.pi) % (2 * math.pi) - math.pi
    return action


def compute_int(r0, r1, d, x0, x1, y0, y1):
    """
    Computes the intersection points between two circles
    :param r0: radius of the first circle
    :param r1: radius of the second circle
    :param d: distance between the two circle centers
    :param x0: x position of the first circle center
    :param x1: x position of the second circle center
    :param y0: y position of the first circle center
    :param y1: y position of the second circle center
    :return: the coordinates of the two intersection points
    """
    # https://stackoverflow.com/a/55817881/8785420
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(r0 ** 2 - a ** 2)
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d
    x3 = x2 + h * (y1 - y0) / d
    y3 = y2 - h * (x1 - x0) / d

    x4 = x2 - h * (y1 - y0) / d
    y4 = y2 + h * (x1 - x0) / d

    return (x3, y3), (x4, y4)


def get_intersections(p0, p1, r0, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    x0, y0 = p0
    x1, y1 = p1

    d = math.hypot(p0[0] - p1[0], p0[1] - p1[1])

    # non-intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return np.inf
    # coincident circles
    if d == 0 and r0 == r1:
        return np.inf
    else:
        return compute_int(r0, r1, d, x0, x1, y0, y1)
