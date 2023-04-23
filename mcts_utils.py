import math

import numpy as np


def sample_centered_robot_arena(center: np.ndarray, number):
    return np.random.multivariate_normal(
        mean=center,
        cov=np.diag([0.3 / 2, 0.38 * 2]),
        size=number
    )




def uniform_random(node, planner):
    state = node.state
    config = planner.config
    return np.uniform(
        low=np.array([config.min_speed, state.x[2] - config.max_angle_change], dtype=np.float64),
        high=np.array([config.max_speed, state.x[2] + config.max_angle_change], dtype=np.float64)
    )


def compute_int(r0, r1, d, x0, x1, y0, y1):
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
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        return compute_int(r0, r1, d, x0, x1, y0, y1)
