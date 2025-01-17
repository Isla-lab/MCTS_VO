import math

import numpy as np
try:
    from MCTS_VO.bettergym.compiled_utils import get_tangents
except ModuleNotFoundError:
    from bettergym.compiled_utils import get_tangents


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

    return np.column_stack((np.column_stack((x3, y3)), np.column_stack((x4, y4))))

# @jit(cache=True, nopython=True)
# def check_circle_segment_intersect(robot_pos, robot_radius, segments):
#     r2 = robot_radius ** 2
#     A = segments[:, 2] - segments[:, 0]
#     B = segments[:, 3] - segments[:, 1]
#     C = segments[:, 0] - robot_pos[0]
#     D = segments[:, 1] - robot_pos[1]
#     a = A ** 2 + B ** 2
#     b = 2 * (A * C + B * D)
#     c = C ** 2 + D ** 2 - r2
#     discriminant = b ** 2 - 4 * a * c

#     valid_discriminant = discriminant >= 0
#     return valid_discriminant, discriminant, b, a, segments, A, B



# def find_circle_segment_intersections(discriminant, valid_discriminant, b, a, segments, A, B):
#     """
#     Find the intersection points between circle segments and lines.
#     Parameters:
#     discriminant (np.ndarray): Array of discriminant values for the quadratic equation.
#     valid_discriminant (np.ndarray): Boolean array indicating which discriminant values are valid.
#     b (np.ndarray): Array of b coefficients for the quadratic equation.
#     a (np.ndarray): Array of a coefficients for the quadratic equation.
#     segments (np.ndarray): Array of segment start points.
#     A (np.ndarray): Array of x-direction vectors for the segments.
#     B (np.ndarray): Array of y-direction vectors for the segments.
#     Returns:
#     np.ndarray: Array of intersection points. Each row contains four values: 
#                 [xi_t1, yi_t1, xi_t2, yi_t2], where (xi_t1, yi_t1) and (xi_t2, yi_t2) 
#                 are the intersection points.
#     """
    
#     sqrt_discriminant = np.sqrt(discriminant[valid_discriminant])
#     t1 = (-b[valid_discriminant] + sqrt_discriminant) / (2 * a[valid_discriminant])
#     t2 = (-b[valid_discriminant] - sqrt_discriminant) / (2 * a[valid_discriminant])

#     valid_t1 = np.logical_and(t1 >= 0, t1 <= 1)
#     valid_t2 = np.logical_and(t2 >= 0, t2 <= 1)

#     if np.any(valid_t1):
#         xi_t1 = segments[valid_discriminant, 0] + t1[valid_t1] * A[valid_discriminant]
#         yi_t1 = segments[valid_discriminant, 1] + t1[valid_t1] * B[valid_discriminant]
#     else:
#         return np.empty((0, 4))

#     if np.any(valid_t2):
#         xi_t2 = segments[valid_discriminant, 0] + t2[valid_t2] * A[valid_discriminant]
#         yi_t2 = segments[valid_discriminant, 1] + t2[valid_t2] * B[valid_discriminant]
#     else:
#         return np.empty((0, 4))

#     return np.column_stack((np.column_stack((xi_t1, yi_t1)), np.column_stack((xi_t2, yi_t2))))


def angle_distance_vector(a1, angles):
    # Compute the absolute difference between the angles
    diff = np.abs(a1 - angles)

    # Ensure the shortest distance is used by considering wrap-around

    diff = np.minimum(diff, 2 * math.pi - diff)

    return diff



def get_intersections_vectorized(x, obs_x, r0, r1):
    x_exp = np.expand_dims(x, 1)
    d = np.hypot(obs_x[:, 0] - x_exp[0, :], obs_x[:, 1] - x_exp[1, :])

    # Non-intersecting
    no_intersection = d > 1.6*(r0 + r1)

    # One circle within the other
    one_within_other = d < np.max((r0, r1), axis=0)

    # Coincident circles
    coincident = d == 0

    intersecting = np.logical_not(np.logical_or.reduce((no_intersection, one_within_other, coincident)))
    # Compute intersection points
    if np.any(intersecting):
        intersection_points = get_tangents(x, r0[intersecting]+r1[intersecting], obs_x[intersecting], d[intersecting])
    else:
        intersection_points = None

    output_vec = np.empty((len(d), 4))
    output_vec[:] = None
    output_vec[np.logical_or(one_within_other, coincident), :] = np.inf
    output_vec[intersecting, :] = intersection_points

    return output_vec, d, intersecting
