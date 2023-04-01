import math

import numpy as np
from numba import njit


# @njit
def sample_centered_robot_arena(center: np.ndarray):
    return np.random.multivariate_normal(
        center,
        np.diag([0.3 / 10, 1.9 / 10])
    )


@njit
def velocity_obstacle_nearest(x, obs, dt, ROBOT_RADIUS, OBS_RADIUS):
    """
    Compute Forbidden Angular Velocities
    :param x: current robot state
    :param obs: obstacle positions
    :param dt:
    :param ROBOT_RADIUS: radius of the robot
    :param OBS_RADIUS: radius of the obstacle
    :return: angular velocity that would result in a collision
    """
    # compute euclidian distance from each obstacle
    distances = np.array([math.hypot(ob[0] - x[0], ob[1] - x[1]) for ob in obs])
    dist_to_ob_idx = np.argmin(distances)
    dist_to_ob = distances[dist_to_ob_idx]
    ob = obs[dist_to_ob_idx]
    # compute angle between the middle of the cone and one of the cone walls
    beta = math.asin((ROBOT_RADIUS + OBS_RADIUS) / dist_to_ob)
    # convert angle from [-pi/2, pi/2] to [-pi, pi]
    beta = math.atan2(math.sin(beta), math.cos(beta))
    # angle between plane and the middle of the cone
    theta = math.atan2(ob[1] - x[1], ob[0] - x[0])
    # compute right angle and left angle then subtract current angle
    forbidden_angles = [theta - beta - x[2], theta + beta - x[2]]
    forbidden_ang_vel = [a / dt for a in forbidden_angles]
    return forbidden_ang_vel
