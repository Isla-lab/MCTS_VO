import math
import numpy as np
from numba import jit
FASTMATH = False

@jit('f8[:, :](f8[:], f8[:], f8[:, :], f8[:])', nopython=True, cache=True, fastmath=FASTMATH)
def get_tangents(robot_state, obs_r, obstacles, d):
    """
    Calculate the tangent points from the robot to each obstacle.

    :param robot_state: The state of the robot, typically containing its position.
    :param obs_r: Radii of the obstacles.
    :param obstacles: Array of obstacle positions.
    :param d: Distance from the robot to each obstacle.
    :return: Array of tangent points.
    """
    # Calculate angles from the robot to each obstacle
    alphas = np.arctan2(robot_state[1] - obstacles[:, 1], robot_state[0] - obstacles[:, 0])
    # Calculate the angles for the tangent points
    phi = np.arccos(obs_r / d)
    # Calculate the tangent points on the obstacles
    P1 = obs_r[:, None] * np.hstack((np.cos(phi)[:, None], np.sin(phi)[:, None]))
    P2 = obs_r[:, None] * np.hstack((np.cos(-phi)[:, None], np.sin(-phi)[:, None]))
    new_P1 = np.empty((phi.shape[0], 2), dtype=np.float64)
    new_P2 = np.empty((phi.shape[0], 2), dtype=np.float64)
    for i in range(len(alphas)):
        alpha = alphas[i]
        # Create rotation matrices for each angle
        matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        # Apply the rotation matrices and translate the points to the robot's position
        new_P1[i] = matrix @ P1[i] + obstacles[i][:2]
        new_P2[i] = matrix @ P2[i] + obstacles[i][:2]

    # Combine the tangent points into a single array and return them
    intersections = np.hstack((new_P1, new_P2))
    return intersections

@jit('f8[:](f8[:], f8[:], f8, f8, f8, f8)', nopython=True, cache=True, fastmath=FASTMATH)
def compute_uniform_towards_goal_jit(
        x: np.ndarray,
        goal: np.ndarray,
        max_angle_change: float,
        min_speed: float,
        max_speed: float,
        amplitude: float,
):
    mean_angle = np.arctan2(goal[1] - x[1], goal[0] - x[0])
    linear_velocity = np.random.uniform(low=min_speed, high=max_speed)
    # Make sure angle is within range of -π to π
    min_angle = x[2] - max_angle_change
    max_angle = x[2] + max_angle_change
    angle = np.random.uniform(low=mean_angle - amplitude, high=mean_angle + amplitude)

    angle = max(min(angle, max_angle), min_angle)
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return np.array([linear_velocity, angle])



@jit('f8[:](f8[:], f8[:], f8)', nopython=True, cache=True, fastmath=FASTMATH)
def robot_dynamics(state_x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Computes the new state of the robot given the current state, control inputs, and time step.
    Parameters:
    x (np.ndarray): The current state of the robot, represented as a numpy array.
    u (np.ndarray): The control inputs, represented as a numpy array.
    dt (float): The time step for the motion prediction.
    Returns:
    np.ndarray: The new state of the robot after applying the control inputs for the given time step.
    """
    x, y, theta, v = state_x
    new_x = np.empty(state_x.shape[0], dtype=np.float64)
    d_theta = (u[1] - theta + np.pi) % (2 * np.pi) - np.pi
    omega = d_theta/dt
    matrix = np.array([[np.cos(theta), 0.0],
                       [np.sin(theta), 0.0],
                       [0.0          , 1.0]])
    deltas = matrix @ np.array([[u[0]],[omega]])
    deltas = deltas * dt
    new_x[:3] = state_x[:3] + deltas[:, 0]
    new_x[2] = (new_x[2] + np.pi) % (2 * np.pi) - np.pi
    new_x[3] = u[0] # v
    return new_x

@jit('b1(f8[:], f8[:, :], f8, f8[:])', cache=True, nopython=True, fastmath=FASTMATH)
def check_coll_vectorized(x, obs, robot_radius, obs_size):
    n = obs.shape[0]
    distances = np.empty(n)
    for i in range(n):
        distances[i] = np.sqrt(np.sum((obs[i] - x)**2))
    
    result = np.any(distances <= robot_radius + obs_size)
    return result


@jit('f8(f8[:], f8[:])', cache=True, nopython=True, fastmath=FASTMATH)
# @cc.export('dist_to_goal', 'f8[2](f8[:], f8[:], f8[:])')
def dist_to_goal(goal: np.ndarray, x: np.ndarray):
    return np.sqrt(np.sum((x-goal)**2))

# @jit('f8[:, :](f4[:], f8[:], f8[:], f8)', nopython=True, cache=True, fastmath=FASTMATH)
# def get_points_from_lidar(dist, angles, robot_pos, heading):
#     angles = angles + heading
#     angles = (angles + np.pi) % (2 * np.pi) - np.pi
#     points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
#     return robot_pos + points

@jit('f8[:, :](f4[:], f8[:], f8[:], f8)', nopython=True, cache=True, fastmath=FASTMATH)
def get_points_from_lidar(dist, angles, robot_pos, heading):
    # angles = angles
    # angles = (angles + np.pi) % (2 * np.pi) - np.pi
    points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
    return points

@jit('f8[:](f8, f8, f8, f8)', nopython=True, cache=True, fastmath=FASTMATH)
def uniform_random(min_speed, max_speed, curr_angle, max_angle_change):
    speed = np.random.uniform(min_speed, max_speed)
    angle = np.random.uniform(curr_angle - max_angle_change, curr_angle + max_angle_change)
    angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-π, π]
    action = np.array([speed, angle], dtype=np.float64)
    return action