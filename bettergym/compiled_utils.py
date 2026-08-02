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
    # angle = np.random.uniform(low=mean_angle - amplitude, high=mean_angle + amplitude)
    angle = mean_angle
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

# @jit('f8[:](f8[:], f8[:], f8)', nopython=True, cache=True, fastmath=FASTMATH)
# def robot_dynamics(state_x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
#     """
#     Computes the new state of the robot given the current state, control inputs, and time step.
#     Parameters:
#     x (np.ndarray): The current state of the robot, represented as a numpy array.
#     u (np.ndarray): The control inputs, represented as a numpy array.
#     dt (float): The time step for the motion prediction.
#     Returns:
#     np.ndarray: The new state of the robot after applying the control inputs for the given time step.
#     """
#     x, y, theta, v = state_x
#     new_x = np.empty(state_x.shape[0], dtype=np.float64)
#     new_x[0] = x + (u[0] * np.cos(u[1])) * dt
#     new_x[1] = y + (u[0] * np.sin(u[1])) * dt
#     new_x[2] = u[1]
#     new_x[3] = u[0] # v
#     return new_x

@jit('b1(f8[:], f8[:, :], f8, f8[:])', cache=True, nopython=True, fastmath=FASTMATH)
def check_coll_vectorized(x, obs, robot_radius, obs_size):
    n = obs.shape[0]
    distances = np.empty(n)
    for i in range(n):
        distances[i] = np.sqrt(np.sum((obs[i] - x)**2))
    
    result = np.any(distances <= robot_radius)
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
    angles = angles + heading
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
    points = points + robot_pos
    return points

@jit('f8[:](f8, f8, f8, f8)', nopython=True, cache=True, fastmath=FASTMATH)
def uniform_random(min_speed, max_speed, curr_angle, max_angle_change):
    speed = np.random.uniform(min_speed, max_speed)
    angle = np.random.uniform(curr_angle - max_angle_change, curr_angle + max_angle_change)
    angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-π, π]
    action = np.array([speed, angle], dtype=np.float64)
    return action

@jit('f8[:, :](f8[:], f8[:, :], f8)', nopython=True, cache=True, fastmath=FASTMATH)
def predict_obstacles(robot_position, obstacles, dt):
    v = obstacles[:, 3]
    angle = np.arctan2(obstacles[:, 1] - robot_position[1], obstacles[:, 0] - robot_position[0])
    new_obstacles = np.empty_like(obstacles)
    new_obstacles[:, 0] = obstacles[:, 0] + v * np.cos(angle) * dt
    new_obstacles[:, 1] = obstacles[:, 1] + v * np.sin(angle) * dt
    new_obstacles[:, 2] = obstacles[:, 2]
    new_obstacles[:, 3] = obstacles[:, 3]
    return new_obstacles

@jit('f8(f8[:], f8[:], f8[:, :], f8, f8, f8, f8, f8, i8, f8, f8)',
     nopython=True, cache=True, fastmath=FASTMATH)
def fused_rollout(x0, goal, obs_xy, dt, max_angle_change, max_speed,
                  robot_radius, max_eudist, depth, discount, eps):
    """
    Run a whole MCTS rollout in one compiled call and return its discounted return.

    This is a fusion of, per step, `epsilon_uniform_uniform` (the goal-oriented,
    VO-free rollout policy of Algorithm 6), `robot_dynamics`, and the collision,
    goal and reward logic of `Env.step_check_coll` and `Env.reward_grad`. Driving
    those from Python costs about 7 us per step, almost all of it interpreter
    overhead: two `State` objects allocated and discarded per step, plus a
    separate numba dispatch for each of the three small jitted helpers. Fused, a
    200 step rollout takes about 11 us in total.

    Obstacles are held frozen for the whole rollout, as everywhere else in the
    planner: the method assumes no obstacle motion model, only positions and a
    maximum speed.

    Collision checking is selected by what is passed in `obs_xy`, not by a flag:
      - the obstacle positions -> terminate with -100 when an obstacle centre
        comes within `robot_radius`, reproducing `check_coll_vectorized`
        (which likewise ignores the obstacle radii it is handed);
      - an empty (0, 2) array -> no collision termination at all, i.e.
        `step_no_check_coll` semantics, leaving avoidance entirely to VO pruning
        in the tree.
    Both are the same machine code and the empty case simply never enters the
    inner loop, so neither variant pays for the other.

    :param x0: robot state [x, y, theta, v]; v is unused, the dynamics overwrite it
    :param goal: goal position [x, y]
    :param obs_xy: (n, 2) obstacle positions, or (0, 2) to disable collisions
    :param depth: number of steps to simulate (budget minus current depth)
    :param eps: probability of the uniform-random branch of the rollout policy
    :return: the discounted return of the rollout
    """
    x = x0[0]
    y = x0[1]
    theta = x0[2]
    n_obs = obs_xy.shape[0]

    total_reward = 0.0
    gamma = 1.0
    two_pi = 2.0 * np.pi

    for _ in range(depth):
        # --- rollout policy: epsilon_uniform_uniform. min_speed is pinned to 0.0
        # exactly as the Python version does, so a rollout never reverses. The
        # draws are kept in the same order as the originals.
        if np.random.random() <= 1.0 - eps:
            # compute_uniform_towards_goal_jit: head straight at the goal,
            # clipped to what the robot can turn to in one step.
            angle = np.arctan2(goal[1] - y, goal[0] - x)
            velocity = np.random.uniform(0.0, max_speed)
            min_angle = theta - max_angle_change
            max_angle = theta + max_angle_change
            angle = max(min(angle, max_angle), min_angle)
        else:
            # uniform_random
            velocity = np.random.uniform(0.0, max_speed)
            angle = np.random.uniform(theta - max_angle_change,
                                      theta + max_angle_change)
        angle = (angle + np.pi) % two_pi - np.pi

        # --- robot_dynamics: differential drive, heading reached within one dt
        d_theta = (angle - theta + np.pi) % two_pi - np.pi
        x += velocity * np.cos(theta) * dt
        y += velocity * np.sin(theta) * dt
        theta = (theta + d_theta + np.pi) % two_pi - np.pi

        # --- step_check_coll + reward_grad. reward_grad tests the goal before
        # the collision, so a step that does both scores +100; keep that order.
        dist_goal = np.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)
        if dist_goal <= robot_radius:
            total_reward += gamma * 100.0
            return total_reward

        for i in range(n_obs):
            if np.sqrt((obs_xy[i, 0] - x) ** 2 +
                       (obs_xy[i, 1] - y) ** 2) <= robot_radius:
                total_reward += gamma * -100.0
                return total_reward

        # out_boundaries is hard-coded False in step_check_coll, so the wall
        # reward is unreachable here and is deliberately not reproduced.
        total_reward += gamma * (-dist_goal / max_eudist)
        gamma *= discount

    return total_reward
