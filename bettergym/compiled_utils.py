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


@jit('f8[:](f8, f8, i8)', nopython=True, cache=True, fastmath=FASTMATH)
def _linspace(start, stop, num):
    """
    np.linspace with endpoint=True, laid out the way numpy does it (a scaled
    arange with the last sample pinned to `stop`) so the samples come out
    bit-identical rather than merely close.
    """
    out = np.empty(num, dtype=np.float64)
    if num == 1:
        out[0] = start
        return out
    step = (stop - start) / (num - 1)
    for i in range(num):
        out[i] = start + step * i
    out[num - 1] = stop
    return out


@jit('f8[:, :](f8[:], f8, f8, f8, i8, i8)',
     nopython=True, cache=True, fastmath=FASTMATH)
def discrete_actions(x, max_angle_change, min_speed, max_speed, n_angles, n_vel):
    """
    The unpruned discrete action set: every (velocity, heading) pair the robot can
    reach in one step. Compiled because it is on the path taken whenever velocity
    obstacles prune nothing, which is most tree nodes.

    Reproduces `BetterEnv.get_actions_discrete`, including its two special cases:
    the current heading and a zero velocity are appended when the linear spacing
    misses them (which it does for even n_angles and for n_vel not straddling 0).
    """
    angles = _linspace(x[2] - max_angle_change, x[2] + max_angle_change, n_angles)
    n_a = n_angles
    has_curr = False
    for i in range(n_angles):
        if angles[i] == x[2]:
            has_curr = True
            break
    if not has_curr:
        n_a = n_angles + 1
        tmp = np.empty(n_a, dtype=np.float64)
        tmp[:n_angles] = angles
        tmp[n_angles] = x[2]
        angles = tmp
    for i in range(n_a):
        angles[i] = (angles[i] + np.pi) % (2 * np.pi) - np.pi

    vels = _linspace(min_speed, max_speed, n_vel)
    n_v = n_vel
    has_zero = False
    for i in range(n_vel):
        if vels[i] == 0.0:
            has_zero = True
            break
    if not has_zero:
        n_v = n_vel + 1
        tmp = np.empty(n_v, dtype=np.float64)
        tmp[:n_vel] = vels
        tmp[n_vel] = 0.0
        vels = tmp

    # np.transpose([tile(vels, n_a), repeat(angles, n_v)])
    out = np.empty((n_a * n_v, 2), dtype=np.float64)
    k = 0
    for i in range(n_a):
        for j in range(n_v):
            out[k, 0] = vels[j]
            out[k, 1] = angles[i]
            k += 1
    return out


@jit('f8[:, :](f8[:], f8[:, :], f8[:], f8[:])',
     nopython=True, cache=True, fastmath=FASTMATH)
def vo_forbidden_ranges(robot_state, obstacles, r0, r1):
    """
    Angle ranges the robot must not head into, one or two per obstacle.

    Fuses `get_intersections_vectorized`, `get_tangents` and `get_unsafe_angles`,
    which between them allocated an (n, 4) tangent array, three boolean masks and
    a dozen small temporaries per call, all to end up with a handful of angle
    pairs. Called once per new tree node, so the numpy dispatch overhead of those
    small operations dominated the actual geometry.

    For each obstacle, the enlarged radius r0 + r1 spans an angular sector seen
    from the robot, delimited by the two tangent points. Obstacles further than
    1.6 * (r0 + r1) forbid nothing. If the robot is already inside an enlarged
    obstacle the whole circle is forbidden, which is returned as the single range
    [-pi, pi]: subtracting it leaves nothing, exactly as the old code did by
    marking that obstacle infinite and forbidding the entire reachable span.

    :return: (m, 2) array of [low, high] forbidden ranges, m == 0 if none
    """
    n = obstacles.shape[0]
    out = np.empty((2 * n, 2), dtype=np.float64)
    m = 0
    rx = robot_state[0]
    ry = robot_state[1]

    for i in range(n):
        ox = obstacles[i, 0]
        oy = obstacles[i, 1]
        dx = ox - rx
        dy = oy - ry
        d = np.sqrt(dx * dx + dy * dy)
        r_sum = r0[i] + r1[i]

        if d > 1.6 * r_sum:
            continue
        if d < r_sum or d == 0.0:
            out[0, 0] = -np.pi
            out[0, 1] = np.pi
            return out[:1]

        # Tangent points, i.e. the original rotation of (cos(+-phi), sin(+-phi))
        # by alpha followed by a translation onto the obstacle centre.
        alpha = np.arctan2(ry - oy, rx - ox)
        phi = np.arccos(r_sum / d)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cp = np.cos(phi)
        sp = np.sin(phi)

        a1 = np.arctan2(oy + r_sum * (sa * cp + ca * sp) - ry,
                        ox + r_sum * (ca * cp - sa * sp) - rx)
        a2 = np.arctan2(oy + r_sum * (sa * cp - ca * sp) - ry,
                        ox + r_sum * (ca * cp + sa * sp) - rx)

        if a1 <= a2:
            out[m, 0] = a1
            out[m, 1] = a2
            m += 1
        else:
            # the sector straddles +-pi, so it splits in two
            out[m, 0] = a1
            out[m, 1] = np.pi
            m += 1
            out[m, 0] = -np.pi
            out[m, 1] = a2
            m += 1

    return out[:m]



@jit('f8[:](f8[:], f8[:], i8[:], i8)', nopython=True, cache=True, fastmath=FASTMATH)
def _fit_circle(px, py, idx, n):
    """
    Least-squares circle through n points, by the algebraic (Kasa) method.

    Recentring on the centroid and solving the resulting 2x2 normal equations
    is exact in closed form, so this is one small solve rather than an
    iteration. Returns [cx, cy, r, rms_residual]; r is negative when the points
    are collinear and no circle exists.

    :param idx: indices into px/py of the points to fit, first n used
    """
    out = np.empty(4, dtype=np.float64)

    mx = 0.0
    my = 0.0
    for k in range(n):
        mx += px[idx[k]]
        my += py[idx[k]]
    mx /= n
    my /= n

    suu = 0.0
    svv = 0.0
    suv = 0.0
    suuu = 0.0
    svvv = 0.0
    suvv = 0.0
    svuu = 0.0
    for k in range(n):
        u = px[idx[k]] - mx
        v = py[idx[k]] - my
        uu = u * u
        vv = v * v
        suu += uu
        svv += vv
        suv += u * v
        suuu += uu * u
        svvv += vv * v
        suvv += u * vv
        svuu += v * uu

    det = 2.0 * (suu * svv - suv * suv)
    if det == 0.0:
        out[0] = mx
        out[1] = my
        out[2] = -1.0
        out[3] = np.inf
        return out

    b1 = suuu + suvv
    b2 = svvv + svuu
    uc = (svv * b1 - suv * b2) / det
    vc = (suu * b2 - suv * b1) / det

    r = np.sqrt(uc * uc + vc * vc + (suu + svv) / n)

    cx = uc + mx
    cy = vc + my
    ss = 0.0
    for k in range(n):
        dx = px[idx[k]] - cx
        dy = py[idx[k]] - cy
        e = np.sqrt(dx * dx + dy * dy) - r
        ss += e * e

    out[0] = cx
    out[1] = cy
    out[2] = r
    out[3] = np.sqrt(ss / n)
    return out


@jit('f8[:](f8[:], f8[:], i8, i8, i8, f8, f8)',
     nopython=True, cache=True, fastmath=FASTMATH)
def ransac_circle(px, py, lo, hi, max_trials, residual_threshold, stop_probability):
    """
    RANSAC circle fit over the points px[lo:hi], py[lo:hi].

    The same algorithm skimage.measure.ransac runs, with the same parameters:
    draw a minimal sample of 3 points, fit a circle to it, count inliers within
    residual_threshold, keep the best model, and stop early once the probability
    of having seen an all-inlier sample exceeds stop_probability. The final model
    is refitted on the inliers of the best sample, as skimage does.

    It is reimplemented here purely for speed. skimage's version costs about
    1.3 ms per cluster and that cost is dispatch, not trials: at max_trials=100
    it is 1.33 ms and at 20 it is 1.35 ms. On the 3 to 20 point clusters a LIDAR
    scan produces, the Python and skimage machinery around the fit dwarfs the
    fit itself.

    :return: [cx, cy, r, rms_residual], with r negative if no model was found
    """
    out = np.empty(4, dtype=np.float64)
    n = hi - lo
    if n < 3:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = -1.0
        out[3] = np.inf
        return out

    all_idx = np.empty(n, dtype=np.int64)
    for k in range(n):
        all_idx[k] = lo + k

    sample = np.empty(3, dtype=np.int64)
    best_inliers = np.empty(n, dtype=np.int64)
    inliers = np.empty(n, dtype=np.int64)
    n_best = 0
    best_residual = np.inf
    trials = 0

    while trials < max_trials:
        trials += 1

        # Minimal sample of 3 distinct points
        sample[0] = lo + np.int64(np.random.randint(0, n))
        sample[1] = sample[0]
        while sample[1] == sample[0]:
            sample[1] = lo + np.int64(np.random.randint(0, n))
        sample[2] = sample[0]
        while sample[2] == sample[0] or sample[2] == sample[1]:
            sample[2] = lo + np.int64(np.random.randint(0, n))

        model = _fit_circle(px, py, sample, 3)
        if model[2] < 0.0:
            continue

        n_in = 0
        for k in range(n):
            i = all_idx[k]
            dx = px[i] - model[0]
            dy = py[i] - model[1]
            if abs(np.sqrt(dx * dx + dy * dy) - model[2]) < residual_threshold:
                inliers[n_in] = i
                n_in += 1

        # More inliers wins; equal inliers is broken by the tighter fit, which
        # is how skimage orders its candidates.
        if n_in > n_best or (n_in == n_best and model[3] < best_residual):
            n_best = n_in
            best_residual = model[3]
            for k in range(n_in):
                best_inliers[k] = inliers[k]

            if n_in == n:
                break
            # Probability that no sample so far has been all inliers.
            w = n_in / n
            p_no_outliers = 1.0 - w * w * w
            if p_no_outliers <= 0.0:
                break
            if p_no_outliers >= 1.0:
                continue
            if trials >= np.log(1.0 - stop_probability) / np.log(p_no_outliers):
                break

    if n_best < 3:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = -1.0
        out[3] = np.inf
        return out

    return _fit_circle(px, py, best_inliers, n_best)


@jit('Tuple((f8[:, :], f8[:]))(f8[:], i8[:], f8[:, :], f8, f8, f8, i8, f8, f8, f8, i8, f8, f8)',
     nopython=True, cache=True, fastmath=FASTMATH)
def cluster_and_fit_circles(dist, idx, points, angle_increment, lambda_angle, sigma,
                            min_points, radius_scale, max_radius, residual_threshold,
                            max_trials, stop_probability, max_residual):
    """
    Group a LIDAR scan into obstacles and fit a circle to each, in one call.

    Keeps the two stages of the original pipeline - cluster the returns, then fit
    each cluster with RANSAC - and changes only how each is computed.

    Clustering is the adaptive breakpoint criterion: two consecutive returns
    belong to different objects when they are further apart than

        d * sin(angle_increment) / sin(lambda_angle - angle_increment) + 3 * sigma

    which is the range gap a single surface could produce at the most oblique
    incidence lambda_angle still considered one surface, plus three standard
    deviations of range noise. It is the same density criterion HDBSCAN applies
    in two dimensions, specialised to the fact that a scan is already sorted by
    bearing - so it is a scan over the ranges rather than a mutual-reachability
    tree over the points. sklearn's HDBSCAN costs about 15.5 ms per call
    regardless of how many points it is given (15.5 ms at 36 points, 18.6 at
    240): fixed overhead, and more than the whole control period allows.

    A gap in `idx` also breaks a cluster: get_scan drops invalid returns, which
    would otherwise make the survivors on either side of a dropout look adjacent.

    :param dist: ranges of the valid returns, ordered by bearing
    :param idx: index of each valid return within the raw scan
    :param points: (n, 2) cartesian positions of the same returns, in world frame
    :param min_points: clusters smaller than this are discarded
    :param radius_scale: factor applied to every fitted radius
    :param max_radius: fitted radii above this are discarded, before scaling
    :param max_residual: fits whose RMS residual exceeds this are discarded as
        non-circular; pass np.inf to disable, which reproduces the old pipeline
    :return: (centres (m, 2), radii (m,))
    """
    n = dist.shape[0]
    centres = np.empty((n, 2), dtype=np.float64)
    radii = np.empty(n, dtype=np.float64)
    m = 0

    if n == 0:
        return centres[:0], radii[:0]

    px = np.ascontiguousarray(points[:, 0])
    py = np.ascontiguousarray(points[:, 1])

    sin_inc = np.sin(angle_increment)
    sin_lambda = np.sin(lambda_angle - angle_increment)

    start = 0
    for i in range(1, n + 1):
        split = i == n
        if not split:
            # A dropped return between the two, so they are not neighbours.
            if idx[i] != idx[i - 1] + 1:
                split = True
            else:
                d_max = dist[i - 1] * sin_inc / sin_lambda + 3.0 * sigma
                if abs(dist[i] - dist[i - 1]) > d_max:
                    split = True

        if split:
            if i - start >= min_points:
                model = ransac_circle(px, py, start, i, max_trials,
                                      residual_threshold, stop_probability)
                r = model[2]
                if r > 0.0 and r <= max_radius and model[3] <= max_residual:
                    centres[m, 0] = model[0]
                    centres[m, 1] = model[1]
                    radii[m] = r * radius_scale
                    m += 1
            start = i

    return centres[:m], radii[:m]
