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

@jit('Tuple((f8, i8))(f8[:], f8[:], f8[:, :], f8, f8, f8, f8, f8, i8, f8, f8)',
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
    :return: (discounted return, steps actually simulated). The step count is
        returned rather than inferred because a rollout that reaches the goal or
        hits an obstacle stops early, and the caller used to record the budget
        instead - which made the reported rollout depth a constant equal to the
        budget, and so incapable of showing anything at all.
    """
    x = x0[0]
    y = x0[1]
    theta = x0[2]
    n_obs = obs_xy.shape[0]

    total_reward = 0.0
    gamma = 1.0
    two_pi = 2.0 * np.pi

    steps = 0
    for _ in range(depth):
        steps += 1
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
            return total_reward, steps

        for i in range(n_obs):
            if np.sqrt((obs_xy[i, 0] - x) ** 2 +
                       (obs_xy[i, 1] - y) ** 2) <= robot_radius:
                total_reward += gamma * -100.0
                return total_reward, steps

        # out_boundaries is hard-coded False in step_check_coll, so the wall
        # reward is unreachable here and is deliberately not reproduced.
        total_reward += gamma * (-dist_goal / max_eudist)
        gamma *= discount

    return total_reward, steps


@jit('Tuple((f8, i8))(f8[:], f8[:], f8[:, :], f8, f8, f8, f8, f8, i8, f8, f8, f8[:, :])',
     nopython=True, cache=True, fastmath=FASTMATH)
def fused_rollout_traj(x0, goal, obs_xy, dt, max_angle_change, max_speed,
                       robot_radius, max_eudist, depth, discount, eps, traj):
    """
    Same as `fused_rollout`, but also records the per-step state history into
    `traj`, an (n, 4) buffer with n >= depth supplied by the caller.

    Kept as a separate compiled function rather than a flag on `fused_rollout`
    so the default, far more common path (no trajectory needed) never pays for
    the buffer allocation or the per-step write. `traj` is a caller-owned,
    reused buffer rather than one allocated here: allocating fresh inside the
    jit function cost about as much as the rest of the rollout combined
    (~20% overhead, almost all of it the allocation itself), against ~2% for
    writing into a buffer the caller already owns. This is safe because the
    caller copies `traj`'s contents out before the next rollout can start.

    :return: (discounted return, steps actually simulated). Only the first
        `steps` rows of `traj` are real; the caller slices to `[:steps]`
        before use, exactly as it already does for `fused_rollout`'s step
        count.
    """
    x = x0[0]
    y = x0[1]
    theta = x0[2]
    n_obs = obs_xy.shape[0]

    total_reward = 0.0
    gamma = 1.0
    two_pi = 2.0 * np.pi

    steps = 0
    for _ in range(depth):
        if np.random.random() <= 1.0 - eps:
            angle = np.arctan2(goal[1] - y, goal[0] - x)
            velocity = np.random.uniform(0.0, max_speed)
            min_angle = theta - max_angle_change
            max_angle = theta + max_angle_change
            angle = max(min(angle, max_angle), min_angle)
        else:
            velocity = np.random.uniform(0.0, max_speed)
            angle = np.random.uniform(theta - max_angle_change,
                                      theta + max_angle_change)
        angle = (angle + np.pi) % two_pi - np.pi

        d_theta = (angle - theta + np.pi) % two_pi - np.pi
        x += velocity * np.cos(theta) * dt
        y += velocity * np.sin(theta) * dt
        theta = (theta + d_theta + np.pi) % two_pi - np.pi

        traj[steps, 0] = x
        traj[steps, 1] = y
        traj[steps, 2] = theta
        traj[steps, 3] = velocity
        steps += 1

        dist_goal = np.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)
        if dist_goal <= robot_radius:
            total_reward += gamma * 100.0
            return total_reward, steps

        for i in range(n_obs):
            if np.sqrt((obs_xy[i, 0] - x) ** 2 +
                       (obs_xy[i, 1] - y) ** 2) <= robot_radius:
                total_reward += gamma * -100.0
                return total_reward, steps

        total_reward += gamma * (-dist_goal / max_eudist)
        gamma *= discount

    return total_reward, steps


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


@jit('f8[:, :](f8[:, :])', nopython=True, cache=True, fastmath=FASTMATH)
def unique_rows(actions):
    """
    np.unique(actions, axis=0) for the (n, 2) action set, at a twentieth of the
    cost - it was 18 us, half of what a pruned node spent after the rest of the
    path was compiled.

    np.unique compares numerically rather than bitwise (it folds -0.0 into 0.0)
    and returns rows in lexicographic order, so a stable sort on (v, angle)
    keeping the first of each equal group reproduces it. n is at most a few
    dozen, hence insertion sort.

    The one divergence is which sign of zero survives when a column holds both
    +0.0 and -0.0: over 20000 adversarial arrays the shapes and values always
    matched and only signbit differed, and no action set out of 11500 sampled
    states contained a negative zero at all. Velocities come from the config and
    angles from a linspace over the safe ranges; -0.0 would need a range endpoint
    to be exactly it.

    Most of the duplicates are structural: the velocity interval of a pruned
    range is a single point, so linspace hands back n_vel identical copies of it
    and every angle appears n_vel times.
    """
    n = actions.shape[0]
    if n == 0:
        return actions

    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        order[i] = i
    for i in range(1, n):
        cur = order[i]
        cv = actions[cur, 0]
        ca = actions[cur, 1]
        j = i - 1
        while j >= 0:
            pv = actions[order[j], 0]
            pa = actions[order[j], 1]
            if pv > cv or (pv == cv and pa > ca):
                order[j + 1] = order[j]
                j -= 1
            else:
                break
        order[j + 1] = cur

    out = np.empty((n, 2), dtype=np.float64)
    m = 0
    for i in range(n):
        r = order[i]
        if m > 0 and actions[r, 0] == out[m - 1, 0] and actions[r, 1] == out[m - 1, 1]:
            continue
        out[m, 0] = actions[r, 0]
        out[m, 1] = actions[r, 1]
        m += 1
    return out[:m]


@jit('Tuple((f8[:, :], b1))(f8[:], f8[:, :], f8[:], f8, f8, f8, f8, f8, b1)',
     nopython=True, cache=True, fastmath=FASTMATH)
def vo_safe_ranges(robot_state, obstacles, obs_rad, dt, robot_radius, vmax,
                   think_margin, max_angle_change, legacy):
    """
    Heading ranges left after subtracting every velocity obstacle, in one call.

    Fuses get_radii, vo_forbidden_ranges, get_robot_angles and
    compute_ranges_difference, which together were 16 us per pruning pass and
    ran twice per node - against 0.7 us for the geometry itself. All four are a
    handful of arithmetic on at most a few intervals; the cost was numpy
    dispatch and Python list building, not the work.

    :return: (safe_ranges, any_vo). any_vo is False when no obstacle forbade
        anything, in which case safe_ranges is the full reachable span. An empty
        safe_ranges with any_vo True means no heading is safe.
    """
    two_pi = 2.0 * np.pi

    # Reachable span, split when it straddles +-pi (get_robot_angles).
    lo = (robot_state[2] - max_angle_change + np.pi) % two_pi - np.pi
    hi = (robot_state[2] + max_angle_change + np.pi) % two_pi - np.pi
    base = np.empty((2, 2), dtype=np.float64)
    if lo > hi:
        base[0, 0] = lo
        base[0, 1] = np.pi
        base[1, 0] = -np.pi
        base[1, 1] = hi
        n_base = 2
    else:
        base[0, 0] = lo
        base[0, 1] = hi
        n_base = 1

    # Forbidden sectors, one or two per obstacle (vo_forbidden_ranges).
    n = obstacles.shape[0]
    forb = np.empty((2 * n, 2), dtype=np.float64)
    nf = 0
    rx = robot_state[0]
    ry = robot_state[1]
    trapped = False

    for i in range(n):
        r_ball = obstacles[i, 3] * (dt + think_margin) + obs_rad[i] + robot_radius
        r_reach = r_ball + vmax * dt
        if legacy:
            # ball = r0 + r1, reach = 1.6 * (r0 + r1)
            r_ball = r_reach
            r_reach = 1.6 * r_ball

        ox = obstacles[i, 0]
        oy = obstacles[i, 1]
        dx = ox - rx
        dy = oy - ry
        d = np.sqrt(dx * dx + dy * dy)

        if d > r_reach:
            continue
        if d < r_ball or d == 0.0:
            trapped = True
            break

        alpha = np.arctan2(ry - oy, rx - ox)
        phi = np.arccos(r_ball / d)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cp = np.cos(phi)
        sp = np.sin(phi)

        a1 = np.arctan2(oy + r_ball * (sa * cp + ca * sp) - ry,
                        ox + r_ball * (ca * cp - sa * sp) - rx)
        a2 = np.arctan2(oy + r_ball * (sa * cp - ca * sp) - ry,
                        ox + r_ball * (ca * cp + sa * sp) - rx)

        if a1 <= a2:
            forb[nf, 0] = a1
            forb[nf, 1] = a2
            nf += 1
        else:
            forb[nf, 0] = a1
            forb[nf, 1] = np.pi
            nf += 1
            forb[nf, 0] = -np.pi
            forb[nf, 1] = a2
            nf += 1

    if trapped:
        return np.empty((0, 2), dtype=np.float64), True
    if nf == 0:
        return base[:n_base].copy(), False

    # Drop the degenerate ones, as the numpy version's hi > lo filter did.
    keep = np.empty((nf, 2), dtype=np.float64)
    nk = 0
    for i in range(nf):
        if forb[i, 1] > forb[i, 0]:
            keep[nk, 0] = forb[i, 0]
            keep[nk, 1] = forb[i, 1]
            nk += 1
    if nk == 0:
        return base[:n_base].copy(), True

    # Stable sort by lower bound; nk is at most 2 * n_obstacles.
    for i in range(1, nk):
        klo = keep[i, 0]
        khi = keep[i, 1]
        j = i - 1
        while j >= 0 and keep[j, 0] > klo:
            keep[j + 1, 0] = keep[j, 0]
            keep[j + 1, 1] = keep[j, 1]
            j -= 1
        keep[j + 1, 0] = klo
        keep[j + 1, 1] = khi

    # Union, so overlapping obstacles are subtracted once.
    merged = np.empty((nk, 2), dtype=np.float64)
    nm = 1
    merged[0, 0] = keep[0, 0]
    merged[0, 1] = keep[0, 1]
    for i in range(1, nk):
        if keep[i, 0] <= merged[nm - 1, 1]:
            if keep[i, 1] > merged[nm - 1, 1]:
                merged[nm - 1, 1] = keep[i, 1]
        else:
            merged[nm, 0] = keep[i, 0]
            merged[nm, 1] = keep[i, 1]
            nm += 1

    out = np.empty((n_base * (nm + 1), 2), dtype=np.float64)
    no = 0
    for b in range(n_base):
        b_lo = base[b, 0]
        b_hi = base[b, 1]
        if not (b_hi > b_lo):
            continue
        cursor = b_lo
        for f in range(nm):
            f_lo = merged[f, 0]
            f_hi = merged[f, 1]
            if f_hi <= cursor:
                continue
            if f_lo >= b_hi:
                break
            if f_lo > cursor:
                # The numpy version deduplicated emitted pairs exactly; nm is
                # small enough that a linear scan is the same thing.
                dup = False
                for k in range(no):
                    if out[k, 0] == cursor and out[k, 1] == f_lo:
                        dup = True
                        break
                if not dup:
                    out[no, 0] = cursor
                    out[no, 1] = f_lo
                    no += 1
            cursor = f_hi
            if cursor >= b_hi:
                break
        if cursor < b_hi:
            dup = False
            for k in range(no):
                if out[k, 0] == cursor and out[k, 1] == b_hi:
                    dup = True
                    break
            if not dup:
                out[no, 0] = cursor
                out[no, 1] = b_hi
                no += 1

    return out[:no], True


@jit('i8[:](f8[:, :], i8, b1)', nopython=True, cache=True, fastmath=FASTMATH)
def _range_sample_counts(space, n_sample, use_width):
    """
    Split `n_sample` samples across the intervals of `space`, proportionally.

    Reproduces BetterEnv.get_discrete_space exactly, including the 1e-6 floor
    that keeps a degenerate interval from taking zero samples and the
    floor-on-even / ceil-on-odd rounding. `use_width` selects the width metric
    over the norm; see get_discrete_space for which results were made with which.
    """
    m = space.shape[0]
    sizes = np.empty(m, dtype=np.float64)
    for i in range(m):
        if use_width:
            sizes[i] = abs(space[i, 1] - space[i, 0]) + 1e-6
        else:
            sizes[i] = np.sqrt(space[i, 0] * space[i, 0]
                               + space[i, 1] * space[i, 1]) + 1e-6

    total = 0.0
    for i in range(m):
        total += sizes[i]

    out = np.empty(m, dtype=np.int64)
    for i in range(m):
        d = (sizes[i] / total) * n_sample
        out[i] = np.int64(np.floor(d) if i % 2 == 0 else np.ceil(d))
    return out


@jit('f8[:, :](f8[:, :], f8[:, :], i8, i8, b1)',
     nopython=True, cache=True, fastmath=FASTMATH)
def discrete_actions_multi_range(aspace, vspace, n_angles, n_vel, use_width):
    """
    The pruned action set: the (velocity, heading) grid over each safe range.

    Compiled equivalent of BetterEnv.get_discrete_actions_multi_range, which at
    36 us a call was 40% of a VO-pruned node - twenty times the cost of the
    velocity-obstacle geometry it exists to consume. The work is two proportional
    splits and an outer product over one to three intervals; all of it was numpy
    dispatch on arrays of length 4 and 6.

    Row order matches the tile/repeat it replaces: velocities vary fastest.
    """
    m = aspace.shape[0]
    n_a = _range_sample_counts(aspace, n_angles, use_width)
    n_v = _range_sample_counts(vspace, n_vel, use_width)

    total = 0
    for i in range(m):
        total += n_a[i] * n_v[i]

    out = np.empty((total, 2), dtype=np.float64)
    k = 0
    for i in range(m):
        na = n_a[i]
        nv = n_v[i]
        if na <= 0 or nv <= 0:
            continue
        angles = _linspace(aspace[i, 0], aspace[i, 1], na)
        vels = _linspace(vspace[i, 0], vspace[i, 1], nv)
        for ai in range(na):
            for vi in range(nv):
                out[k, 0] = vels[vi]
                out[k, 1] = angles[ai]
                k += 1
    return out[:k]


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


@jit('b1(f8[:], f8[:, :], f8[:], f8, f8, f8, f8, b1)',
     nopython=True, cache=True, fastmath=FASTMATH)
def any_robot_inside_ball(robot_state, obstacles, obs_rad, dt, robot_radius,
                          vmax, think_margin, legacy):
    """
    Algorithm 4 line 10, on its own: is the robot centre inside some B(p_i, r1)?

    Same radius as `get_radii`, recomputed here rather than taking r1 as an
    argument. Building that array first costs about 8 us of numpy dispatch per
    call - more than this whole test, and more than the tangent geometry it is
    meant to let the caller skip.
    """
    for i in range(obstacles.shape[0]):
        r_ball = obstacles[i, 3] * (dt + think_margin) + obs_rad[i] + robot_radius
        if legacy:
            r_ball += vmax * dt
        dx = obstacles[i, 0] - robot_state[0]
        dy = obstacles[i, 1] - robot_state[1]
        if dx * dx + dy * dy < r_ball * r_ball:
            return True
    return False


@jit('Tuple((b1, f8))(f8[:], f8[:, :], f8[:], f8, f8, f8, f8, b1)',
     nopython=True, cache=True, fastmath=FASTMATH)
def trapped_escape_heading(robot_state, obstacles, obs_rad, dt, robot_radius,
                            vmax, think_margin, legacy):
    """
    Single escape heading out of every B(p_i, r1) the robot centre is inside,
    as a weighted vector sum of "away from obstacle i" - the ORIGINAL design
    (`--trapped-escape blended`), kept only for reproducing/comparing against
    that A/B arm; `--trapped-escape per-obstacle[-no-stop]` (below,
    `trapped_escape_headings`, plural) replaced it as the default because
    averaging "away from A" and "away from B" can point straight at a third,
    non-trapping obstacle that neither term accounts for - measured as an 80%
    voluntary-collision rate at gamma=0.25 on intention_complex.

    Weight is penetration depth (r_ball_i - d_i): always > 0 for a trapping
    obstacle, naturally maximal at d_i == 0 - no 1/clearance divide-by-zero
    guard needed for the weight. The *direction* is still undefined at
    d_i == 0, so it falls back to the reverse of the current heading there
    (unlike the per-obstacle version, which excludes that obstacle instead -
    a single blended direction cannot skip a contributor without changing
    what "blended" means, so this earlier design keeps its original
    fallback rather than adopting the newer function's fix).

    If the weighted sum cancels out (near zero - e.g. two obstacles
    straddling the robot on opposite sides at equal penetration), fall back
    to heading + pi (back away from where the robot is currently facing) so
    this never returns NaN.

    :return: (any_trapped, escape_heading). escape_heading is meaningless
        when any_trapped is False.
    """
    vx = 0.0
    vy = 0.0
    any_trapped = False
    heading = robot_state[2]
    for i in range(obstacles.shape[0]):
        r_ball = obstacles[i, 3] * (dt + think_margin) + obs_rad[i] + robot_radius
        if legacy:
            r_ball += vmax * dt
        dx = robot_state[0] - obstacles[i, 0]
        dy = robot_state[1] - obstacles[i, 1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < r_ball:
            any_trapped = True
            w = r_ball - d
            if d < 1e-9:
                ux = -math.cos(heading)
                uy = -math.sin(heading)
            else:
                ux = dx / d
                uy = dy / d
            vx += w * ux
            vy += w * uy
    if not any_trapped:
        return False, 0.0
    norm = math.sqrt(vx * vx + vy * vy)
    if norm < 1e-9:
        return True, (heading + math.pi + math.pi) % (2 * math.pi) - math.pi
    return True, math.atan2(vy, vx)


@jit('Tuple((b1[:], f8[:]))(f8[:], f8[:, :], f8[:], f8, f8, f8, f8, b1)',
     nopython=True, cache=True, fastmath=FASTMATH)
def trapped_escape_headings(robot_state, obstacles, obs_rad, dt, robot_radius,
                             vmax, think_margin, legacy):
    """
    One escape heading per trapping obstacle (Algorithm 4's trapped case:
    the robot centre inside some B(p_i, r1)), not a single blended direction.
    A blended (weighted-vector-sum) direction was tried first and dropped:
    averaging "away from A" and "away from B" can point straight at a THIRD,
    non-trapping obstacle that neither term accounts for - observed directly
    as an 80% voluntary-collision rate at gamma=0.25 on intention_complex.
    Handing the tree one clean "away from A" and one clean "away from B"
    candidate instead lets it discard whichever one is actually bad via
    rollout, which a single averaged compromise never allows.

    d_i == 0 (robot centre exactly on an obstacle centre) is not a near-miss
    to route an escape heading for - it is already the physical collision
    itself (the real robot_radius+obs_rad collision test fires long before
    centre-to-centre distance reaches exactly 0). That obstacle contributes
    no candidate at all here, same as a non-trapping one; if every trapping
    obstacle is in this state the caller sees zero candidates and falls back
    to the old forced stop, rather than a fabricated recovery direction.

    :return: (is_trapping, heading) - both length obstacles.shape[0], one
        row per input obstacle. is_trapping[i] is True only when obstacle i
        traps the robot AND is not the d_i == 0 case; heading[i] is
        meaningless where is_trapping[i] is False.
    """
    n = obstacles.shape[0]
    is_trapping = np.zeros(n, dtype=np.bool_)
    heading = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r_ball = obstacles[i, 3] * (dt + think_margin) + obs_rad[i] + robot_radius
        if legacy:
            r_ball += vmax * dt
        dx = robot_state[0] - obstacles[i, 0]
        dy = robot_state[1] - obstacles[i, 1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < r_ball and d >= 1e-9:
            is_trapping[i] = True
            heading[i] = math.atan2(dy, dx)
    return is_trapping, heading


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

    Three cases per obstacle, following Algorithm 4:

        d > r0 + r1     the robot cannot reach the ball within one step, so
                        nothing is forbidden
        r1 <= d <= r0+r1  the tangents to B(p_i, r1) delimit the sector that
                        reaches it; that sector is forbidden
        d < r1          the robot is already inside the ball, so the whole
                        circle is forbidden, returned as the single range
                        [-pi, pi]: subtracting it leaves nothing, exactly as the
                        old code did by marking that obstacle infinite and
                        forbidding the entire reachable span

    Note that r0 is a distance travelled in one step, not a body radius. It sets
    how far out an obstacle can still matter, and so belongs in the first test
    only; the tangents are taken to r1 alone. Building them on r0 + r1 instead
    pushes the trapped test out to d < r0 + r1, which leaves the cone band
    nothing but the single point d == r0 + r1 - and that degeneracy is what the
    former 1.6 * (r0 + r1) cutoff existed to paper over.

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
        # Reach: how far the obstacle can still matter. Radius: the ball the
        # tangents are taken to.
        r_reach = r0[i] + r1[i]
        r_ball = r1[i]

        if d > r_reach:
            continue
        if d < r_ball or d == 0.0:
            out[0, 0] = -np.pi
            out[0, 1] = np.pi
            return out[:1]

        # Tangent points, i.e. the original rotation of (cos(+-phi), sin(+-phi))
        # by alpha followed by a translation onto the obstacle centre.
        alpha = np.arctan2(ry - oy, rx - ox)
        phi = np.arccos(r_ball / d)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cp = np.cos(phi)
        sp = np.sin(phi)

        a1 = np.arctan2(oy + r_ball * (sa * cp + ca * sp) - ry,
                        ox + r_ball * (ca * cp - sa * sp) - rx)
        a2 = np.arctan2(oy + r_ball * (sa * cp - ca * sp) - ry,
                        ox + r_ball * (ca * cp + sa * sp) - rx)

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
