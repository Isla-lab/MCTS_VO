import math
import random
from typing import Any
import numpy as np

try:
    from MCTS_VO.bettergym.agents.planner import Planner
    from MCTS_VO.bettergym.agents.utils.utils import get_robot_angles, compute_uniform_towards_goal_jit
    from MCTS_VO.mcts_utils import get_intersections_vectorized, angle_distance_vector
    from MCTS_VO.bettergym.compiled_utils import uniform_random, vo_forbidden_ranges, any_robot_inside_ball, vo_safe_ranges, trapped_escape_headings
except ModuleNotFoundError:
    from bettergym.agents.planner import Planner
    from bettergym.agents.utils.utils import get_robot_angles, compute_uniform_towards_goal_jit
    from mcts_utils import get_intersections_vectorized, angle_distance_vector
    from bettergym.compiled_utils import uniform_random, vo_forbidden_ranges, any_robot_inside_ball, vo_safe_ranges, trapped_escape_headings

# def print_to_file(param):
#     # with open("OUTPUT.txt", "a") as f:
#     #     f.write(str(param))
#     pass

# Set by loopHandler_copy.py's --vo-geometry, before any planning happens.
# False is the corrected geometry: tangents to the ball of radius r1, obstacles
# beyond r0 + r1 ignored, trapped below r1 - Algorithm 4 as written. True
# restores the geometry used up to and including the 180-run campaign, so the
# two can be compared on identical scenes. See get_radii for how.
LEGACY_VO = False


def set_legacy_vo(enabled: bool) -> None:
    """Select the VO geometry. Must be called before the first planning step."""
    global LEGACY_VO
    LEGACY_VO = enabled


# Set by loopHandler_copy.py's --trapped-escape, before any planning happens.
# False (default) is today's Algorithm 4 behaviour: trapped forces a full
# stop (every heading, v=0). True replaces that with a computed escape
# action - see compute_trapped_escape. Orthogonal to LEGACY_VO: that knob
# picks which geometry defines "trapped", this one picks what to do once
# trapped, under either geometry.
TRAPPED_ESCAPE = False


def set_trapped_escape(enabled: bool) -> None:
    """Select the trapped fallback. Must be called before the first planning step."""
    global TRAPPED_ESCAPE
    TRAPPED_ESCAPE = enabled


def _signed_angle_diff(a, b):
    """Shortest signed difference a-b, wrapped to [-pi, pi]. angle_distance
    (below) returns an unsigned difference - this project has no existing
    signed version, needed here to pick a turn direction, not just a size."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def compute_trapped_escape(x, circle_obs_x, circle_obs_rad, config):
    """
    Forward/reverse escape candidates, one PAIR per trapping obstacle - not
    one direction blended across all of them. A blended (weighted-sum)
    direction was tried first and dropped: it can point straight at a third,
    non-trapping obstacle that neither term accounts for (measured as an 80%
    voluntary-collision rate at gamma=0.25 on intention_complex). Giving the
    tree one clean "away from A" and one clean "away from B" candidate lets
    it discard whichever is actually bad via rollout, which a single
    averaged compromise never allows.

    Two distinct speed quantities matter here, kept separate on purpose:
    - the ball-radius test itself (is the robot trapped at all) uses the
      symmetric vmax_for_ball = min(config.max_speed, abs(config.min_speed)),
      exactly the reasoning `robot_trapped` already uses (trapped at the
      smaller of the two implies trapped at the larger).
    - the candidate escape speeds use the real, asymmetric config.max_speed /
      config.min_speed - the robot's true forward/backward limits, not the
      min'd value.

    :return: [] when the escape heuristic is disabled (TRAPPED_ESCAPE is
        False) or the robot is not trapped by any obstacle with d > 0 (an
        obstacle at d == 0 is already a physical collision, not a near-miss
        to route an escape heading for - see trapped_escape_headings).
        Otherwise a list of (heading_fwd, heading_rev, delta_fwd) tuples, one
        per trapping obstacle - delta_fwd is the signed turn (from the
        current heading) the forward candidate needs; callers use it to
        tie-break which candidate is "better" (smaller turn) when they must
        pick just one. Checking TRAPPED_ESCAPE here rather than at each call
        site keeps every caller correct without importing the global itself -
        `from vo import TRAPPED_ESCAPE` would capture its value at import
        time, not a live reference, and set_trapped_escape() would then
        silently fail to affect it anywhere but this module.
    """
    if not TRAPPED_ESCAPE:
        return []
    vmax_for_ball = min(config.max_speed, abs(config.min_speed))
    is_trapping, escape_headings = trapped_escape_headings(
        x, circle_obs_x, circle_obs_rad, config.dt, config.robot_radius,
        vmax_for_ball, config.think_margin, LEGACY_VO,
    )
    if not is_trapping.any():
        return []

    heading = x[2]
    mac = config.max_angle_change
    candidates = []
    for escape_heading, trapping in zip(escape_headings, is_trapping):
        if not trapping:
            continue

        delta_fwd = _signed_angle_diff(escape_heading, heading)
        heading_fwd = heading + max(-mac, min(mac, delta_fwd))
        heading_fwd = (heading_fwd + math.pi) % (2 * math.pi) - math.pi

        reverse_target = (escape_heading + math.pi + math.pi) % (2 * math.pi) - math.pi
        delta_rev = _signed_angle_diff(reverse_target, heading)
        heading_rev = heading + max(-mac, min(mac, delta_rev))
        heading_rev = (heading_rev + math.pi) % (2 * math.pi) - math.pi

        candidates.append((heading_fwd, heading_rev, delta_fwd))
    return candidates


def get_radii(circle_obs_x, circle_obs_rad, dt, robot_radius, vmax, think_margin=0.1):
    """
    Radii of the two circles whose tangents delimit a velocity obstacle.

    r1 covers how far an obstacle can travel while the robot is not moving under
    a fresh command, i.e. the control step plus the time spent sensing and
    planning. `think_margin` used to be the literal 0.1 here, which matched the
    roughly 95 ms the loop then took to think; it is a parameter so that it
    follows the compute time down instead of staying pinned to it. The default
    preserves the old behaviour for callers that do not set it.
    """
    r1 = circle_obs_x[:, 3] * (dt + think_margin) + circle_obs_rad + robot_radius
    r0 = np.full_like(r1, vmax * dt)

    if LEGACY_VO:
        # Reproduce the pre-correction geometry without touching the consumers.
        # They take tangents to a ball of radius r1 and ignore obstacles beyond
        # r0 + r1; the old code took tangents to r0 + r1 and ignored beyond
        # 1.6 * (r0 + r1). Both are recovered exactly by rescaling the inputs:
        #   ball  = r1' = r0 + r1
        #   reach = r0' + r1' = 0.6 * (r0 + r1) + (r0 + r1) = 1.6 * (r0 + r1)
        r_sum = r0 + r1
        return r_sum, 0.6 * r_sum

    return r1, r0


def robot_trapped(x, circle_obs_x, circle_obs_rad, config):
    """
    True when the robot centre is inside some obstacle ball B(p_i, r1).

    Algorithm 4 line 10: that case sets A_c to the empty set and breaks, so the
    answer is already known - no heading is safe and V_c = {0}. The tangents of
    every other obstacle are then computed only to be subtracted from a span
    that is empty regardless.

    Speed does not enter r1 under the paper geometry, so one test covers both
    the forward and the reverse pruning. LEGACY_VO folds r0 into the ball, and
    r0 = vmax * dt does grow with the speed, so there the test is taken at the
    smaller of the two: trapped at the smaller vmax implies trapped at the
    larger, which is what makes skipping both passes sound.
    """
    return any_robot_inside_ball(
        x, circle_obs_x, circle_obs_rad, config.dt, config.robot_radius,
        min(config.max_speed, abs(config.min_speed)), config.think_margin,
        LEGACY_VO,
    )


def uniform_towards_goal_vo(node: Any, planner: Planner, std_angle_rollout: float):
    config = planner.environment.gym_env.config
    x = node.state.x

    if len(node.state.obstacles) == 0:
        return compute_uniform_towards_goal_jit(
            x=x,
            goal=node.state.goal,
            max_angle_change=config.max_angle_change,
            amplitude=std_angle_rollout,
            min_speed=0.0,
            max_speed=config.max_speed,
        )

    # Extract robot information
    dt = config.dt
    ROBOT_RADIUS = config.robot_radius
    VMAX = config.max_speed

    # Extract obstacle information
    circle_obs_x, circle_obs_rad = node.state.obstacles
    intersection_points = np.empty((0, 4), dtype=np.float64)

    if len(circle_obs_x) != 0:
        # Calculate radii
        r1, r0 = get_radii(circle_obs_x, circle_obs_rad, dt, ROBOT_RADIUS, VMAX,
                           think_margin=config.think_margin)
        # Calculate intersection points
        intersection_points, dist, mask = get_intersections_vectorized(x, circle_obs_x, r0, r1)


    # CASE 1 no obs intersection and no wall intersection
    if np.isnan(intersection_points).all():
        return compute_uniform_towards_goal_jit(
            x=x,
            goal=node.state.goal,
            max_angle_change=config.max_angle_change,
            amplitude=std_angle_rollout,
            min_speed=0.0,
            max_speed=config.max_speed,
        )
    # CASE 2 only wall intersection
    # CASE 3 only obs intersection
    # CASE 4 both wall and obs intersection
    else:
        angle_space, velocity_space, flip = new_get_spaces([None, (circle_obs_x, circle_obs_rad), None], x, config, intersection_points, wall_angles=None)
        mean_angle = np.arctan2(node.state.goal[1] - x[1], node.state.goal[0] - x[0])
        in_space = False
        for a_space in angle_space:
            if a_space[0] <= mean_angle <= a_space[1]:
                in_space = True
                break
        
        if not in_space:
            angle_space = np.array(angle_space)
            angles = np.random.uniform(low=mean_angle - std_angle_rollout, high=mean_angle + std_angle_rollout, size=20)
            if flip:
                angles_copy = (angles + math.pi + math.pi) % (2 * math.pi) - math.pi
                in_range = (angle_space[:, 0] <= angles_copy[:, np.newaxis]) & (angle_space[:, 1] >= angles_copy[:, np.newaxis])
            else:
                in_range = (angle_space[:, 0] <= angles[:, np.newaxis]) & (angle_space[:, 1] >= angles[:, np.newaxis])
            if not np.any(in_range):
                action = sample_multiple_spaces(center=None, a_space=angle_space, v_space=velocity_space, number=1)[0]
                if action[0] < 0 and flip:
                    action[1] = action[1] + math.pi
                    action[1] = (action[1] + math.pi) % (2 * math.pi) - math.pi
                return action
            else:
                idx_angles, idx_ranges = np.where(in_range)
                idx = random.randint(0, len(idx_angles) - 1)
                angle = angles[idx_angles[idx]]
                velocity = np.random.uniform(low=velocity_space[idx_ranges[idx]][0], high=velocity_space[idx_ranges[idx]][1])
        else:
            velocity = np.random.uniform(low=velocity_space[0][0], high=velocity_space[0][1])
            angle = mean_angle
            
        return np.array([velocity, angle])


def sample_multiple_spaces(center, a_space, number, v_space):
    lengths_aspace = np.linalg.norm(a_space, axis=1)
    percentages_aspace = np.cumsum(lengths_aspace / np.sum(lengths_aspace))
    pct = random.random()
    idx_space = np.flatnonzero(pct <= percentages_aspace)[0]
    return np.vstack(
        [
            np.random.uniform(low=v_space[idx_space][0], high=v_space[idx_space][1], size=number),
            np.random.uniform(low=a_space[idx_space][0], high=a_space[idx_space][1], size=number),
        ]
    ).T


def sample_single_space(center, a_space, number, v_space):
    return np.vstack(
        [
            np.random.uniform(low=v_space[0], high=v_space[1], size=number),
            np.random.uniform(low=a_space[0], high=a_space[1], size=number),
        ]
    ).T


def sample(center, a_space, v_space, number):
    if len(a_space) == 1:
        return sample_single_space(center, a_space[0], number, v_space)
    else:
        return sample_multiple_spaces(center, a_space, number, v_space)


def angle_distance(angle1, angle2):
    # Compute the absolute difference between the angles
    diff = abs(angle1 - angle2)

    # Ensure the shortest distance is used by considering wrap-around
    diff = min(diff, 2 * math.pi - diff)

    return diff


def compute_ranges_difference(robot_angles, forbidden_ranges):
    """
    Subtract the forbidden angle ranges from the robot's reachable angle ranges.

    Straight interval subtraction on sorted arrays, which replaces two
    `intervaltree.IntervalTree` allocations per call. Both inputs hold a handful
    of intervals at most - at most 2 reachable ranges, at most 2 per obstacle
    forbidden - so building a tree to chop them cost far more than the
    subtraction itself, and this runs once per new tree node.

    Semantics match the tree version: half-open intervals, degenerate
    (zero-length) intervals dropped from both sides, the forbidden set treated
    as a union, and identical output fragments emitted once. That last point
    only bites when the *base* ranges overlap, which `get_robot_angles` never
    produces - it returns either one span or two split at +-pi - but the tree
    returned a set and so collapsed duplicates, and matching it costs nothing on
    inputs this small. Verified identical on 4000 randomized inputs including
    degenerate and overlapping ranges.

    One deliberate difference: the tree returned `all_intervals`, a *set*, so the
    order of the safe ranges was arbitrary. That order is not inert downstream -
    `BetterEnv.get_discrete_space` floors the sample count of even-indexed ranges
    and ceils the odd-indexed ones, so how many actions came out of each range
    depended on the iteration order of a set. These are returned sorted by lower
    bound instead, which makes that allocation deterministic. The safe angle
    space itself is unchanged; over 500 randomized states the resulting action
    set differed in 3, always by which range got the extra sample.
    """
    base = np.asarray(robot_angles, dtype=np.float64).reshape(-1, 2)
    base = base[base[:, 1] > base[:, 0]]
    if len(base) == 0:
        return []

    forbidden = np.asarray(forbidden_ranges, dtype=np.float64).reshape(-1, 2)
    forbidden = forbidden[forbidden[:, 1] > forbidden[:, 0]]
    if len(forbidden) == 0:
        return base.tolist()

    # Union of the forbidden ranges, so overlapping obstacles are subtracted once
    forbidden = forbidden[np.argsort(forbidden[:, 0], kind="stable")]
    merged = [list(forbidden[0])]
    for lo, hi in forbidden[1:]:
        if lo <= merged[-1][1]:
            if hi > merged[-1][1]:
                merged[-1][1] = hi
        else:
            merged.append([lo, hi])

    result = []
    seen = set()

    def emit(lo, hi):
        if (lo, hi) not in seen:
            seen.add((lo, hi))
            result.append([lo, hi])

    for lo, hi in base:
        cursor = lo
        for f_lo, f_hi in merged:
            if f_hi <= cursor:
                continue
            if f_lo >= hi:
                break
            if f_lo > cursor:
                emit(cursor, f_lo)
            cursor = f_hi
            if cursor >= hi:
                break
        if cursor < hi:
            emit(cursor, hi)
    return result

  
def get_unsafe_angles(intersection_points, robot_angles, x):
    forbidden_ranges = []
    none_points = np.isnan(intersection_points).all(axis=1)
    inf_points = np.isinf(intersection_points).all(axis=1)
    if np.any(inf_points):
        forbidden_ranges.extend(robot_angles)

    new_points = intersection_points[np.logical_not(np.logical_or(none_points, inf_points))]
    if new_points.shape[0] != 0:
        if len(new_points.shape) == 1:
            new_points = np.expand_dims(new_points, axis=0)
        p1 = new_points[:, :2]
        p2 = new_points[:, 2:]
        vec_p1 = np.array([p1[:, 0] - x[0], p1[:, 1] - x[1]])
        vec_p2 = np.array([p2[:, 0] - x[0], p2[:, 1] - x[1]])
        angle1 = np.arctan2(vec_p1[1], vec_p1[0])
        angle2 = np.arctan2(vec_p2[1], vec_p2[0])
        angle1_greater_mask = angle1 > angle2
        forbidden_ranges.extend(
            np.column_stack((
                angle1[~angle1_greater_mask],
                angle2[~angle1_greater_mask]
            ))
        )
        forbidden_ranges.extend(
            np.vstack((
                np.column_stack((angle1[angle1_greater_mask], np.full_like(angle1[angle1_greater_mask], math.pi))),
                np.column_stack((np.full_like(angle2[angle1_greater_mask], -math.pi), angle2[angle1_greater_mask]))
            ))
        )
    return forbidden_ranges


def get_unsafe_angles_wall(intersection_points, x):
    unsafe_angles = np.array(get_unsafe_angles(intersection_points, None, x), copy=True)
    approximations = np.arange(-np.pi, np.pi + 1, np.pi / 2)
    forbidden_ranges = []

    for i, unsafe_angle in enumerate(unsafe_angles):
        for j, angle in enumerate(unsafe_angles[i]):
            if angle in approximations:
                continue
            dist = angle_distance_vector(angle, approximations)
            idx = np.argmin(dist)
            unsafe_angles[i][j] = approximations[idx]

    angle1 = unsafe_angles[:, 0]
    angle2 = unsafe_angles[:, 1]
    angle1_greater_mask = angle1 > angle2
    forbidden_ranges.extend(np.column_stack((angle1[~angle1_greater_mask], angle2[~angle1_greater_mask])))
    forbidden_ranges.extend(
        np.vstack((
            np.column_stack((angle1[angle1_greater_mask], np.full_like(angle1[angle1_greater_mask], math.pi))),
            np.column_stack((np.full_like(angle2[angle1_greater_mask], -math.pi), angle2[angle1_greater_mask]))
        ))
    )
    return forbidden_ranges


def compute_safe_angle_space(intersection_points, max_angle_change, x, wall_angles):
    robot_angles = get_robot_angles(x, max_angle_change)

    # convert points into angles and define the forbidden angles space
    forbidden_ranges = get_unsafe_angles(intersection_points, robot_angles, x)

    new_robot_angles = compute_ranges_difference(robot_angles, forbidden_ranges)
    if len(new_robot_angles) == 0:
        return None, robot_angles
    else:
        return new_robot_angles, robot_angles


def compute_safe_angle_space_fast(x, circle_obs_x, circle_obs_rad, config, vmax):
    """
    Safe heading ranges for the tree's action pruning, at a given top speed.

    Same result as `get_radii` + `get_intersections_vectorized` +
    `compute_safe_angle_space`, done in one compiled call (`vo_safe_ranges`)
    instead of a chain of small numpy operations and Python list building, each
    of whose dispatch cost dominated the arithmetic it performed. The reference
    chain is kept below as `compute_safe_angle_space_fast_python`.

    Used only by `BetterEnv.get_actions_discrete_vo2`, i.e. the in-tree pruning
    that runs once per new node. The reactive VO-PLANNER keeps the original
    path: it evaluates VO once per control step, where the cost is irrelevant.

    :param vmax: top speed the pruning is computed for. Forward pruning uses
        config.max_speed, reverse pruning abs(config.min_speed).
    :return: (safe_ranges, any_vo). safe_ranges is None when no heading is safe
        and the full reachable span when no obstacle constrains it. any_vo says
        whether any obstacle produced a velocity obstacle at all, which is what
        lets the caller skip pruning entirely.
    """
    safe, any_vo = vo_safe_ranges(
        x, circle_obs_x, circle_obs_rad, config.dt, config.robot_radius, vmax,
        config.think_margin, config.max_angle_change, LEGACY_VO,
    )
    if not any_vo:
        return safe, False
    return (safe if len(safe) != 0 else None), True


def compute_safe_angle_space_fast_python(x, circle_obs_x, circle_obs_rad, config, vmax):
    """Reference implementation of the above; kept for the equivalence test."""
    r1, r0 = get_radii(
        circle_obs_x=circle_obs_x,
        circle_obs_rad=circle_obs_rad,
        dt=config.dt,
        robot_radius=config.robot_radius,
        vmax=vmax,
        think_margin=config.think_margin,
    )
    forbidden = vo_forbidden_ranges(x, circle_obs_x, r0, r1)
    robot_angles = get_robot_angles(x, config.max_angle_change)
    if len(forbidden) == 0:
        return robot_angles, False

    safe = compute_ranges_difference(robot_angles, forbidden)
    return (safe if len(safe) != 0 else None), True


def vo_negative_speed(obstacles, x, config):
    VELOCITY = np.abs(config.min_speed)
    ROBOT_RADIUS = config.robot_radius
    intersection_points = np.empty((0, 4), dtype=np.float64)
    max_angle_change = config.max_angle_change
    _, circle_obs, _ = obstacles

    # CIRCULAR OBSTACLES
    circle_obs_x = circle_obs[0]
    circle_obs_rad = circle_obs[1]

    if len(circle_obs_x) != 0:
        # Calculate radii
        r1, r0 = get_radii(
                circle_obs_x=circle_obs_x,
                circle_obs_rad=circle_obs_rad,
                dt=config.dt,
                robot_radius=ROBOT_RADIUS,
                vmax=VELOCITY,
                think_margin=config.think_margin,
            )
        intersection_points, dist, mask = get_intersections_vectorized(x, circle_obs_x, r0, r1)
    

    if np.isnan(intersection_points).all():
        # all robot angles are safe
        return get_robot_angles(x, config.max_angle_change), False
    else:
        x_copy = x.copy()
        x_copy[2] = x_copy[2] + np.pi
        x_copy[2] = (x_copy[2] + math.pi) % (2 * math.pi) - math.pi
        safe_angles, robot_span = compute_safe_angle_space(intersection_points, max_angle_change, x_copy, None)

        return safe_angles, True


def new_get_spaces(obstacles, x, config, intersection_points, wall_angles):
    safe_angles, robot_span = compute_safe_angle_space(intersection_points, config.max_angle_change, x, wall_angles)
    flip = False
    if safe_angles is None:
        safe_angles, flip = vo_negative_speed(obstacles, x, config)
        if safe_angles is None:
            candidates = []
            if TRAPPED_ESCAPE:
                circle_obs_x, circle_obs_rad = obstacles[1]
                if len(circle_obs_x) != 0:
                    candidates = compute_trapped_escape(x, circle_obs_x, circle_obs_rad, config)
            if candidates:
                # VO-PLANNER is reactive (no tree to weigh candidates against
                # each other via rollout, unlike VO-TREE/env.py), so it must
                # commit to one now. Pick whichever of every forward/reverse
                # pair, across every trapping obstacle, needs the smallest
                # turn from the current heading - the same "less turning is
                # better" reasoning a single candidate's own tie-break uses,
                # just extended to compare across obstacles too.
                best_heading, best_speed, best_abs_delta = None, None, None
                for heading_fwd, heading_rev, delta_fwd in candidates:
                    delta_rev = _signed_angle_diff(heading_rev, x[2])
                    if best_abs_delta is None or abs(delta_fwd) < best_abs_delta:
                        best_heading, best_speed, best_abs_delta = heading_fwd, config.max_speed, abs(delta_fwd)
                    if abs(delta_rev) < best_abs_delta:
                        best_heading, best_speed, best_abs_delta = heading_rev, config.min_speed, abs(delta_rev)
                # Headings here are already absolute (computed straight from
                # x[2]), unlike vo_negative_speed's flip-frame output above -
                # reusing flip=True here would double-add pi downstream.
                flip = False
                vspace = [best_speed, best_speed]
                safe_angles = [[best_heading, best_heading]]
            else:
                vspace = [0.0, 0.0]
                safe_angles = [[-math.pi, math.pi]]
        else:
            vspace = [config.min_speed, config.min_speed]
            # if flip:
            #     actions_backward[:, 1] = actions_backward[:, 1] + np.pi
            #         actions_backward[:, 1] = (actions_backward[:, 1] + np.pi) % (2 * np.pi) - np.pi
            pass
                
    else:
        vspace = [config.max_speed, config.max_speed]

    velocity_space = [*([vspace] * len(safe_angles))]
        
    angle_space = [*safe_angles]

    return angle_space, velocity_space, flip

def uniform_random_vo(node, planner):
    config = planner.environment.gym_env.config
    if len(node.state.obstacles) == 0:
        return uniform_random(
            min_speed=config.min_speed, 
            max_speed=config.max_speed, 
            curr_angle=node.state.x[2],
            max_angle_change=config.max_angle_change
        )

    # Extract robot information
    x = node.state.x
    dt = planner.environment.config.dt
    ROBOT_RADIUS = planner.environment.config.robot_radius
    VMAX = 0.3

    # Extract obstacle information
    intersection_points = np.empty((0, 4), dtype=np.float64)
    circle_obs_x, circle_obs_rad = node.state.obstacles


    if len(circle_obs_x) != 0:
        # Calculate radii
        r1, r0 = get_radii(circle_obs_x, circle_obs_rad, dt, ROBOT_RADIUS, VMAX,
                           think_margin=config.think_margin)

        # Calculate intersection points
        intersection_points, dist, mask = get_intersections_vectorized(x, circle_obs_x, r0, r1)
    

    # If there are no intersection points
    if np.isnan(intersection_points).all():
        return uniform_random(
            min_speed=config.min_speed, 
            max_speed=config.max_speed, 
            curr_angle=node.state.x[2],
            max_angle_change=config.max_angle_change
        )
    else:
        angle_space, velocity_space, flip = new_get_spaces([None, (circle_obs_x, circle_obs_rad), None], x, config, intersection_points,  wall_angles=None)
        sample = sample_multiple_spaces(center=None, a_space=angle_space, v_space=velocity_space, number=1)[0]
        if flip:
            sample[1] = sample[1] + np.pi
            sample[1] = (sample[1] + np.pi) % (2 * np.pi) - np.pi
        return sample
            


def epsilon_uniform_uniform_vo(
        node: Any, planner: Planner, std_angle_rollout: float, eps=0.1
):
    prob = random.random()
    if prob <= 1 - eps:
        return uniform_towards_goal_vo(node, planner, std_angle_rollout)
    else:
        return uniform_random_vo(node, planner)
