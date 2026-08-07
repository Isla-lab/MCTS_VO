import math
import random
from typing import Any
import numpy as np

try:
    from MCTS_VO.bettergym.agents.planner import Planner
    from MCTS_VO.bettergym.agents.utils.utils import get_robot_angles, compute_uniform_towards_goal_jit
    from MCTS_VO.mcts_utils import get_intersections_vectorized, angle_distance_vector
    from MCTS_VO.bettergym.compiled_utils import uniform_random, vo_forbidden_ranges
except ModuleNotFoundError:
    from bettergym.agents.planner import Planner
    from bettergym.agents.utils.utils import get_robot_angles, compute_uniform_towards_goal_jit
    from mcts_utils import get_intersections_vectorized, angle_distance_vector
    from bettergym.compiled_utils import uniform_random, vo_forbidden_ranges
    
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
    `compute_safe_angle_space`, with the geometry done in one compiled call
    (`vo_forbidden_ranges`) instead of a chain of small numpy operations, each
    of whose dispatch cost dominated the arithmetic it performed.

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
