import math
import random
from copy import deepcopy
from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Any, Tuple

import numpy as np

# How get_discrete_space measures an interval when splitting a fixed sample
# budget across several of them. False keeps np.linalg.norm(space, axis=1),
# i.e. sqrt(lo**2 + hi**2), which is what every result to date was produced
# with; True uses the width hi - lo, which is what the proportional split is
# supposed to mean. Set from loopHandler_copy.py before anything is built, the
# same way set_legacy_vo works.
RANGE_SIZE_WIDTH = False


def set_range_size_metric(use_width: bool) -> None:
    global RANGE_SIZE_WIDTH
    RANGE_SIZE_WIDTH = use_width



try:
    from MCTS_VO.bettergym.agents.utils.vo import compute_safe_angle_space, vo_negative_speed
    from MCTS_VO.bettergym.agents.utils.vo import compute_safe_angle_space_fast, robot_trapped
    from MCTS_VO.bettergym.agents.utils.utils import get_robot_angles
    from MCTS_VO.bettergym.better_gym import BetterGym
    from MCTS_VO.mcts_utils import get_intersections_vectorized
    from MCTS_VO.bettergym.agents.utils.vo import get_radii
    from MCTS_VO.bettergym.compiled_utils import robot_dynamics, check_coll_vectorized, dist_to_goal, discrete_actions, discrete_actions_multi_range, unique_rows
except ModuleNotFoundError:
    from bettergym.agents.utils.vo import compute_safe_angle_space, vo_negative_speed
    from bettergym.agents.utils.vo import compute_safe_angle_space_fast, robot_trapped
    from bettergym.agents.utils.utils import get_robot_angles
    from bettergym.better_gym import BetterGym
    from mcts_utils import get_intersections_vectorized
    from bettergym.agents.utils.vo import get_radii
    from bettergym.compiled_utils import robot_dynamics, check_coll_vectorized, dist_to_goal, discrete_actions, discrete_actions_multi_range, unique_rows


@dataclass(frozen=True)
class EnvConfig:
    """
    simulation parameter class
    """

    # robot parameter
    # Max U[0]
    max_speed: float = 0.3  # [m/s]
    # Min U[0]
    min_speed: float = -0.1  # [m/s]
    # Max and Min U[1]
    max_angle_change: float = 1.9  # [rad/s]

    max_speed_person: float = 0.2  # [m/s]

    dt: float = 1.0  # [s] Time tick for motion prediction
    robot_radius: float = 0.3  # [m] for collision check
    obs_size: float = 0.2

    # Time the robot spends sensing and planning, during which it is stopped but
    # the obstacles are not. Velocity obstacles must cover the distance an
    # obstacle can travel in dt + think_margin, not just dt. This used to be
    # hard-coded as 0.1 s inside get_radii, which was right only as long as
    # thinking took about that long; as a parameter it shrinks with the compute
    # time, which is what gives the planner back its manoeuvring room.
    think_margin: float = 0.1

    bottom_limit: float = -5.16911307
    upper_limit: float = 4.83088693

    right_limit: float = 4.09
    left_limit: float = -5.91

    n_vel: int = None
    n_angles: int = None
    max_yaw_rate = 1.9  # [rad/s]
    max_accel = 6  # [m/ss]
    max_delta_yaw_rate = 40 * math.pi / 180.0  # [rad/ss]
    v_resolution = 0.1  # [m/s]
    yaw_rate_resolution = max_yaw_rate / 11.0  # [rad/s]
    # predict_time = 15.0 * dt  # [s]
    predict_time = 100.0 * dt  # [s]
    to_goal_cost_gain = 1.
    speed_cost_gain = 0.0
    obstacle_cost_gain = 100.
    robot_stuck_flag_cons = 0.0  # constant to prevent robot stucked
    num_humans: int = 40

class State:
    def __init__(self, x: np.ndarray, goal: np.ndarray, obstacles: list, radius: float, obs_type: str = None):
        # x, y, angle ,vel_lin
        self.x: np.ndarray = x
        # x(m), y(m)
        self.goal: np.ndarray = goal
        self.obstacles: list = obstacles
        self.radius: float = radius
        self.obs_type: str = obs_type

    def __hash__(self):
        # tuple(bytes) expands to one int per byte, so the previous form built a
        # ~220 element tuple of ints on every lookup of the transposition table.
        # Hashing the bytes objects themselves is the same key, 3x cheaper.
        return hash((self.x.tobytes(),) + tuple(o.tobytes() for o in self.obstacles))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def copy(self):
        return State(
            self.x.copy(), self.goal, self.obstacles, self.radius, self.obs_type
        )

    def to_cartesian(self):
        self_copy = deepcopy(self)
        x, y, theta, v = self.x
        vx = v * math.cos(theta)
        vy = v * math.sin(theta)
        self_copy.x = np.array([x, y, vx, vy])

        if self.obstacles is not None:
            for i, ob in enumerate(self.obstacles):
                self_copy.obstacles[i] = ob.to_cartesian()

        return self_copy

    def to_polar(self):
        self_copy = deepcopy(self)
        x, y, vx, vy = self.x
        v = np.sqrt(vx ** 2 + vy ** 2)
        theta = np.arctan2(vy, vx)
        self_copy.x = np.array([x, y, theta, v])

        if self.obstacles is not None:
            for i, ob in enumerate(self.obstacles):
                self_copy.obstacles[i] = ob.to_polar()

        return self_copy


class Env:
    def __init__(self,
                 config: EnvConfig = EnvConfig(),
                 collision_rwrd: bool = True):
        self.config = EnvConfig()
        self.state = None
        bl_corner = np.array([config.bottom_limit, config.left_limit])
        ur_corner = np.array([config.upper_limit, config.right_limit])
        self.max_eudist = math.hypot(
            ur_corner[0] - bl_corner[0], ur_corner[1] - bl_corner[1]
        )
        self.config = config
        self.dist_goal_t1 = None
        self.dist_goal_t = None
        self.WALL_REWARD: float = -100.0

        self.reward = self.reward_grad
        self.step_idx = 0
        self.behaviour_type = "INTENTION"

    def is_within_range_check_with_points(self, p1_x, p1_y, p2_x, p2_y, threshold_distance):
        euclidean_distance = np.linalg.norm(np.array([p1_x, p1_y]) - np.array([p2_x, p2_y]))
        return euclidean_distance <= threshold_distance

    def generate_humans(self, robot_state):
        g1 = [self.config.bottom_limit, self.config.left_limit]
        g2 = [self.config.bottom_limit, self.config.right_limit]
        g3 = [self.config.upper_limit, self.config.right_limit]
        g4 = [self.config.upper_limit, self.config.left_limit]
        all_goals_list = [g1, g2, g3, g4]
        humans = []

        def generate_human_state():
            return State(x=np.array([math.floor(self.config.right_limit * random.random()),
                                     math.floor(self.config.upper_limit * random.random()), 0,
                                     self.config.max_speed_person]),
                         goal=np.array(all_goals_list[random.randint(0, len(all_goals_list) - 1)]),
                         obstacles=None,
                         radius=self.config.obs_size,
                         obs_type="circle"
                         )

        for _ in range(self.config.num_humans):
            human = generate_human_state()
            while self.is_within_range_check_with_points(human.x[0], human.x[1], robot_state.x[0], robot_state.x[1], 2):
                human = generate_human_state()
            humans.append(human)

        return humans

    def reward_grad(
            self,
            is_collision: bool,
            is_goal: bool,
            out_boundaries: bool,
    ) -> float:
        """
        Defines the reward the agent receives
        :param state: current robot state
        :param action: action performed by the agent
        :param is_collision: boolean value indicating if the robot is colliding
        :param is_goal: boolean value indicating if the robot has reached the goal
        :param out_boundaries: boolean value indicating if the robot is out of the map
        :return: The numerical reward of the agent
        """

        GOAL_REWARD: float = 100.0
        COLLISION_REWARD: float = -100.0

        if is_goal:
            return GOAL_REWARD

        if is_collision:
            return COLLISION_REWARD

        if out_boundaries:
            return self.WALL_REWARD

        return -self.dist_goal_t1 / self.max_eudist

    def check_out_boundaries(self, state: State) -> bool:
        """
        Check if the agent goes out of the map
        :param state: state of the robot
        :return:
        """
        x_pos, y_pos = state.x[:2]
        c = self.config
        return x_pos + c.robot_radius > c.right_limit or \
               x_pos - c.robot_radius < c.left_limit or \
               y_pos + c.robot_radius > c.upper_limit or \
               y_pos - c.robot_radius < c.bottom_limit

    def step_check_coll(self, action: np.ndarray) -> Tuple[State, float, bool, Any, Any]:
        """
        Functions that computes all the things derived from a step
        :param action: action performed by the agent
        :return:
        """
        self.state.x = robot_dynamics(self.state.x, action, self.config.dt)
        self.dist_goal_t1 = dist_to_goal(self.state.goal, self.state.x[:2])
        if len(self.state.obstacles[0]) != 0:
            collision = check_coll_vectorized(
                x=self.state.x[:2], 
                obs=self.state.obstacles[0][:, :2], 
                robot_radius=self.config.robot_radius, 
                obs_size=self.state.obstacles[1]
            )  
        else:
            collision = False
        robot_collision = collision and action[0] != 0
        goal = self.dist_goal_t1 <= self.config.robot_radius
        out_boundaries = False
        reward = self.reward(collision, goal, out_boundaries)
        # observation, reward, terminal, truncated, info
        return (
            self.state.copy(),
            reward,
            collision or goal or out_boundaries,
            None,
            {"collision": int(robot_collision), 
             "out_boundaries": int(out_boundaries),
             "colision_pedestrians": int(collision)},
        )

    def step_no_check_coll(self, action: np.ndarray) -> Tuple[State, float, bool, Any, Any]:
        """
        Functions that computes all the things derived from a step
        :param action: action performed by the agent
        :return:
        """
        self.state.x = robot_dynamics(self.state.x, action, self.config.dt)
        self.dist_goal_t1 = dist_to_goal(self.state.goal, self.state.x[:2])
        collision = False
        goal = self.dist_goal_t1 <= self.config.robot_radius
        out_boundaries = False
        reward = self.reward(collision, goal, out_boundaries)
        # observation, reward, terminal, truncated, info
        return (
            self.state.copy(),
            reward,
            collision or goal or out_boundaries,
            None,
            None,
        )

    def move_human_trefoil(self, human_state: State, t: int):
        omega = 0.1
        # t = random.randint(a=0, b=100)
        # human_state.x[0]
        # human_state.x[1]
        scale_x = 0.12
        scale_y = 0.1
        multiplier = random.choice([-1, 1])
        new_x = multiplier * scale_x * (sin(omega * t) + 2 * sin(2 * omega * t)) + human_state.x[0]
        new_y = multiplier * scale_y * (cos(omega * t) - 2 * cos(2 * omega * t)) + human_state.x[1]

        new_x = np.clip(new_x, self.config.left_limit, self.config.right_limit)
        new_y = np.clip(new_y, self.config.bottom_limit, self.config.upper_limit)

        delta_x = new_x - human_state.x[0]
        delta_y = new_y - human_state.x[1]

        delta_s = math.sqrt(delta_x ** 2 + delta_y ** 2)

        # Assuming a time interval (delta_t) of 1 (you can adjust this)
        delta_t = self.config.dt

        # Calculate speed
        speed = delta_s / delta_t
        # print(f"Speed {speed:.2f}, t {t}")

        heading_angle = np.arctan2(delta_y, delta_x)
        new_human_state = State(
            x=np.array([new_x, new_y, heading_angle, human_state.x[3]]),
            goal=human_state.goal,
            obstacles=None,
            radius=human_state.radius,
            obs_type=human_state.obs_type
        )
        return new_human_state

    def move_human_intentions(self, human_state: State, time_step: float):
        rand_num = (random.random() - 0.5)
        human_vel = random.choices([
            self.config.max_speed_person,
            random.uniform(-self.config.max_speed / 2, self.config.max_speed / 2),
        ])[0]
        # CLIP HUMAN VELOCITY
        human_vel = min(max(human_vel, -self.config.max_speed_person), self.config.max_speed_person)

        heading_angle = atan2((human_state.goal[1] - human_state.x[1]),
                              (human_state.goal[0] - human_state.x[0])) + rand_num * 2.5
        new_x = human_state.x[0] + (human_vel * time_step) * cos(heading_angle)
        new_y = human_state.x[1] + (human_vel * time_step) * sin(heading_angle)

        new_x = np.clip(new_x, self.config.left_limit, self.config.right_limit)
        new_y = np.clip(new_y, self.config.bottom_limit, self.config.upper_limit)

        new_human_state = State(
            x=np.array([new_x, new_y, heading_angle, human_state.x[3]]),
            goal=human_state.goal,
            obstacles=None,
            radius=human_state.radius,
            obs_type=human_state.obs_type
        )
        return new_human_state

    def move_humans_fixed(self):
        state_copy = deepcopy(self.state)
        state_copy.obstacles = self.obs_fixed[self.step_idx]
        self.state = state_copy

    def move_humans_nofixed(self):
        state_copy = deepcopy(self.state)
        to_remove = []
        for human_idx in range(len(self.state.obstacles)):
            human_state = self.state.obstacles[human_idx]
            if human_state.obs_type == "circle":
                if self.behaviour_type == "TREFOIL":
                    state_copy.obstacles[human_idx] = self.move_human_trefoil(human_state, self.step_idx)
                elif self.behaviour_type == "INTENTION":
                    human_state = self.move_human_intentions(human_state, self.config.dt)
                    state_copy.obstacles[human_idx] = human_state

                # remove obstacle when reaches goal
                if self.is_within_range_check_with_points(state_copy.obstacles[human_idx].x[0],
                                                          state_copy.obstacles[human_idx].x[1],
                                                          state_copy.obstacles[human_idx].goal[0],
                                                          state_copy.obstacles[human_idx].goal[1],
                                                          1):
                    to_remove.append(human_idx)
        # remove obstacles if near goal
        if len(to_remove) != 0:
            state_copy.obstacles = [elem for idx, elem in enumerate(state_copy.obstacles) if idx not in to_remove]
        self.state = state_copy

    def step_real(self, action: np.ndarray):
        self.move_humans()
        self.step_idx += 1
        return self.step_check_coll(action)

    # def step_sim_check_coll(self, action: np.ndarray):
    #     return self.step_check_coll(action)

    # def step_sim_no_check_coll(self, action: np.ndarray):
    #     return self.step_no_check_coll(action)

    def add_walls(self, state: State):
        walls = [
            # BOTTOM WALL
            [0., 0., 10., 0.],
            # UPPER WALL
            [10., 10., 0., 10.],
            # LEFT WALL
            [0., 10., 0., 0.],
            # RIGHT WALL
            [10., 0., 10., 10.]
        ]

        for wall in walls:
            state.obstacles.append(
                State(
                    x=np.array(wall),
                    goal=np.array([0, 0]),
                    obstacles=None,
                    radius=None,
                    obs_type="wall",
                )
            )

    def reset(self):
        state = State(
            x=np.array([1, 1, math.pi / 8.0, 0.0]),
            goal=np.array([9., 9.]),
            obstacles=None,
            radius=self.config.robot_radius,
        )
        # state.obstacles = self.generate_humans(state)
        # self.add_walls(state)
        return state, None


class BetterEnv(BetterGym):
    def __init__(
            self,
            discrete_env: bool,
            vo: bool,
            config: EnvConfig,
            collision_rwrd: bool,
            sim_env: bool,
            obs_pos: list,
    ):
        super().__init__(
            Env(
                config=config,
                collision_rwrd=collision_rwrd,
            )
        )

        if discrete_env:
            if not vo:
                self.get_actions = self.get_actions_discrete
            else:
                self.get_actions = self.get_actions_discrete_vo2

        if sim_env:
            if collision_rwrd:
                self.gym_env.step = self.gym_env.step_check_coll
            else:
                self.gym_env.step = self.gym_env.step_no_check_coll
            self.set_state = self.set_state_sim
        else:
            self.gym_env.step = self.gym_env.step_real
            self.set_state = self.set_state_real

        if obs_pos is not None:
            self.gym_env.obs_fixed = obs_pos
            self.gym_env.move_humans = self.gym_env.move_humans_fixed
        else:
            self.gym_env.move_humans = self.gym_env.move_humans_nofixed

    def get_discrete_actions_multi_range(self, aspace, vspace, config):
        return discrete_actions_multi_range(
            np.asarray(aspace, dtype=np.float64).reshape(-1, 2),
            np.asarray(vspace, dtype=np.float64).reshape(-1, 2),
            config.n_angles, config.n_vel, RANGE_SIZE_WIDTH,
        )

    def get_discrete_actions_multi_range_python(self, aspace, vspace, config):
        """Reference implementation of the above; kept for the equivalence test."""
        available_angles = self.get_discrete_space(aspace, config.n_angles)
        available_velocities = self.get_discrete_space(vspace, config.n_vel)
        actions = np.concatenate(
            [
                np.transpose([
                    np.tile(available_velocities[i], len(available_angles[i])),
                    np.repeat(available_angles[i], len(available_velocities[i])),
                ])
                for i in range(len(aspace))
            ]
        )
        return actions

    def get_actions_discrete_vo2(self, state: State):
        config = self.gym_env.config

        # The unpruned action set is only needed on the two early-return paths
        # below (no obstacles, or no velocity obstacle intersecting the reachable
        # set). Whenever VO actually prunes it used to be built and thrown away,
        # at 26 us a time on every new tree node.
        if len(state.obstacles) == 0:
            return self.get_actions_discrete(state)

        # Extract robot information
        x = state.x

        # CIRCULAR OBSTACLES
        circle_obs_x = state.obstacles[0]
        circle_obs_rad = state.obstacles[1]

        if len(circle_obs_x) == 0:
            return self.get_actions_discrete(state)

        # Trapped: the outcome is fixed before any geometry runs, so take it
        # here rather than through two pruning passes that both end up with an
        # empty span. Same action set as the fallback branch below, built the
        # same way and put through the same np.unique, so this is a pure
        # shortcut - it changes timings only, never a run.
        if robot_trapped(x, circle_obs_x, circle_obs_rad, config):
            actions = self.get_discrete_actions_multi_range(
                [[-math.pi, math.pi]], [[0.0, 0.0]], config
            )
            return unique_rows(actions)

        safe_angles_forward, any_vo = compute_safe_angle_space_fast(
            x, circle_obs_x, circle_obs_rad, config, config.max_speed
        )

        # No obstacle constrains the reachable headings at top speed: nothing to
        # prune, so the full discrete action set stands.
        if not any_vo:
            return self.get_actions_discrete(state)
        else:
            if forward_available := (safe_angles_forward is not None):
                vspace = [config.max_speed, config.max_speed]
                v_space_forward = [*([vspace] * len(safe_angles_forward))]
                actions_forward = self.get_discrete_actions_multi_range(safe_angles_forward, v_space_forward, config)
            else:
                actions_forward = np.empty((0, 2))

            # Reverse: the same geometry seen from the opposite heading and at
            # the reverse top speed, mirroring vo_negative_speed. When no
            # velocity obstacle applies there, the span is the unflipped one and
            # the resulting angles need no flipping back.
            x_flipped = x.copy()
            x_flipped[2] = (x[2] + np.pi + np.pi) % (2 * np.pi) - np.pi
            safe_angles_backward, any_vo_backward = compute_safe_angle_space_fast(
                x_flipped, circle_obs_x, circle_obs_rad, config, abs(config.min_speed)
            )
            flip = any_vo_backward
            if not any_vo_backward:
                safe_angles_backward = get_robot_angles(x, config.max_angle_change)

            if retro_available := (safe_angles_backward is not None):
                vspace = [config.min_speed, config.min_speed]
                v_space_backward = [*([vspace] * len(safe_angles_backward))]
                actions_backward = self.get_discrete_actions_multi_range(safe_angles_backward, v_space_backward, config)
                if flip:
                    actions_backward[:, 1] = actions_backward[:, 1] + np.pi
                    actions_backward[:, 1] = (actions_backward[:, 1] + np.pi) % (2 * np.pi) - np.pi
            else:
                actions_backward = np.empty((0, 2))

            if not (forward_available or retro_available):
                vspace = [[0.0, 0.0]]
                safe_angles = [[-math.pi, math.pi]]
                actions = self.get_discrete_actions_multi_range(safe_angles, vspace, config)
            else:
                actions = np.concatenate([actions_forward, actions_backward])

            actions = unique_rows(actions)
            if len(actions) > config.n_angles * config.n_vel:
                actions = np.random.choice(actions, size=config.n_angles * config.n_vel, replace=False)
            return actions

    def get_actions_discrete(self, state: State):
        config = self.gym_env.config
        return discrete_actions(
            state.x,
            config.max_angle_change,
            config.min_speed,
            config.max_speed,
            config.n_angles,
            config.n_vel,
        )


    def get_discrete_space(self, space, n_sample):
        """
        Split `n_sample` samples across the intervals in `space`, in proportion
        to how large each one is.

        "Large" should mean the width, hi - lo. The original measure was
        np.linalg.norm(space, axis=1) = sqrt(lo**2 + hi**2), the distance of the
        pair from the origin, which tracks how far the interval sits from zero
        far more than how wide it is: [3.0, 3.2] and [-0.1, 0.1] have the same
        width but norms of 4.38 and 0.14, a factor of 31 in samples allocated.

        It only bites with more than one interval, since a single one
        normalises to 1 either way. get_robot_angles returns two whenever the
        reachable span crosses +-pi, i.e. when the robot heads roughly west, so
        the action set depended on absolute heading and not only on the local
        geometry.

        Kept switchable because every result up to and including the 2400-run
        campaign was produced with the norm.
        """
        if RANGE_SIZE_WIDTH:
            # abs, not a bare difference: the norm this replaces was always
            # non-negative, and a signed width would be a new failure mode -
            # a negative size gives a negative proportion, hence a negative
            # div, which np.linspace takes as a sample count. Callers should
            # only ever pass lo <= hi, so this is a guard rather than a fix.
            sp = np.asarray(space, dtype=np.float64).reshape(-1, 2)
            range_sizes = np.abs(sp[:, 1] - sp[:, 0])
        else:
            range_sizes = np.linalg.norm(space, axis=1)
        # ensure that the range sizes are not zero
        range_sizes += 1e-6
        proportion = range_sizes / np.sum(range_sizes)
        div = proportion * n_sample
        #  floor all odd indices and ceil all even indices
        div[::2] = np.floor(div[::2])
        div[1::2] = np.ceil(div[1::2])
        div = div.astype(int)
        return [
            np.linspace(start=space[i][0], stop=space[i][1], num=div[i])
            for i in range(len(space))
        ]

    def set_state_sim(self, state: State) -> None:
        self.gym_env.state = state.copy()

    def set_state_real(self, state: State) -> None:
        self.gym_env.state = deepcopy(state)
