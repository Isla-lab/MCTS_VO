import math
import random
from copy import deepcopy
from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Any

import numpy as np

from bettergym.agents.utils.utils import get_robot_angles
from bettergym.agents.utils.vo import new_get_spaces
from bettergym.better_gym import BetterGym
from bettergym.environments.env_utils import dist_to_goal, check_coll_vectorized
from mcts_utils import get_intersections_vectorized


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
    max_angle_change: float = None  # [rad/s]

    max_speed_person: float = 0.3  # [m/s]

    dt: float = 1.0  # [s] Time tick for motion prediction
    robot_radius: float = 0.3  # [m] for collision check
    obs_size: float = 0.2

    bottom_limit: float = 0.0
    upper_limit: float = 10.0

    right_limit: float = 10.0
    left_limit: float = 0.0

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
    def __init__(self, x: np.ndarray, goal: np.ndarray, obstacles: list, radius: float):
        # x, y, angle ,vel_lin
        self.x: np.ndarray = x
        # x(m), y(m)
        self.goal: np.ndarray = goal
        self.obstacles: list = obstacles
        self.radius: float = radius

    def __hash__(self):
        return hash(
            tuple(self.x.tobytes()) + tuple(o.x.tobytes() for o in self.obstacles)
        )

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def copy(self):
        return State(
            np.array(self.x, copy=True), self.goal, self.obstacles, self.radius
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

        # if collision_rwrd:
        #     self.step = self.step_check_coll
        # else:
        #     self.step = self.step_no_check_coll

    def is_within_range_check_with_points(self, p1_x, p1_y, p2_x, p2_y, threshold_distance):
        euclidean_distance = np.linalg.norm(np.array([p1_x, p1_y]) - np.array([p2_x, p2_y]))
        return euclidean_distance <= threshold_distance

    def move_human(self, human_state: State, time_step: float):
        rand_num = (random.random() - 0.5)
        # rand_num = (random.random() - 0.5) * 0.1
        # heading_angle = deepcopy(human_state.x[2])
        human_vel = random.choices([self.config.max_speed_person, self.config.max_speed_person / 2 + rand_num * 0.1])[0]
        # human_vel = 0.3

        heading_angle = atan2((human_state.goal[1] - human_state.x[1]),
                              (human_state.goal[0] - human_state.x[0])) + rand_num
        new_x = human_state.x[0] + (human_vel * time_step) * cos(heading_angle)
        new_y = human_state.x[1] + (human_vel * time_step) * sin(heading_angle)

        new_x = np.clip(new_x, self.config.left_limit, self.config.right_limit)
        new_y = np.clip(new_y, self.config.bottom_limit, self.config.upper_limit)
        new_human_state = State(
            x=np.array([new_x, new_y, heading_angle, human_state.x[3]]),
            goal=human_state.goal,
            obstacles=None,
            radius=human_state.radius
        )
        return new_human_state

    def check_collision(self, state: State) -> bool:
        """
        Check if the robot is colliding with some obstacle
        :param state: state of the robot
        :return:
        """
        # config = self.config
        obs_pos = []
        obs_rad = []
        for ob in state.obstacles:
            obs_pos.append(ob.x[:2])
            obs_rad.append(ob.radius)

        return check_coll_vectorized(x=state.x, obs=obs_pos, robot_radius=self.config.robot_radius,
                                     obs_size=np.array(obs_rad))

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
                         radius=self.config.obs_size)

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

        return - self.dist_goal_t1 / self.max_eudist

    def check_out_boundaries(self, state: State) -> bool:
        """
        Check if the agent goes out of the map
        :param state: state of the robot
        :return:
        """
        x_pos, y_pos = state.x[:2]
        c = self.config
        # Right and Left Map Limit
        if (
                x_pos + c.robot_radius > c.right_limit
                or x_pos - c.robot_radius < c.left_limit
        ):
            return True
        # Upper and Bottom Map Limit
        if (
                y_pos + c.robot_radius > c.upper_limit
                or y_pos - c.robot_radius < c.bottom_limit
        ):
            return True

        return False

    def robot_motion(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Describes how the robot moves
        :param x: current robot state
        :param u: action performed by the robot
        :return: the new robot state
        """
        dt = self.config.dt
        new_x = np.array(x, copy=True)
        u = np.array(u, copy=True)
        # print(new_x)
        new_x[0] += u[0] * math.cos(u[1]) * dt
        # y
        new_x[1] += u[0] * math.sin(u[1]) * dt
        # angle
        new_x[2] = u[1]
        # vel lineare
        new_x[3] = u[0]

        return new_x

    def step_check_coll(
            self, action: np.ndarray
    ) -> tuple[State, float, bool, Any, Any]:
        """
        Functions that computes all the things derived from a step
        :param action: action performed by the agent
        :return:
        """
        self.dist_goal_t = dist_to_goal(self.state.x[:2], self.state.goal)
        self.state.x = self.robot_motion(self.state.x, action)
        self.dist_goal_t1 = dist_to_goal(self.state.x[:2], self.state.goal)
        collision = self.check_collision(self.state) and action[0] > 0
        goal = self.dist_goal_t1 <= self.config.robot_radius
        out_boundaries = self.check_out_boundaries(self.state)
        reward = self.reward(collision, goal, out_boundaries)
        # observation, reward, terminal, truncated, info
        return (
            self.state.copy(),
            reward,
            collision or goal or out_boundaries,
            None,
            {"collision": int(collision), "out_boundaries": int(out_boundaries)},
        )

    def step_no_check_coll(
            self, action: np.ndarray
    ) -> tuple[State, float, bool, Any, Any]:
        """
        Functions that computes all the things derived from a step
        :param action: action performed by the agent
        :return:
        """
        self.dist_goal_t = dist_to_goal(self.state.x[:2], self.state.goal)
        self.state.x = self.robot_motion(self.state.x, action)
        self.dist_goal_t1 = dist_to_goal(self.state.x[:2], self.state.goal)
        # TODO check collision
        collision = False
        goal = self.dist_goal_t1 <= self.config.robot_radius
        out_boundaries = self.check_out_boundaries(self.state)
        reward = self.reward(collision, goal, out_boundaries)
        # observation, reward, terminal, truncated, info
        return (
            self.state.copy(),
            reward,
            collision or goal or out_boundaries,
            None,
            None,
        )

    def move_humans_fixed(self):
        state_copy = deepcopy(self.state)
        state_copy.obstacles = self.obs_fixed[self.step_idx]
        self.state = state_copy

    def move_humans(self):
        state_copy = deepcopy(self.state)
        to_remove = []
        for human_idx in range(len(self.state.obstacles)):
            for _ in range(10):
                state_copy.obstacles[human_idx] = self.move_human(self.state.obstacles[human_idx], 0.1)
                if self.is_within_range_check_with_points(state_copy.obstacles[human_idx].x[0],
                                                          state_copy.obstacles[human_idx].x[1],
                                                          state_copy.obstacles[human_idx].goal[0],
                                                          state_copy.obstacles[human_idx].goal[1],
                                                          1):
                    to_remove.append(human_idx)
                    break
        # remove obstacles if near goal
        if len(to_remove) != 0:
            state_copy.obstacles = [elem for idx, elem in enumerate(state_copy.obstacles) if idx not in to_remove]
        self.state = state_copy

    def step_real(self, action: np.ndarray):
        self.move_humans()
        self.step_idx += 1
        return self.step_check_coll(action)

    def step_sim_check_coll(self, action: np.ndarray):
        return self.step_check_coll(action)

    def step_sim_no_check_coll(self, action: np.ndarray):
        return self.step_no_check_coll(action)

    def reset(self):
        state = State(
            x=np.array([1, 1, math.pi / 8.0, 0.0]),
            goal=np.array([9., 9.]),
            obstacles=None,
            radius=self.config.robot_radius,
        )
        state.obstacles = self.generate_humans(state)
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
                self.gym_env.step = self.gym_env.step_sim_check_coll
            else:
                self.gym_env.step = self.gym_env.step_sim_no_check_coll
            self.set_state = self.set_state_sim
        else:
            self.gym_env.step = self.gym_env.step_real
            self.set_state = self.set_state_real

        if obs_pos is not None:
            self.gym_env.obs_fixed = obs_pos
            self.gym_env.move_humans = self.gym_env.move_humans_fixed

    def get_actions_discrete(self, state: State):
        config = self.gym_env.config
        available_angles = np.linspace(
            start=state.x[2] - config.max_angle_change,
            stop=state.x[2] + config.max_angle_change,
            num=config.n_angles,
        )
        if (curr_angle := state.x[2]) not in available_angles:
            available_angles = np.append(available_angles, curr_angle)
        available_angles = (available_angles + np.pi) % (2 * np.pi) - np.pi
        available_velocities = np.linspace(
            start=config.min_speed, stop=config.max_speed, num=config.n_vel
        )
        if 0.0 not in available_velocities:
            available_velocities = np.append(available_velocities, 0.0)

        actions = np.transpose(
            [
                np.tile(available_velocities, len(available_angles)),
                np.repeat(available_angles, len(available_velocities)),
            ]
        )
        return actions

    def get_discrete_actions_basic(self, x, config, min_speed, max_speed):
        feasibile_range = get_robot_angles(x, config.max_angle_change)
        if len(feasibile_range) == 1:
            available_angles = np.linspace(
                start=x[2] - config.max_angle_change,
                stop=x[2] + config.max_angle_change,
                num=config.n_angles,
            )
        else:
            range_sizes = np.linalg.norm(feasibile_range, axis=1)
            proportion = range_sizes / np.sum(range_sizes)
            div = proportion * config.n_angles
            div[0] = np.floor(div[0])
            div[1] = np.ceil(div[1])
            div = div.astype(int)
            available_angles1 = np.linspace(
                start=feasibile_range[0][0],
                stop=feasibile_range[0][1],
                num=div[0]
            )
            available_angles2 = np.linspace(
                start=feasibile_range[1][0],
                stop=feasibile_range[1][1],
                num=div[1]
            )
            available_angles = np.concatenate([available_angles1, available_angles2])

        if (curr_angle := x[2]) not in available_angles:
            available_angles = np.append(available_angles, curr_angle)
        available_velocities = np.linspace(
            start=min_speed, stop=max_speed, num=config.n_vel
        )
        if 0.0 not in available_velocities:
            available_velocities = np.append(available_velocities, 0.0)

        actions = np.transpose(
            [
                np.tile(available_velocities, len(available_angles)),
                np.repeat(available_angles, len(available_velocities)),
            ]
        )
        return actions

    def get_discrete_space(self, space, n_sample):
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

    def get_discrete_actions_multi_range(self, aspace, vspace, config):
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
        actions = self.get_discrete_actions_basic(state.x, config, config.min_speed, config.max_speed)

        if len(state.obstacles) == 0:
            return actions
        # Extract obstacle information
        obstacles = state.obstacles
        obs_x = np.array([ob.x for ob in obstacles])
        obs_rad = np.array([ob.radius for ob in obstacles])

        # Extract robot information
        x = state.x
        dt = config.dt
        ROBOT_RADIUS = config.robot_radius
        VMAX = 0.3

        # Calculate radii
        r1 = obs_x[:, 3] * dt + obs_rad + VMAX * dt
        r0 = np.full_like(r1, ROBOT_RADIUS)

        # Calculate intersection points
        intersection_points, dist, mask = get_intersections_vectorized(x, obs_x, r0, r1)
        # If there are no intersection points
        if np.isnan(intersection_points).all():
            return actions
        else:
            angle_space, velocity_space = new_get_spaces(obs_x, mask, r0, r1, x, config, intersection_points)
            actions = self.get_discrete_actions_multi_range(angle_space, velocity_space, config)
            return actions

    def set_state_sim(self, state: State) -> None:
        self.gym_env.state = state.copy()

    def set_state_real(self, state: State) -> None:
        self.gym_env.state = deepcopy(state)
