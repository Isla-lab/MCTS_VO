import math
import random
from dataclasses import dataclass
from typing import Any, Union, Tuple, List

import numpy as np
from numba import njit

from bettergym.better_gym import BetterGym


@dataclass
class Config:
    """
    simulation parameter class
    """
    # robot parameter
    # Max U[0]
    max_speed = 0.3  # [m/s]
    # Min U[0]
    min_speed = -0.1  # [m/s]
    # Max and Min U[1]
    max_angle_change = 0.38  # [rad/s]

    dt = 0.2  # [s] Time tick for motion prediction
    robot_radius = 0.3  # [m] for collision check
    obs_size = 0.2

    bottom_limit = -0.5
    upper_limit = 11.5

    right_limit = 11.5
    left_limit = -0.5

    num_discrete_actions = 5

    # obstacles [x(m) y(m), ....]
    ob = np.array([
        [4.5, 5.0],
        [5.0, 4.5],
        [5.0, 5.0],
        [5.0, 5.5],
        [5.5, 5.0],
    ])


class RobotArenaState:
    def __init__(self, x: np.ndarray, goal: np.ndarray):
        # x, y, angle ,vel_lin, vel_ang
        self.x: np.ndarray = x
        # x(m), y(m)
        self.goal: np.ndarray = goal

    def __hash__(self):
        return hash(self.x.tobytes())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def copy(self):
        return RobotArenaState(
            np.array(self.x, copy=True),
            self.goal
        )


@njit
def check_coll_jit(x, obs, robot_radius, obs_size):
    for ob in obs:
        dist_to_ob = math.hypot(ob[0] - x[0], ob[1] - x[1])
        if dist_to_ob <= robot_radius + obs_size:
            return True
    return False


@njit
def reward_grad_jit(is_goal: bool, is_collision: bool, out_boundaries: bool, goal: np.ndarray, max_eudist: float,
                    x: np.ndarray):
    GOAL_REWARD: float = 1.0
    COLLISION_REWARD: float = -100.0

    if is_goal:
        return GOAL_REWARD

    if is_collision or out_boundaries:
        return COLLISION_REWARD

    return -np.linalg.norm(goal - x) / max_eudist


@njit
def check_goal_jit(goal: np.ndarray, x: np.ndarray, robot_radius: float):
    dist_to_goal = math.hypot(goal[0] - x[0], goal[1] - x[1])
    return dist_to_goal <= robot_radius


class RobotArena:
    def __init__(self, initial_position: Union[Tuple, List, np.ndarray], config: Config = Config(),
                 gradient: bool = True):
        self.state = RobotArenaState(
            x=np.array([initial_position[0], initial_position[1], math.pi / 8.0, 0.0]),
            goal=np.array([8.0, 8.0])
        )
        bl_corner = np.array([config.bottom_limit, config.left_limit])
        ur_corner = np.array([config.upper_limit, config.right_limit])
        self.max_eudist = math.hypot(ur_corner[0] - bl_corner[0], ur_corner[1] - bl_corner[1])
        self.config = config

        if gradient:
            self.reward = self.reward_grad
        else:
            self.reward = self.reward_no_grad

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[RobotArenaState, Any]:
        return self.state.copy(), None

    def check_goal(self, state: RobotArenaState) -> bool:
        """
        Check if the robot reached the goal
        :param state: state of the robot
        :return:
        """
        return check_goal_jit(
            goal=state.goal,
            x=state.x,
            robot_radius=self.config.robot_radius
        )

    def check_out_boundaries(self, state: RobotArenaState) -> bool:
        """
        Check if the agent goes out of the map
        :param state: state of the robot
        :return:
        """
        x_pos, y_pos = state.x[:2]
        c = self.config
        # Right and Left Map Limit
        if x_pos + c.robot_radius > c.right_limit or x_pos - c.robot_radius < c.left_limit:
            return True
        # Upper and Bottom Map Limit
        if y_pos + c.robot_radius > c.upper_limit or y_pos - c.robot_radius < c.bottom_limit:
            return True

        return False

    def check_collision(self, x: np.ndarray) -> bool:
        """
        Check if the robot is colliding with some obstacle
        :param x: state of the robot
        :return:
        """
        config = self.config
        return check_coll_jit(x, config.ob, config.robot_radius, config.obs_size)

    def motion(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Describes how the robot moves
        :param x: current robot state
        :param u: action performed by the robot
        :return: the new robot state
        """
        dt = self.config.dt
        new_x = np.array(x, copy=True)
        # lin velocity
        if u[0] > self.config.max_speed:
            u[0] = self.config.max_speed
        elif u[0] < self.config.min_speed:
            u[0] = self.config.min_speed

        # angle
        if u[1] > u[1] + self.config.max_angle_change:
            u[1] = self.config.max_angle_change
        elif u[1] < u[1] - self.config.max_angle_change:
            u[1] = -self.config.max_angle_change

        # angle
        # Make sure angle is within range of -π to π
        u[1] = (u[1] + math.pi) % (2 * math.pi) - math.pi
        new_x[2] = u[1]

        # vel lineare
        new_x[3] = u[0]
        # x
        new_x[0] += u[0] * math.cos(new_x[2]) * dt
        # y
        new_x[1] += u[0] * math.sin(new_x[2]) * dt

        return new_x

    def step(self, action: np.ndarray) -> tuple[RobotArenaState, float, bool, bool, Any]:
        """
        Functions that computes all the things derived from a step
        :param action: action performed by the agent
        :return:
        """
        self.state.x = self.motion(self.state.x, action)
        collision = self.check_collision(self.state.x)
        goal = self.check_goal(self.state)
        out_boundaries = self.check_out_boundaries(self.state)
        reward = self.reward(self.state, action, collision, goal, out_boundaries)
        # observation, reward, terminal, truncated, info
        return self.state.copy(), reward, collision or goal or out_boundaries, False, None

    def reward_no_grad(self, state: RobotArenaState, action: np.ndarray, is_collision: bool, is_goal: bool,
                       out_boundaries: bool) -> float:
        """
        Defines the reward the agent receives
        :param state: current robot state
        :param action: action performed by the agent
        :param is_collision: boolean value indicating if the robot is colliding
        :param is_goal: boolean value indicating if the robot has reached the goal
        :param out_boundaries: boolean value indicating if the robot is out of the map
        :return: The numerical reward of the agent
        """
        GOAL_REWARD: float = 1.0
        COLLISION_REWARD: float = -100.0
        STEP_REWARD: float = -0.01

        if is_goal:
            return GOAL_REWARD

        if is_collision or out_boundaries:
            return COLLISION_REWARD

        return STEP_REWARD

    def reward_grad(self, state: RobotArenaState, action: np.ndarray, is_collision: bool, is_goal: bool,
                    out_boundaries: bool) -> float:
        """
        Defines the reward the agent receives
        :param state: current robot state
        :param action: action performed by the agent
        :param is_collision: boolean value indicating if the robot is colliding
        :param is_goal: boolean value indicating if the robot has reached the goal
        :param out_boundaries: boolean value indicating if the robot is out of the map
        :return: The numerical reward of the agent
        """
        return reward_grad_jit(
            is_goal=is_goal,
            is_collision=is_collision,
            out_boundaries=out_boundaries,
            goal=state.goal,
            max_eudist=self.max_eudist,
            x=state.x[:2]
        )


class UniformActionSpace:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high

    def sample(self):
        return np.array(
            [
                random.uniform(self.low[0], self.high[0]),
                random.uniform(self.low[1], self.high[1])
            ]
        )


class BetterRobotArena(BetterGym):

    def __init__(self, initial_position: Union[Tuple, List, np.ndarray], gradient: bool = True, discrete: bool = False,
                 config: Config = Config()):
        if discrete:
            self.get_actions = self.get_actions_discrete
        else:
            self.get_actions = self.get_actions_continuous

        super().__init__(RobotArena(initial_position=initial_position, gradient=gradient, config=config))

    def get_actions_continuous(self, state: RobotArenaState):
        config = self.gym_env.config

        return UniformActionSpace(
            low=np.array([config.min_speed, state.x[2] - config.max_angle_change], dtype=np.float64),
            high=np.array([config.max_speed, state.x[2] + config.max_angle_change], dtype=np.float64)
        )

    def get_actions_discrete(self, state: RobotArenaState):
        config = self.gym_env.config

        return np.linspace(
            start=np.array([config.min_speed, state.x[2] - config.max_angle_change], dtype=np.float64),
            stop=np.array([config.max_speed, state.x[2] + config.max_angle_change], dtype=np.float64),
            num=config.num_discrete_actions
        )

    def set_state(self, state: RobotArenaState) -> None:
        self.gym_env.state = state.copy()
