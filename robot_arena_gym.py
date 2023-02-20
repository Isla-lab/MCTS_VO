import math
from dataclasses import dataclass
from typing import Any, Union, Tuple, List

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame


@dataclass
class Config:
    """
    simulation parameter class
    """
    # robot parameter
    # Max U[0]
    max_speed = 1.0  # [m/s]
    # Min U[0]
    min_speed = -0.5  # [m/s]
    # Max and Min U[1]
    max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]

    # The action can be the relative change in speed -0.2 to 0.2
    max_accel = 0.2  # [m/ss]
    max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]

    # yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
    dt = 0.1  # [s] Time tick for motion prediction
    robot_radius = 0.3  # [m] for collision check

    bottom_limit = -0.5
    upper_limit = 10.5

    right_limit = 10.5
    left_limit = -0.5
    obs_size = 0.2

    # obstacles [x(m) y(m), ....]
    ob = np.array([
        [4.5, 5.0],
        [5.0, 4.5],
        [5.0, 5.0],
        [5.0, 5.5],
        [5.5, 5.0],
    ])


@dataclass
class RobotArenaState:
    # x, y, angle ,vel_lin, vel_ang
    x: np.ndarray
    # x(m), y(m)
    goal: np.ndarray


class RobotArena(gym.Env):
    def __init__(self, initial_position: Union[Tuple, List, np.ndarray], config: Config = Config()):
        self.state = RobotArenaState(
            x=np.array([initial_position[0], initial_position[1], math.pi / 8.0, 0.0, 0.0]),
            goal=np.array([10.0, 10.0])
        )
        self.config = config

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[RobotArenaState, Any]:
        return self.state, None

    def check_goal(self, state: RobotArenaState) -> bool:
        """
        Check if the robot reached the goal
        :param state: state of the robot
        :return:
        """
        x = state.x[:2]
        config = self.config
        goal = state.goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            return True
        return False

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

    def check_collision(self, state: RobotArenaState) -> bool:
        """
        Check if the robot is colliding with some obstacle
        :param state: state of the robot
        :return:
        """
        x = state.x
        config = self.config
        obs = self.config.ob
        tmp = np.expand_dims(x[:2], axis=0) - obs
        dist_to_obs = np.linalg.norm(tmp, axis=1)
        if np.any(dist_to_obs <= config.robot_radius+self.config.obs_size):
            return True
        return False

    def motion(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Describes how the robot moves
        :param x: current robot state
        :param u: action performed by the robot
        :return: the new robot state
        """
        dt = self.config.dt
        new_x = x.copy()
        # x
        new_x[0] += u[0] * math.cos(x[2]) * dt
        # y
        new_x[1] += u[0] * math.sin(x[2]) * dt
        # angle
        new_x[2] += u[1] * dt
        # vel lineare
        new_x[3] = u[0]
        # vel angolare
        new_x[4] = u[1]

        return new_x

    def step(self, action: np.ndarray) -> tuple[RobotArenaState, float, bool, bool, Any]:
        """
        Functions that computes all the things derived from a step
        :param action: action performed by the agent
        :return:
        """
        self.state.x = self.motion(self.state.x, action)
        collision = self.check_collision(self.state)
        goal = self.check_goal(self.state)
        out_boundaries = self.check_out_boundaries(self.state)
        reward = self.reward(self.state, action, collision, goal, out_boundaries)
        # observation, reward, terminal, truncated, info
        return self.state, reward, collision or goal or out_boundaries, False, None

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def reward(self, state: RobotArenaState, action: np.ndarray, is_collision: bool, is_goal: bool,
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
        COLLISION_REWARD: float = -5.0
        STEP_REWARD: float = -0.01

        if is_goal:
            return GOAL_REWARD

        if is_collision:
            return COLLISION_REWARD

        x_state = state.x

        return STEP_REWARD
