import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from bettergym.better_gym import BetterGym


@dataclass(frozen=True)
class Config:
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

    dt: float = 0.2  # [s] Time tick for motion prediction
    robot_radius: float = 0.3  # [m] for collision check
    obs_size: float = 0.6

    bottom_limit: float = -0.5
    upper_limit: float = 11.5

    right_limit: float = 11.5
    left_limit: float = -0.5

    n_vel: int = None
    n_angles: int = None


class RobotArenaState:
    def __init__(self, x: np.ndarray, goal: np.ndarray, obstacles: list, radius: float):
        # x, y, angle ,vel_lin, vel_ang
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
        return RobotArenaState(
            np.array(self.x, copy=True), self.goal, self.obstacles, self.radius
        )


@njit
def check_coll_jit(x, obs, robot_radius, obs_size):
    for i, ob in enumerate(obs):
        dist_to_ob = np.linalg.norm(ob - x[:2])
        if dist_to_ob <= robot_radius + obs_size[i]:
            return True
    return False


@njit
def dist_to_goal(goal: np.ndarray, x: np.ndarray):
    return np.linalg.norm(x - goal)


class RobotArena:
    def __init__(
            self,
            initial_state: RobotArenaState,
            config: Config = Config(),
            gradient: bool = True,
            collision_rwrd: bool = False,
            multiagent: bool = False
    ):
        self.state = initial_state
        bl_corner = np.array([config.bottom_limit, config.left_limit])
        ur_corner = np.array([config.upper_limit, config.right_limit])
        self.max_eudist = math.hypot(
            ur_corner[0] - bl_corner[0], ur_corner[1] - bl_corner[1]
        )
        self.config = config
        self.dist_goal_t1 = None
        self.dist_goal_t = None
        self.WALL_REWARD: float = -100.0

        if gradient:
            self.reward = self.reward_grad
        else:
            self.reward = self.reward_no_grad

        if multiagent:
            if collision_rwrd:
                self.step = self.multiagent_step_check_coll
            else:
                self.step = self.multiagent_step_no_check_coll
        else:
            if collision_rwrd:
                self.step = self.step_check_coll
            else:
                self.step = self.step_no_check_coll

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RobotArenaState, Any]:
        return self.state.copy(), None

    def check_out_boundaries(self, state: RobotArenaState) -> bool:
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

    def check_collision(self, state: RobotArenaState) -> bool:
        """
        Check if the robot is colliding with some obstacle
        :param x: state of the robot
        :return:
        """
        # config = self.config
        obs_pos = []
        obs_rad = []
        for ob in state.obstacles:
            obs_pos.append(ob.x[:2])
            obs_rad.append(ob.radius)
        return check_coll_jit(
            state.x, np.array(obs_pos), state.radius, np.array(obs_rad)
        )

    def motion(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Describes how the robot moves
        :param x: current robot state
        :param u: action performed by the robot
        :return: the new robot state
        """
        dt = self.config.dt
        new_x = np.array(x, copy=True)
        u = np.array(u, copy=True)
        # lin velocity
        # u[0] = max(-0.1, min(u[0], 0.3))
        # u[1] = max(x[2] - self.config.max_angle_change, min(u[1], x[2] + self.config.max_angle_change))

        # x
        new_x[0] += u[0] * math.cos(u[1]) * dt
        # y
        new_x[1] += u[0] * math.sin(u[1]) * dt
        # angle
        new_x[2] = u[1]
        # vel lineare
        new_x[3] = u[0]

        return new_x

    def multiagent_step_check_coll(self, action: np.ndarray) -> tuple[RobotArenaState, float, bool, Any, Any]:
        s1, r1, terminal1, truncated1, env_info1 = self.step_check_coll(action)
        dynamic_obs = self.state.obstacles[-1]
        action_dynamic_obs = dynamic_obs.x[-2:][::-1]
        self.state = dynamic_obs
        s2, _, _, _, _ = self.step_check_coll(action_dynamic_obs)
        s1.obstacles[-1] = s2
        return s1, r1, terminal1, truncated1, env_info1

    def multiagent_step_no_check_coll(self, action: np.ndarray) -> tuple[RobotArenaState, float, bool, Any, Any]:
        s1, r1, terminal1, truncated1, env_info1 = self.step_no_check_coll(action)
        dynamic_obs = self.state.obstacles[-1]
        action_dynamic_obs = dynamic_obs.x[:-2]
        self.state = dynamic_obs
        s2, _, _, _, _ = self.step_no_check_coll(action_dynamic_obs)
        s1.obstacles[-1] = s2
        return s1, r1, terminal1, truncated1, env_info1

    def step_check_coll(
            self, action: np.ndarray
    ) -> tuple[RobotArenaState, float, bool, Any, Any]:
        """
        Functions that computes all the things derived from a step
        :param action: action performed by the agent
        :return:
        """
        self.dist_goal_t = dist_to_goal(self.state.x[:2], self.state.goal)
        self.state.x = self.motion(self.state.x, action)
        self.dist_goal_t1 = dist_to_goal(self.state.x[:2], self.state.goal)
        collision = self.check_collision(self.state)
        goal = self.dist_goal_t1 <= self.config.robot_radius
        out_boundaries = self.check_out_boundaries(self.state)
        reward = self.reward(self.state, action, collision, goal, out_boundaries)
        # observation, reward, terminal, truncated, info
        return (
            self.state.copy(),
            reward,
            collision or goal or out_boundaries,
            None,
            None,
        )

    def step_no_check_coll(
            self, action: np.ndarray
    ) -> tuple[RobotArenaState, float, bool, Any, Any]:
        """
        Functions that computes all the things derived from a step
        :param action: action performed by the agent
        :return:
        """
        self.dist_goal_t = dist_to_goal(self.state.x[:2], self.state.goal)
        self.state.x = self.motion(self.state.x, action)
        self.dist_goal_t1 = dist_to_goal(self.state.x[:2], self.state.goal)
        goal = self.dist_goal_t1 <= self.config.robot_radius
        out_boundaries = self.check_out_boundaries(self.state)
        collision = False
        reward = self.reward(self.state, action, collision, goal, out_boundaries)
        # observation, reward, terminal, truncated, info
        return self.state.copy(), reward, goal or out_boundaries, None, None

    def reward_no_grad(
            self,
            state: RobotArenaState,
            action: np.ndarray,
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
        STEP_REWARD: float = -0.01

        if is_goal:
            return GOAL_REWARD

        if is_collision:
            return COLLISION_REWARD

        if out_boundaries:
            return self.WALL_REWARD

        return STEP_REWARD

    def reward_grad(
            self,
            state: RobotArenaState,
            action: np.ndarray,
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

        return (self.dist_goal_t - self.dist_goal_t1) / self.max_eudist


class UniformActionSpace:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high

    def sample(self):
        return np.array(
            [
                random.uniform(self.low[0], self.high[0]),
                random.uniform(self.low[1], self.high[1]),
            ]
        )


class BetterRobotArena(BetterGym):
    def __init__(
            self,
            initial_state: RobotArenaState,
            gradient: bool,
            discrete_env: bool,
            config: Config,
            collision_rwrd: bool,
            multiagent: bool = False
    ):
        if discrete_env:
            self.get_actions = self.get_actions_discrete
        else:
            self.get_actions = self.get_actions_continuous

        super().__init__(
            RobotArena(
                initial_state=initial_state,
                config=config,
                gradient=gradient,
                collision_rwrd=collision_rwrd,
                multiagent=multiagent
            )
        )

    def get_actions_continuous(self, state: RobotArenaState):
        config = self.gym_env.config

        return UniformActionSpace(
            low=np.array(
                [config.min_speed, state.x[2] - config.max_angle_change],
                dtype=np.float64,
            ),
            high=np.array(
                [config.max_speed, state.x[2] + config.max_angle_change],
                dtype=np.float64,
            ),
        )

    def get_actions_discrete(self, state: RobotArenaState):
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

    def set_state(self, state: RobotArenaState) -> None:
        self.gym_env.state = state.copy()
