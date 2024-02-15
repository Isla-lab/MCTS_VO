# from cmath import atan
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from bettergym.environments.env_utils import check_coll_jit, dist_to_goal
from experiment_utils import plot_frame, plot_frame2


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

    max_speed_person: float = 1.0  # [m/s]
    
    dt: float = 1.0  # [s] Time tick for motion prediction
    robot_radius: float = 0.3  # [m] for collision check
    obs_size: float = 0.2

    bottom_limit: float = 0.0
    upper_limit: float = 100.0

    right_limit: float = 100.0
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
    num_humans: int = 100


class State:
    def __init__(self, x: np.ndarray, goal: np.ndarray, obstacles: list, radius: float):
        # x, y, angle ,vel_lin
        self.x: np.ndarray = x
        # x(m), y(m)
        self.goal: np.ndarray = goal
        self.obstacles: list = obstacles
        self.radius: float = radius
        # self.dynamic_obs: bool = False

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

        if collision_rwrd:
            self.step = self.step_check_coll
        # else:
        #     self.step = self.step_no_check_coll

    def is_within_range_check_with_points(self, p1_x, p1_y, p2_x, p2_y, threshold_distance):
        euclidean_distance = ((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2) ** 0.5
        return euclidean_distance <= threshold_distance

    def move_human(self, human_state: State, time_step: float):
        rand_num = (random.random() - 0.5) * 0.1
        heading_angle = deepcopy(human_state.x[2])
        # First Quadrant
        if human_state.goal[0] >= human_state.x[0] and human_state.goal[1] >= human_state.x[1]:
            if human_state.goal[0] == human_state.x[0]:
                new_x = human_state.x[0]
                new_y = human_state.x[1] + (human_state.x[3]) * time_step + rand_num
            elif human_state.goal[1] == human_state.x[1]:
                new_x = human_state.x[0] + (human_state.x[3]) * time_step + rand_num
                new_y = human_state.x[1]
            else:
                heading_angle = atan2((human_state.goal[1] - human_state.x[1]),
                                      (human_state.goal[0] - human_state.x[0]))
                new_x = human_state.x[0] + ((human_state.x[3]) * time_step + rand_num) * cos(heading_angle)
                new_y = human_state.x[1] + ((human_state.x[3]) * time_step + rand_num) * sin(heading_angle)
        # Second Quadrant
        elif human_state.goal[0] <= human_state.x[0] and human_state.goal[1] >= human_state.x[1]:
            if human_state.goal[0] == human_state.x[0]:
                new_x = human_state.x[0]
                new_y = human_state.x[1] + (human_state.x[3]) * time_step + rand_num
            elif human_state.goal[1] == human_state.x[1]:
                new_x = human_state.x[0] - (human_state.x[3]) * time_step - rand_num
                new_y = human_state.x[1]
            else:
                heading_angle = atan2((human_state.goal[1] - human_state.x[1]),
                                      (human_state.goal[0] - human_state.x[0]))
                new_x = human_state.x[0] - ((human_state.x[3]) * time_step + rand_num) * cos(heading_angle)
                new_y = human_state.x[1] - ((human_state.x[3]) * time_step + rand_num) * sin(heading_angle)
        # Third Quadrant
        elif human_state.goal[0] <= human_state.x[0] and human_state.goal[1] <= human_state.x[1]:
            if human_state.goal[0] == human_state.x[0]:
                new_x = human_state.x[0]
                new_y = human_state.x[1] - (human_state.x[3]) * time_step - rand_num
            elif human_state.goal[1] == human_state.x[1]:
                new_x = human_state.x[0] - (human_state.x[3]) * time_step - rand_num
                new_y = human_state.x[1]
            else:
                heading_angle = atan2((human_state.goal[1] - human_state.x[1]),
                                      (human_state.goal[0] - human_state.x[0]))
                new_x = human_state.x[0] - ((human_state.x[3]) * time_step + rand_num) * cos(heading_angle)
                new_y = human_state.x[1] - ((human_state.x[3]) * time_step + rand_num) * sin(heading_angle)
        # Fourth Quadrant
        # elif human_state.goal[0] >= human_state.x[0] and human_state.goal[1] <= human_state.x[1]:
        else:
            if human_state.goal[0] == human_state.x[0]:
                new_x = human_state.x[0]
                new_y = human_state.x[1] - (human_state.x[3]) * time_step - rand_num
            elif human_state.goal[1] == human_state.x[1]:
                new_x = human_state.x[0] - (human_state.x[3]) * time_step + rand_num
                new_y = human_state.x[1]
            else:
                heading_angle = atan2((human_state.goal[1] - human_state.x[1]),
                                      (human_state.goal[0] - human_state.x[0]))
                new_x = human_state.x[0] + ((human_state.x[3]) * time_step + rand_num) * cos(heading_angle)
                new_y = human_state.x[1] + ((human_state.x[3]) * time_step + rand_num) * sin(heading_angle)

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
        return check_coll_jit(
            state.x, np.array(obs_pos), state.radius, np.array(obs_rad)
        )

    def generate_humans(self, robot_state):
        g1 = [self.config.bottom_limit, self.config.left_limit]
        g2 = [self.config.bottom_limit, self.config.right_limit]
        g3 = [self.config.upper_limit, self.config.right_limit]
        g4 = [self.config.upper_limit, self.config.left_limit]
        all_goals_list = [g1, g2, g3, g4]
        humans = []

        def generate_human_state():
            return State(x=np.array([math.floor(self.config.right_limit * random.random()),
                                     math.floor(self.config.upper_limit * random.random()), 0, self.config.max_speed_person]),
                         goal=np.array(all_goals_list[random.randint(0, len(all_goals_list) - 1)]),
                         obstacles=None,
                         radius=self.config.obs_size)

        for _ in range(self.config.num_humans):
            human = generate_human_state()
            while self.is_within_range_check_with_points(human.x[0], human.x[1], robot_state.x[0], robot_state.x[1], 5.0):
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
        collision = self.check_collision(self.state)
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

    def step_real(self, state: State, action: np.ndarray):
        state_copy = deepcopy(state)
        for human_idx in range(len(state_copy.obstacles)):
            for _ in range(10):
                state_copy.obstacles[human_idx] = self.move_human(state.obstacles[human_idx], 0.1)
        self.state = state_copy
        return self.step_check_coll(action)

    def step_sim(self, state: State, action: np.ndarray):
        return self.step_check_coll(action)

    def reset(self):
        state = State(
            x=np.array([self.config.left_limit, 25.0, math.pi / 8.0, 0.0]),
            goal=np.array([self.config.right_limit, 75.0]),
            obstacles=None,
            radius=self.config.robot_radius,
        )
        state.obstacles = self.generate_humans(state)
        return state


def main():
    dt_real = 1.0
    real_c = EnvConfig(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=7, n_vel=10
    )
    env = Env(config=real_c)
    s0 = env.reset()
    env.state = s0
    trajectory = np.array(s0.x)
    goal = s0.goal
    obs = [s0.obstacles]
    s = s0
    for step_n in range(1000):
        print(f"Step Number {step_n}")
        s, r, terminal, truncated, env_info = env.step_real(state=s, action=[0.0, 0.0])
        trajectory = np.vstack((trajectory, s.x))  # store state history
        obs.append(s.obstacles)

    print("Creating Animation")
    fig, ax = plt.subplots()
    ani = FuncAnimation(
        fig,
        plot_frame2,
        fargs=(goal, real_c, obs, trajectory, ax),
        frames=tqdm(range(len(trajectory)), file=sys.stdout),
        save_count=None,
        cache_frame_data=False,
    )
    ani.save(f"debug/test_animation.gif", fps=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
