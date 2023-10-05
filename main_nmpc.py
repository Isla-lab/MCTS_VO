import argparse
import gc
import os
import pickle
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numpy import mean, std

from bettergym.agents.planner_dwa import Dwa
from bettergym.environments.robot_arena import dist_to_goal
from environment_creator import (
    create_env_five_small_obs_continuous,
)
from experiment_utils import (
    print_and_notify,
    plot_frame,
)
from mcts_utils import uniform_random

DEBUG_DATA = False
DEBUG_ANIMATION = False
ANIMATION = True


@dataclass(frozen=True)
class ExperimentData:
    rollout_policy: Callable
    action_expansion_policy: Callable
    discrete: bool
    obstacle_reward: bool
    std_angle: float
    n_sim: int = 1000
    c: float = 150
    vo: bool = False


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    seed_numba(seed_value)


def run_experiment(experiment: ExperimentData, arguments):
    global exp_num
    # input [forward speed, yaw_rate]
    start_pos = (1, 1)

    real_env, sim_env = create_env_five_small_obs_continuous(
        initial_pos=start_pos,
        goal=(10, 10),
        discrete=experiment.discrete,
        rwrd_in_sim=experiment.obstacle_reward,
        out_boundaries_rwrd=arguments.rwrd,
        dt_sim=arguments.dt,
        n_vel=arguments.v,
        n_angles=arguments.a,
        vo=experiment.vo,
    )
    s0, _ = real_env.reset()
    trajectory = np.array(s0.x)
    config = real_env.config

    goal = s0.goal

    s = s0
    sim_env.gym_env.state = s0

    obs = [s0.obstacles]
    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    with open(f"debug/TRAJECTORIES/1AG/treajectoryMPC.pkl", "rb") as f:
        trj = pickle.load(f)

    trj = trj[:2].T
    for step_n in range(trj.shape[0]):
        print(f"Step Number {step_n}")
        x = trj[step_n]
        s.x = x

        dist_goal_t1 = dist_to_goal(s0.goal, s.x)
        real_env.dist_goal_t1 = real_env.gym_env.dist_goal_t1 = dist_goal_t1
        collision = real_env.check_collision(s)
        goal = dist_goal_t1 <= real_env.config.robot_radius
        out_boundaries = real_env.check_out_boundaries(s)
        r = real_env.reward_grad(None, None, collision, goal, out_boundaries)

        rewards.append(r)
        gc.collect()

    discount = 0.99
    print(
        round(
            sum(
                np.array(rewards)
                * np.array([discount**e for e in range(len(rewards))])
            ),
            2,
        )
    )
    # with open(f"debug/trajectory_real_nmpc.pkl", "wb") as f:
    #     pickle.dump(trajectory, f)

    # exp_name = "_".join([k + ":" + str(v) for k, v in arguments.__dict__.items()])
    # print_and_notify(
    #     f"Simulation Ended with Reward: {round(sum(rewards), 2)}\n"
    #     f"Discrete: {experiment.discrete}\n"
    #     f"Std Rollout Angle: {experiment.std_angle}\n"
    #     f"Number of Steps: {step_n}\n"
    #     f"Avg Reward Step: {round(sum(rewards) / step_n, 2)}\n"
    #     # f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n"
    #     f"Total Time: {sum(times)}\n" f"Num Simulations: {experiment.n_sim}",
    #     exp_num,
    #     exp_name,
    # )

    # dist_goal = dist_to_goal(s.x[:2], s.goal)
    # reach_goal = dist_goal <= real_env.config.robot_radius
    # discount = 0.99
    # data = {
    #     "cumRwrd": round(sum(rewards), 2),
    #     "discCumRwrd": round(
    #         sum(
    #             np.array(rewards)
    #             * np.array([discount**e for e in range(len(rewards))])
    #         ),
    #         2,
    #     ),
    #     "nSteps": step_n,
    #     "MeanStepTime": np.round(0.8657047554504039, 2),
    #     "StdStepTime": np.round(0.6575892268710086, 2),
    #     "reachGoal": int(reach_goal),
    # }
    # data = data | arguments.__dict__
    # df = pd.Series(data)
    # df.to_csv(f"nmpc.csv")
    #
    # if ANIMATION:
    #     print("Creating Gif...")
    #     fig, ax = plt.subplots()
    #     ani = FuncAnimation(
    #         fig,
    #         plot_frame,
    #         fargs=(goal, config, obs, trajectory, ax),
    #         frames=len(trajectory),
    #         save_count=None,
    #         cache_frame_data=False,
    #     )
    #     ani.save(f"debug/trajectory_nmpc.gif", fps=150)
    #     plt.close(fig)
    #
    # gc.collect()
    # print("Done")


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algorithm", default="vanilla", type=str, help="The algorithm to run"
    )
    parser.add_argument(
        "--nsim",
        default=1000,
        type=int,
        help="The number of simulation the algorithm will run",
    )
    parser.add_argument("--rwrd", default=-100, type=int, help="")
    parser.add_argument("--dt", default=0.2, type=float, help="")
    parser.add_argument("--std", default=0.38 * 2, type=float, help="")
    parser.add_argument("--stdRollout", default=0.5, type=float, help="")
    parser.add_argument("--amplitude", default=1, type=float, help="")
    parser.add_argument("--c", default=1, type=float, help="")
    parser.add_argument("--rollout", default="normal_towards_goal", type=str, help="")
    parser.add_argument("--alpha", default=0.1, type=float, help="")
    parser.add_argument("--k", default=50, type=float, help="")
    parser.add_argument(
        "--a", default=10, type=int, help="number of discretization of angles"
    )
    parser.add_argument(
        "--v", default=10, type=int, help="number of discretization of velocities"
    )
    parser.add_argument(
        "--num", default=1, type=int, help="number of experiments to run"
    )
    parser.add_argument(
        "--eps_rollout",
        default=0.1,
        type=float,
        help="Percentage of Uniform Rollout in Rollout",
    )
    parser.add_argument(
        "--max_depth",
        default=100,
        type=int,
        help="Maximum Depth of the tree",
    )
    parser.add_argument(
        "--env",
        default="EASY",
        type=str,
        help="Environment",
    )
    parser.add_argument(
        "--start",
        default="corner",
        type=str,
        help="Where to start",
    )
    return parser


def get_experiment_data(arguments):
    return ExperimentData(
        discrete=False,
        obstacle_reward=True,
        action_expansion_policy=None,
        rollout_policy=None,
        std_angle=None,
    )


def main():
    global exp_num
    args = argument_parser().parse_args()
    exp = get_experiment_data(args)
    seed_everything(2)
    for exp_num in range(args.num):
        run_experiment(experiment=exp, arguments=args)


if __name__ == "__main__":
    main()
