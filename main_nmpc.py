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
    create_env_five_small_obs_continuous, create_env_four_obs_difficult_continuous,
    create_env_four_obs_difficult_continuous2,
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
    file_name = "results_obst17_agents1_iters1_HARD_0_60.npz"
    # file_name = "results_obst17_agents1_iters1_HARD2_0_60.npz"
    idx = int(file_name.split('.')[0][-1])
    file = np.load(file_name)
    if "HARD" in file_name:
        real_env, sim_env = create_env_four_obs_difficult_continuous(initial_pos=(1, 1),
                                                                     goal=(10, 10),
                                                                     discrete=experiment.discrete,
                                                                     rwrd_in_sim=experiment.obstacle_reward,
                                                                     out_boundaries_rwrd=arguments.rwrd,
                                                                     dt_sim=arguments.dt,
                                                                     n_vel=arguments.v,
                                                                     n_angles=arguments.a,
                                                                     vo=experiment.vo)
    elif "HARD2" in file_name:
        real_env, sim_env = create_env_four_obs_difficult_continuous2(initial_pos=(1, 1),
                                                                      goal=(10, 10),
                                                                      discrete=experiment.discrete,
                                                                      rwrd_in_sim=experiment.obstacle_reward,
                                                                      out_boundaries_rwrd=arguments.rwrd,
                                                                      dt_sim=arguments.dt,
                                                                      n_vel=arguments.v,
                                                                      n_angles=arguments.a,
                                                                      vo=experiment.vo)
    else:
        real_env, sim_env = create_env_five_small_obs_continuous(
            initial_pos=start_pos,
            goal=(10, 10),
            discrete=experiment.discrete,
            rwrd_in_sim=experiment.obstacle_reward,
            out_boundaries_rwrd=arguments.rwrd,
            dt_sim=arguments.dt,
            n_vel=arguments.v,
            n_angles=arguments.a,
            vo=experiment.vo
        )
    s0, _ = real_env.reset()
    trajectory = np.array(s0.x)
    config = real_env.config

    goal_loc = s0.goal

    s = s0
    sim_env.gym_env.state = s0

    obs = [s0.obstacles]
    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    # with open(f"debug/TRAJECTORIES/1AG/treajectoryMPC.pkl", "rb") as f:
    #     trj = pickle.load(f)
    # file = np.load("results_obst6_agents1_iters1_HARD.npz")

    trajectories = np.expand_dims(file["traj"][idx].squeeze(1),0)
    end_trj = file['steps'][0]
    # trj = trj[:2].T
    for i, trj in enumerate(trajectories):
        trj = trj[:end_trj]
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
        disc_ret = round(
                sum(
                    np.array(rewards)
                    * np.array([discount**e for e in range(len(rewards))])
                ),
                2,
            )
        print(
            disc_ret
        )

        dist_goal = dist_to_goal(s.x[:2], s.goal)
        reach_goal = dist_goal <= real_env.config.robot_radius
        discount = 0.99
        data = {
            "cumRwrd": round(sum(rewards), 2),
            "discCumRwrd": round(
                sum(
                    np.array(rewards)
                    * np.array([discount**e for e in range(len(rewards))])
                ),
                2,
            ),
            "nSteps": step_n,
            "MeanStepTime": np.round(np.average(file['dt']), 2),
            "StdStepTime": np.round(np.std(file['dt']), 2),
            "reachGoal": int(reach_goal),
        }
        df = pd.Series(data)
        df.to_csv(f"debug/nmpc_{idx}.csv")
        #
        # if ANIMATION:
        print(f"AVERAGE TIME PER STEP: {np.average(file['dt'])}")
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame,
            fargs=(goal_loc, config, obs, trj, ax),
            frames=len(trj),
            save_count=None,
            cache_frame_data=False,
        )
        ani.save(f"debug/trajectory_nmpc_{idx}.gif", fps=150)
        plt.close(fig)
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
