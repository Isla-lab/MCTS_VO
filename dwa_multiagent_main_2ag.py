import argparse
import gc
import os
import pickle
import random
import time
from copy import deepcopy
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
from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import towards_goal, voo, epsilon_normal_uniform, epsilon_uniform_uniform
from bettergym.agents.utils.vo import towards_goal_vo, uniform_random_vo, epsilon_normal_uniform_vo, \
    sample_centered_robot_arena, voo_vo, epsilon_uniform_uniform_vo
from bettergym.environments.robot_arena import dist_to_goal
from environment_creator import create_env_multiagent_five_small_obs_continuous
from experiment_utils import print_and_notify, plot_frame_multiagent, plot_frame_tree_traj
from mcts_utils import uniform_random

DEBUG_DATA = False
ANIMATION = True
DEBUG_ANIMATION = True


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    seed_numba(seed_value)


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


def run_experiment(experiment: ExperimentData, arguments):
    global exp_num
    # input [forward speed, yaw_rate]
    real_env_1, real_env_2, \
        sim_env_1, sim_env_2 = create_env_multiagent_five_small_obs_continuous(initial_pos=(1, 1),
                                                                               goal=(10, 10),
                                                                               discrete=experiment.discrete,
                                                                               rwrd_in_sim=experiment.obstacle_reward,
                                                                               out_boundaries_rwrd=arguments.rwrd,
                                                                               dt_sim=arguments.dt,
                                                                               n_vel=arguments.v,
                                                                               n_angles=arguments.a,
                                                                               vo=experiment.vo)
    s0_1, _ = real_env_1.reset()
    s0_2, _ = real_env_2.reset()
    # config is equal
    config = real_env_1.config
    goal_1 = s0_1.goal
    goal_2 = s0_2.goal
    goals = [goal_1, goal_2]
    s1 = s0_1
    s2 = s0_2
    s1.x = np.append(s1.x, 0.0)
    s2.x = np.append(s2.x, 0.0)
    for o in s0_1.obstacles:
        o.x = np.append(o.x, 0.0)
        o.radius *= 1.05
    sim_env_1.gym_env.state = s0_1

    for o in s0_2.obstacles:
        o.x = np.append(o.x, 0.0)
        o.radius *= 1.05
    sim_env_2.gym_env.state = s0_2

    trajectory_1 = np.array(s0_1.x)
    trajectory_2 = np.array(s0_2.x)

    obs = np.array([s.x for s in s0_1.obstacles[:-1]])

    obs1 = [deepcopy(s0_1.obstacles)]
    obs2 = [deepcopy(s0_2.obstacles)]
    planner1 = Dwa(sim_env_1)
    planner2 = Dwa(sim_env_2)

    print("Simulation Started")
    terminal = False
    rewards_1 = []
    rewards_2 = []
    times = []
    infos = [[], []]
    step_n = 0
    terminal1 = False
    terminal2 = False
    while not terminal:
        step_n += 1
        if step_n == 1000:
            break
        print(f"Step Number {step_n}")

        robot_obs_1 = np.array([s0_1.obstacles[-1].x])
        robot_obs_2 = np.array([s0_2.obstacles[-1].x])
        initial_time = time.time()
        u1, info1 = planner1.plan(s1, obs, robot_obs_1)
        final_time = time.time() - initial_time
        times.append(final_time)
        initial_time = time.time()
        u2, info2 = planner2.plan(s2, obs, robot_obs_2)
        final_time = time.time() - initial_time
        times.append(final_time)

        if not terminal1:
            u1_copy = np.array(u1, copy=True)
            u1_copy[1] = s1.x[2] + u1[1] * config.dt
            s1, r1, terminal1, truncated1, env_info1 = real_env_1.step(s1, u1_copy)
            rewards_1.append(r1)
        trajectory_1 = np.vstack(
            (trajectory_1, np.array(s1.x, copy=True))
        )  # store state history
        s1.x[3] = u1[0]
        s1.x[4] = u1[1]
        s1_copy = deepcopy(s1)
        s1_copy.x[2] = 0.0
        s1_copy.x[3] = config.max_speed
        s1_copy.obstacles = None
        s2.obstacles[-1] = s1_copy
        if not terminal2:
            u2_copy = np.array(u2, copy=True)
            u2_copy[1] = s2.x[2] + u2[1] * config.dt
            s2, r2, terminal2, truncated2, env_info2 = real_env_2.step(s2, u2_copy)
            rewards_2.append(r2)
        trajectory_2 = np.vstack(
            (trajectory_2, np.array(s2.x, copy=True))
        )  # store state history
        s2.x[3] = u2[0]
        s2.x[4] = u2[1]

        s2_copy = deepcopy(s2)
        s2_copy.x[2] = 0.0
        s2_copy.x[3] = config.max_speed
        s2_copy.obstacles = None
        s1.obstacles[-1] = s2_copy
        real_env_1.gym_env.state.x = np.array(s1.x, copy=True)
        real_env_2.gym_env.state.x = np.array(s2.x, copy=True)
        sim_env_1.gym_env.state = real_env_1.gym_env.state.copy()
        sim_env_2.gym_env.state = real_env_2.gym_env.state.copy()

        obs1.append(deepcopy(s1.obstacles))
        obs2.append(deepcopy(s2.obstacles))
        terminal = terminal1 and terminal2

    exp_name = '_'.join([k + ':' + str(v) for k, v in arguments.__dict__.items()])
    print_and_notify(
        f"Simulation Ended with Reward1: {round(sum(rewards_1), 2)}\n"
        f"Simulation Ended with Reward2: {round(sum(rewards_2), 2)}\n"
        f"Discrete: {experiment.discrete}\n"
        f"Std Rollout Angle: {experiment.std_angle}\n"
        f"Number of Steps: {step_n}\n"
        f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n"
        f"Total Time: {sum(times)}\n"
        f"Num Simulations: {experiment.n_sim}",
        exp_num,
        exp_name
    )

    dist_goal_1 = dist_to_goal(s1.x[:2], s1.goal)
    dist_goal_2 = dist_to_goal(s2.x[:2], s2.goal)
    reach_goal_1 = dist_goal_1 <= real_env_1.config.robot_radius
    reach_goal_2 = dist_goal_2 <= real_env_2.config.robot_radius
    reach_goal = reach_goal_1 and reach_goal_2
    discount = 0.99
    data = {
        "cumRwrd1": round(sum(rewards_1), 2),
        "cumRwrd2": round(sum(rewards_2), 2),
        "discCumRwrd1": round(sum(np.array(rewards_1) * np.array([discount ** e for e in range(len(rewards_1))])), 2),
        "discCumRwrd2": round(sum(np.array(rewards_2) * np.array([discount ** e for e in range(len(rewards_2))])), 2),
        "nSteps": step_n,
        "MeanStepTime": np.round(mean(times), 2),
        "StdStepTime": np.round(std(times), 2),
        "reachGoal": int(reach_goal)
    }
    data = data | arguments.__dict__
    df = pd.Series(data)
    df.to_csv(f'multiagent2ag.csv')

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame_multiagent,
            fargs=(goal_1, goal_2, config, obs1, trajectory_1, trajectory_2, ax),
            frames=len(trajectory_1)
        )
        ani.save(f"debug/trajectoryMultiagent2ag_dwa.gif", fps=150)
        plt.close(fig)

    with open(f"debug/trajectory_real2ag_dwa.pkl", "wb") as f:
        trajectories = np.array([trajectory_1, trajectory_2])
        pickle.dump(trajectories, f)

    gc.collect()
    print("Done")


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
    return parser


def get_experiment_data(arguments):
    return ExperimentData(
        discrete=False,
        obstacle_reward=True,
        action_expansion_policy=None,
        rollout_policy=None,
        std_angle=None
    )


def main():
    global exp_num
    args = argument_parser().parse_args()
    exp = get_experiment_data(args)
    seed_everything(2)
    for exp_num in range(args.num):
        run_experiment(experiment=exp, arguments=args)


if __name__ == '__main__':
    main()
