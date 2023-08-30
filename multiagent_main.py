import argparse
import gc
import os
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

from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import towards_goal, voo, epsilon_normal_uniform
from bettergym.agents.utils.vo import towards_goal_vo, uniform_random_vo, epsilon_normal_uniform_vo, \
    sample_centered_robot_arena, voo_vo
from environment_creator import create_env_multiagent_five_small_obs_continuous
from experiment_utils import print_and_notify, plot_frame_multiagent
from mcts_utils import uniform_random

DEBUG_DATA = True
ANIMATION = True


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


def run_experiment(experiment: ExperimentData, arguments):
    global exp_num
    # input [forward speed, yaw_rate]
    real_env_1, real_env_2, sim_env_1, sim_env_2 = create_env_multiagent_five_small_obs_continuous(initial_pos=(1, 1),
                                                                                                   goal=(10, 10),
                                                                                                   discrete=experiment.discrete,
                                                                                                   rwrd_in_sim=experiment.obstacle_reward,
                                                                                                   out_boundaries_rwrd=arguments.rwrd,
                                                                                                   dt_sim=arguments.dt,
                                                                                                   n_vel=arguments.v,
                                                                                                   n_angles=arguments.a)
    s0_1, _ = real_env_1.reset()
    s0_2, _ = real_env_2.reset()
    trajectory_1 = np.array(s0_1.x)
    trajectory_2 = np.array(s0_2.x)
    # config is equal
    config = real_env_1.config
    goal_1 = s0_1.goal
    goal_2 = s0_2.goal
    s1 = s0_1
    s2 = s0_2

    obs1 = [s0_1.obstacles]
    if not experiment.discrete:
        planner1 = MctsApw(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env_1,
            computational_budget=arguments.max_depth,
            k=arguments.k,
            alpha=arguments.alpha,
            discount=0.99,
            action_expansion_function=experiment.action_expansion_policy,
            rollout_policy=experiment.rollout_policy,
        )
        planner2 = MctsApw(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env_2,
            computational_budget=arguments.max_depth,
            k=arguments.k,
            alpha=arguments.alpha,
            discount=0.99,
            action_expansion_function=experiment.action_expansion_policy,
            rollout_policy=experiment.rollout_policy,
        )
    else:
        planner1 = Mcts(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env_1,
            computational_budget=arguments.max_depth,
            discount=0.99,
            rollout_policy=experiment.rollout_policy,
        )
        planner2 = Mcts(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env_2,
            computational_budget=arguments.max_depth,
            discount=0.99,
            rollout_policy=experiment.rollout_policy,
        )

    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    infos = []
    step_n = 0
    while not terminal:
        step_n += 1
        if step_n == 1000:
            break
        print(f"Step Number {step_n}")
        initial_time = time.time()
        u1, info = planner1.plan(s1)
        u2, _ = planner2.plan(s2)

        final_time = time.time() - initial_time
        infos.append(info)
        times.append(final_time)

        s1, r1, terminal1, truncated1, env_info1 = real_env_1.step(s1, u1)
        s2.obstacles[-1] = s1.copy()
        s2, r2, terminal2, truncated2, env_info2 = real_env_2.step(s2, u2)
        s1.obstacles[-1] = s2.copy()

        sim_env_1.gym_env.state = real_env_1.gym_env.state.copy()
        sim_env_2.gym_env.state = real_env_2.gym_env.state.copy()

        rewards.append(r1)
        trajectory_1 = np.vstack((trajectory_1, s1.x))  # store state history
        trajectory_2 = np.vstack((trajectory_2, s2.x))  # store state history
        obs1.append(s1.obstacles)
        terminal = terminal1 or terminal2

    exp_name = '_'.join([k + ':' + str(v) for k, v in arguments.__dict__.items()])
    print_and_notify(
        f"Simulation Ended with Reward: {round(sum(rewards), 2)}\n"
        f"Discrete: {experiment.discrete}\n"
        f"Std Rollout Angle: {experiment.std_angle}\n"
        f"Number of Steps: {step_n}\n"
        f"Avg Reward Step: {round(sum(rewards) / step_n, 2)}\n"
        f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n"
        f"Total Time: {sum(times)}\n"
        f"Num Simulations: {experiment.n_sim}",
        exp_num,
        exp_name
    )

    # dist_goal = dist_to_goal(s.x[:2], s.goal)
    # reach_goal = dist_goal <= real_env.config.robot_radius

    data = {
        "cumRwrd": round(sum(rewards), 2),
        "nSteps": step_n,
        "MeanStepTime": np.round(mean(times), 2),
        "StdStepTime": np.round(std(times), 2),
        # "reachGoal": int(reach_goal)
    }
    data = data | arguments.__dict__
    df = pd.Series(data)
    df.to_csv(f'{exp_name}_{exp_num}.csv')

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame_multiagent,
            fargs=(goal_1, goal_2, config, obs1, trajectory_1, trajectory_2, ax),
            frames=len(trajectory_1)
        )
        ani.save(f"debug/trajectoryMultiagent_{exp_name}_{exp_num}.gif", fps=150)
        plt.close(fig)

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
    # var_angle = 0.38 * 2
    std_angle_rollout = arguments.std

    if arguments.rollout == "normal_towards_goal":
        if arguments.algorithm == "VO2":
            rollout_policy = partial(towards_goal_vo, std_angle_rollout=std_angle_rollout)
        else:
            rollout_policy = partial(towards_goal, std_angle_rollout=std_angle_rollout)
    elif arguments.rollout == "uniform":
        if arguments.algorithm == "VO2":
            rollout_policy = uniform_random_vo
        else:
            rollout_policy = uniform_random
    elif arguments.rollout == "epsilon_normal_uniform":
        if arguments.algorithm == "VO2":
            rollout_policy = partial(
                epsilon_normal_uniform_vo,
                std_angle_rollout=std_angle_rollout,
                eps=arguments.eps_rollout,
            )
        else:
            rollout_policy = partial(
                epsilon_normal_uniform,
                std_angle_rollout=std_angle_rollout,
                eps=arguments.eps_rollout,
            )
    else:
        raise ValueError("rollout function not valid")

    sample_centered = partial(sample_centered_robot_arena, std_angle=std_angle_rollout)
    if arguments.algorithm == "VOR":
        # VORONOI + VO (albero + reward ostacoli)
        return ExperimentData(
            action_expansion_policy=partial(voo_vo, eps=0.3, sample_centered=sample_centered),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c
        )
    elif arguments.algorithm == "VOO":
        # VORONOI
        return ExperimentData(
            action_expansion_policy=partial(voo, eps=0.3, sample_centered=sample_centered),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c
        )
    elif arguments.algorithm == "VANILLA":
        # VANILLA
        return ExperimentData(
            action_expansion_policy=None,
            rollout_policy=rollout_policy,
            discrete=True,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c
        )
    else:
        # VORONOI + VO (albero + rollout)
        return ExperimentData(
            action_expansion_policy=partial(voo_vo, eps=0.3, sample_centered=sample_centered),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=False,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c
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
