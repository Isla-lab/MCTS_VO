import argparse
import gc
import os
import pickle
import random
import sys
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
from tqdm import tqdm

from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import (
    voo,
    towards_goal,
    epsilon_normal_uniform, epsilon_uniform_uniform,
)
from bettergym.agents.utils.vo import (
    sample_centered_robot_arena,
    voo_vo,
    towards_goal_vo,
    uniform_random_vo,
    epsilon_normal_uniform_vo,
    epsilon_uniform_uniform_vo,
)
from bettergym.environments.robot_arena import dist_to_goal
from environment_creator import (
    create_pedestrian_env,
)
from experiment_utils import (
    plot_frame2,
    print_and_notify,
    create_animation_tree_trajectory,
)
from mcts_utils import uniform_random

DEBUG_DATA = True
DEBUG_ANIMATION = True
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

    real_env, sim_env = create_pedestrian_env(
        discrete=experiment.discrete,
        rwrd_in_sim=experiment.obstacle_reward,
        out_boundaries_rwrd=arguments.rwrd,
        n_vel=arguments.v,
        n_angles=arguments.a,
        vo=experiment.vo
    )

    s0, _ = real_env.reset()
    trajectory = np.array(s0.x)
    config = real_env.config

    goal = s0.goal

    s = s0

    obs = [s0.obstacles]
    if not experiment.discrete:
        planner = MctsApw(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env,
            computational_budget=arguments.max_depth,
            k=arguments.k,
            alpha=arguments.alpha,
            discount=0.99,
            action_expansion_function=experiment.action_expansion_policy,
            rollout_policy=experiment.rollout_policy,
        )
    else:
        planner = Mcts(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env,
            computational_budget=arguments.max_depth,
            discount=0.99,
            rollout_policy=experiment.rollout_policy,
        )
    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    infos = []
    actions = []
    step_n = 0
    while not terminal:
        step_n += 1
        if step_n == 5:
            break
        print(f"Step Number {step_n}")
        initial_time = time.time()
        u, info = planner.plan(s)
        # del info['q_values']
        # del info['actions']
        # del info['visits']
        # gc.collect()

        actions.append(u)

        # Clip action
        u_copy = np.array(u, copy=True)
        final_time = time.time() - initial_time
        infos.append(info)

        times.append(final_time)
        s, r, terminal, truncated, env_info = real_env.step(s, u_copy)
        sim_env.gym_env.state = real_env.gym_env.state.copy()
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history
        obs.append(s.obstacles)
        gc.collect()

    exp_name = "_".join([k + ":" + str(v) for k, v in arguments.__dict__.items()])
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
        exp_name,
    )

    dist_goal = dist_to_goal(s.x[:2], s.goal)
    reach_goal = dist_goal <= real_env.config.robot_radius
    discount = 0.99
    data = {
        "cumRwrd": round(sum(rewards), 2),
        "discCumRwrd": round(sum(np.array(rewards) * np.array([discount ** e for e in range(len(rewards))])), 2),
        "nSteps": step_n,
        "MeanStepTime": np.round(mean(times), 2),
        "StdStepTime": np.round(std(times), 2),
        "reachGoal": int(reach_goal),
        "meanSmoothVelocity": np.diff(trajectory[:, 3]).mean(),
        "stdSmoothVelocity": np.diff(trajectory[:, 3]).std(),
        "meanSmoothAngle": np.diff(trajectory[:, 2]).mean(),
        "stdSmoothAngle": np.diff(trajectory[:, 2]).std(),
    }
    data = data | arguments.__dict__
    df = pd.Series(data)
    df.to_csv(f"{exp_name}_{exp_num}.csv")

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame2,
            fargs=(goal, config, obs, trajectory, ax),
            frames=tqdm(range(len(trajectory)), file=sys.stdout),
            save_count=None,
            cache_frame_data=False,
        )
        ani.save(f"debug/trajectory_{exp_name}_{exp_num}.gif", fps=150)
        plt.close(fig)

    trajectories = [i["trajectories"] for i in infos]
    rollout_values = [i["rollout_values"] for i in infos]

    with open(f"debug/trajectory_real_{exp_name}_{exp_num}.pkl", "wb") as f:
        pickle.dump(trajectory, f)

    if DEBUG_DATA:
        print("Saving Debug Data...")
        q_vals = [i["q_values"] for i in infos]
        visits = [i["visits"] for i in infos]
        a = [[an.action for an in i["actions"]] for i in infos]
        with open(f"debug/trajectories_{exp_name}_{exp_num}.pkl", "wb") as f:
            pickle.dump(trajectories, f)
        with open(f"debug/rollout_values_{exp_name}_{exp_num}.pkl", "wb") as f:
            pickle.dump(rollout_values, f)
        with open(f"debug/visits_{exp_name}_{exp_num}.pkl", "wb") as f:
            pickle.dump(visits, f)
        with open(f"debug/q_values_{exp_name}_{exp_num}.pkl", "wb") as f:
            pickle.dump(q_vals, f)
        with open(f"debug/actions_{exp_name}_{exp_num}.pkl", "wb") as f:
            pickle.dump(a, f)
        with open(f"debug/chosen_a_{exp_name}_{exp_num}.pkl", "wb") as f:
            pickle.dump(actions, f)

    if DEBUG_ANIMATION:
        print("Creating Tree Trajectories Animation...")
        create_animation_tree_trajectory(
            goal, config, obs, exp_num, exp_name, rollout_values, trajectories
        )
        # create_animation_tree_trajectory_w_steps(goal, config, obs, exp_num)
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
    parser.add_argument(
        "--start",
        default="corner",
        type=str,
        help="Where to start",
    )
    return parser


def get_experiment_data(arguments):
    # var_angle = 0.38 * 2
    std_angle_rollout = arguments.stdRollout

    if arguments.rollout == "normal_towards_goal":
        if arguments.algorithm == "VO2":
            rollout_policy = partial(
                towards_goal_vo, std_angle_rollout=std_angle_rollout,
            )
        else:
            rollout_policy = partial(towards_goal, std_angle_rollout=std_angle_rollout)
    # elif arguments.rollout == "uniform_towards_goal":
    #     rollout_policy = partial(uniform_towards_goal, amplitude=math.radians(arguments.amplitude))
    elif arguments.rollout == "uniform":
        if arguments.algorithm == "VO2":
            rollout_policy = uniform_random_vo
        else:
            rollout_policy = uniform_random
    elif arguments.rollout == "epsilon_uniform_uniform":
        if arguments.algorithm == "VO2" or arguments.algorithm == "VANILLA_VO2" or arguments.algorithm == "VANILLA_VO_ROLLOUT":
            rollout_policy = partial(
                epsilon_uniform_uniform_vo,
                std_angle_rollout=std_angle_rollout,
                eps=arguments.eps_rollout,
            )
        else:
            rollout_policy = partial(
                epsilon_uniform_uniform,
                std_angle_rollout=std_angle_rollout,
                eps=arguments.eps_rollout,
            )
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

    sample_centered = partial(sample_centered_robot_arena, std_angle=arguments.std)
    if arguments.algorithm == "VOR":
        # VORONOI + VO (albero + reward ostacoli)
        return ExperimentData(
            action_expansion_policy=partial(
                voo_vo, eps=0.3, sample_centered=sample_centered
            ),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
        )
    elif arguments.algorithm == "VOO":
        # VORONOI
        return ExperimentData(
            action_expansion_policy=partial(
                voo,
                eps=0.3,
                sample_centered=sample_centered,
            ),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
        )
    elif arguments.algorithm == "VANILLA" or arguments.algorithm == "VANILLA_VO_ROLLOUT":
        # VANILLA
        return ExperimentData(
            action_expansion_policy=None,
            rollout_policy=rollout_policy,
            discrete=True,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
        )
    elif arguments.algorithm == "VANILLA_VO2" or arguments.algorithm == "VANILLA_VO_ALBERO":
        # VANILLA
        return ExperimentData(
            vo=True,
            action_expansion_policy=None,
            rollout_policy=rollout_policy,
            discrete=True,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
        )
    else:
        # VORONOI + VO (albero + rollout)
        return ExperimentData(
            action_expansion_policy=partial(
                voo_vo,
                eps=0.3,
                sample_centered=sample_centered,
            ),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=False,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c,
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
