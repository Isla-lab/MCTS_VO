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
from environment_creator import create_env_4ag
from experiment_utils import print_and_notify, plot_frame_no_obs, plot_frame_tree_traj
from mcts_utils import uniform_random

# DEBUG_DATA = True
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
    real_envs, sim_envs = create_env_4ag(discrete=experiment.discrete, rwrd_in_sim=experiment.obstacle_reward,
                                         out_boundaries_rwrd=arguments.rwrd, dt_sim=arguments.dt, n_angles=arguments.a,
                                         n_vel=arguments.v, vo=experiment.vo, )
    initial_states = [env.reset()[0] for env in real_envs]

    for idx, s0 in enumerate(initial_states):
        s0.x = np.append(s0.x, 0.0)
        for o in s0.obstacles:
            o.radius *= 1.05
        sim_envs[idx].gym_env.state = s0

    states = deepcopy(initial_states)
    trajectories = [np.array(s.x) for s in initial_states]
    # config is equal
    config = real_envs[0].config
    goals = [s.goal for s in initial_states]
    obs = [[s.obstacles] for s in initial_states]

    planners = [
        Dwa(sim_env)for sim_env in sim_envs
    ]

    print("Simulation Started")
    terminal = False
    rewards = [[] for _ in range(len(sim_envs))]
    infos = [[] for _ in range(len(sim_envs))]
    times = []
    step_n = 0
    terminals = [False for _ in range(len(sim_envs))]
    ob = []
    while not terminal:
        step_n += 1
        if step_n == 1000:
            break
        print(f"Step Number {step_n}")

        for i in range(len(states)):
            print(f"Agent {i}")
            robot_ob = np.array([ob.x[:2] for ob in states[i].obstacles])
            initial_time = time.time()
            chosen_action, info = planners[i].plan(states[i], ob, robot_ob)
            final_time = time.time() - initial_time
            times.append(final_time)
            if not terminals[i]:
                chosen_action_copy = np.array(chosen_action, copy=True)
                chosen_action_copy[1] = states[i].x[2] + chosen_action[1] * config.dt
                states[i], r, terminals[i], truncated, env_info = real_envs[i].step(states[i], chosen_action_copy)
                rewards[i].append(r)
            trajectories[i] = np.vstack((trajectories[i], states[i].x))  # store state history
            states[i].x[3] = chosen_action[0]
            states[i].x[4] = chosen_action[1]
            real_envs[i].gym_env.state.x = np.array(states[i].x, copy=True)
            sim_envs[i].gym_env.state = real_envs[i].gym_env.state.copy()
            s_copy = deepcopy(states[i])
            s_copy.x[2] = 0.0
            s_copy.x[3] = config.max_speed
            s_copy.obstacles = None
            # update other states obstacles
            for j in range(len(states)):
                for ob_idx, ob in enumerate(states[j].obstacles):
                    if np.array_equal(ob.goal, s_copy.goal):
                        states[j].obstacles[ob_idx] = s_copy
                        break

            obs[i].append(deepcopy(states[i].obstacles))
            terminal = all(terminals)

    with open(f"debug/trajectory_real4ag_dwa.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    exp_name = '_'.join([k + ':' + str(v) for k, v in arguments.__dict__.items()])
    print_and_notify(
        f"Discrete: {experiment.discrete}\n"
        f"Std Rollout Angle: {experiment.std_angle}\n"
        f"Number of Steps: {step_n}\n"
        f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n"
        f"Total Time: {sum(times)}\n"
        f"Num Simulations: {experiment.n_sim}",
        exp_num,
        exp_name
    )
    discount = 0.99
    dist_goal = [dist_to_goal(s.x[:2], s.goal) for s in states]
    reach_goal = all([d <= real_envs[0].config.robot_radius for d in dist_goal])
    cum_rwrd_dict = {f"cumRwrd{i}": round(sum(rewards[i]), 2) for i in range(len(rewards))}
    disc_cum_rwrd_dict = {
        f"discCumRwrd{i}": round(sum(np.array(rewards[i]) * np.array([discount ** e for e in range(len(rewards[i]))])),
                                 2) for i in range(len(rewards))}
    data = {
        **cum_rwrd_dict,
        **disc_cum_rwrd_dict,
        "nSteps": step_n,
        "MeanStepTime": np.round(mean(times), 2),
        "StdStepTime": np.round(std(times), 2),
        "reachGoal": int(reach_goal)
    }
    data = data | arguments.__dict__
    df = pd.Series(data)
    df.to_csv(f'multiagent4ag_dwa.csv')

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame_no_obs,
            fargs=(goals, config, trajectories, ax),
            frames=len(trajectories[0])
        )
        ani.save(f"debug/trajectoryMultiagent4ag_dwa.gif", fps=150)
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
