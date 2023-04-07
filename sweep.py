import argparse
import os
import random
import time
from functools import partial
from statistics import mean

import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numpy import std

from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import uniform, uniform_discrete, voo, towards_goal, voo_vo
from bettergym.environments.robot_arena import Config, BetterRobotArena
from mcts_utils import sample_centered_robot_arena
from utils import plot_frame


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithm', default="vanilla", type=str, help='The algorithm to run')
    parser.add_argument('--nsim', default=1000, type=int, help='The number of simulation the algorithm will run')
    parser.add_argument('--c', default=1, type=float, help='exploration-exploitation factor')
    parser.add_argument('--ae', default="random", type=str, help='the function to select actions to add to the tree')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--rollout', default='random', type=str, help='the function to select actions during rollout')
    parser.add_argument('--max_depth', default=100, type=int, help='max number of steps during rollout')
    parser.add_argument('--n_actions', default=10, type=int, help='number of actions for vanilla mcts')
    parser.add_argument('--alpha1', default=0, type=float, help='alpha')
    parser.add_argument('--alpha2', default=0.1, type=float, help='alpha2')
    parser.add_argument('--k1', default=1, type=int, help='k1')
    parser.add_argument('--k2', default=1, type=int, help='k2')
    parser.add_argument('--epsilon', default=0.3, type=float, help='epsilon value for epsilon greedy strategies')
    parser.add_argument('--var_angle_towards_goal', default=0.38, type=float,
                        help='Variance of the towards goal rollout')
    parser.add_argument('--discrete', default=False, type=bool,
                        help='If the env is discrete or not')

    return parser


def get_function(function_name):
    dict_args = args.__dict__
    functions = {
        "random": uniform if dict_args["algorithm"] != "vanilla" else uniform_discrete,
        "voo": partial(
            voo,
            eps=dict_args["epsilon"],
            sample_centered=sample_centered_robot_arena
        ),
        "voo_vo": partial(
            voo_vo,
            eps=dict_args["epsilon"],
            sample_centered=sample_centered_robot_arena
        ),
        "towards_goal": partial(towards_goal, var_angle=dict_args["var_angle_towards_goal"])
    }
    return functions[function_name]


def get_planner(env):
    dict_args = args.__dict__
    agent_name = dict_args["algorithm"]
    match agent_name:
        case "vanilla":
            return Mcts(
                num_sim=dict_args["nsim"],
                c=dict_args["c"],
                environment=env,
                computational_budget=dict_args["max_depth"],
                rollout_policy=get_function(dict_args["rollout"]),
                discount=dict_args["gamma"]
            )
        case "apw":
            return MctsApw(
                num_sim=dict_args["nsim"],
                c=dict_args["c"],
                environment=env,
                computational_budget=dict_args["max_depth"],
                k=dict_args["k1"],
                alpha=dict_args["alpha1"],
                action_expansion_function=get_function(dict_args["ae"]),
                rollout_policy=get_function(dict_args["rollout"]),
                discount=dict_args["gamma"]
            )


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    seed_numba(seed_value)


def run_experiment(seed_val):
    dict_args = args.__dict__
    run = wandb.init(config=dict_args, entity="lorenzobonanni", project="robotArena", reinit=True)
    # input [forward speed, yaw_rate]
    c = Config()
    c.num_discrete_actions = dict_args["n_actions"]
    real_env = BetterRobotArena(
        initial_position=(1, 1),
        gradient=True,
        discrete=dict_args["discrete"],
        config=c
    )
    s0, _ = real_env.reset()
    seed_everything(seed_val)
    trajectory = np.array(s0.x)
    config = real_env.config
    infos = []
    actions = []

    goal = s0.goal

    s = s0
    planner = get_planner(real_env)

    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    step_n = 0
    while not terminal:
        step_n += 1
        if step_n == 1000:
            break
        print(f"Step Number {step_n}")
        initial_time = time.time()
        u, info = planner.plan(s)

        final_time = time.time() - initial_time
        actions.append(u)
        infos.append(info)

        times.append(final_time)
        wandb.log(
            {
                "Angle": u[1],
                "Linear Velocity": u[0],
                "x": s.x[0],
                "y": s.x[1]
            }
        )
        s, r, terminal, truncated, env_info = real_env.step(s, u)
        wandb.log(
            {"Reward": r}
        )
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history

    print(
        f"Simulation Ended with Reward: {round(sum(rewards), 2)}\n" +
        f"Number of Steps: {step_n}\n" +
        f"Avg Reward Step: {round(sum(rewards) / step_n, 2)}\n"
        f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n" +
        f"Total Time: {sum(times)}",
    )
    wandb.log(
        {
            "cumulativeReward": round(sum(rewards), 2),
            "Steps": step_n,
            "Avg Step Time": np.round(mean(times), 2),
            "Std Step Time": np.round(std(times), 2)
        }
    )

    print("Creating Gif...")
    fig, ax = plt.subplots()

    ani = FuncAnimation(
        fig,
        plot_frame,
        fargs=(goal, config, trajectory, ax),
        frames=len(trajectory)
    )
    ani.save(f"debug/trajectory_{run.entity}.gif", fps=150)

    # if DEBUG_DATA:
    #     print("Saving Debug Data...")
    #     trajectories = [i["trajectories"] for i in infos]
    #     rollout_values = [i["rollout_values"] for i in infos]
    #     q_vals = [i["q_values"] for i in infos]
    #     a = [[an.action for an in i["actions"]] for i in infos]
    #     np.savez_compressed(f"debug/trajectories_{exp_num}", *trajectories)
    #     np.savez_compressed(f"debug/rollout_values_{exp_num}", *rollout_values)
    #     np.savez_compressed(f"debug/q_values_{exp_num}", *q_vals)
    #     np.savez_compressed(f"debug/actions_{exp_num}", *a)
    #     np.savez_compressed(f"debug/chosen_a_{exp_num}", np.array(actions))

    print("Done")


if __name__ == '__main__':
    global args
    SEED = 5
    args = argument_parser().parse_args()
    run_experiment(SEED)
