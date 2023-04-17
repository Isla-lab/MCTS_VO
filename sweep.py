import argparse
import gc
import os
import random
import time
from functools import partial
from statistics import mean

import numpy as np
from numba import njit
from numpy import std

import wandb
from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import uniform, uniform_discrete, voo, towards_goal, voo_vo
from main import create_env_four_obs_difficult_continuous
from mcts_utils import sample_centered_robot_arena


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
    wandb.init(config=dict_args, entity="lorenzobonanni", project="robotArena", reinit=True)
    # input [forward speed, angle]
    real_env, sim_env = create_env_four_obs_difficult_continuous(initial_pos=(1, 1), goal=(10, 10))
    s0, _ = real_env.reset()
    seed_everything(seed_val)

    s = s0
    planner = get_planner(sim_env)

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
        # FREE Memory
        del planner.id_to_state_node
        del planner.last_id
        del planner.info
        final_time = time.time() - initial_time

        wandb.log(
            {
                "Angle": u[1],
                "Linear Velocity": u[0],
                "x": s.x[0],
                "y": s.x[1]
            }
        )
        s, r, terminal, truncated, env_info = real_env.step(s, u)
        sim_env.gym_env.state = real_env.gym_env.state
        wandb.log(
            {"Reward": r}
        )
        rewards.append(r)
        times.append(final_time)
        gc.collect()

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

    print("Done")


if __name__ == '__main__':
    global args
    SEED = 5
    args = argument_parser().parse_args()
    run_experiment(SEED)
