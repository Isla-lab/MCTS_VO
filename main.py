import argparse
import gc
import itertools
import math
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

from bettergym.agents.controller import Controller
from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import voo, towards_goal, uniform_towards_goal, epsilon_normal_uniform
from bettergym.agents.utils.vo import sample_centered_robot_arena, voo_vo, towards_goal_vo, uniform_random_vo, \
    get_relative_velocity, get_spaces, epsilon_normal_uniform_vo
from bettergym.environments.robot_arena import dist_to_goal
from environment_creator import create_env_five_small_obs_continuous, create_env_four_obs_difficult_continuous
from experiment_utils import print_and_notify, plot_frame, create_animation_tree_trajectory
from mcts_utils import uniform_random, get_intersections

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


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    seed_numba(seed_value)


def run_experiment(experiment: ExperimentData, arguments):
    global exp_num
    # input [forward speed, yaw_rate]
    real_env, sim_env = create_env_four_obs_difficult_continuous(initial_pos=(1, 1),
                                                                 goal=(2, 10),
                                                                 discrete=experiment.discrete,
                                                                 rwrd_in_sim=experiment.obstacle_reward,
                                                                 out_boundaries_rwrd=arguments.rwrd,
                                                                 dt_sim=arguments.dt,
                                                                 n_vel=arguments.v,
                                                                 n_angles=arguments.a)

    # real_env, sim_env = create_env_five_small_obs_continuous(initial_pos=(1, 1),
    #                                                          goal=(10, 10),
    #                                                          discrete=experiment.discrete,
    #                                                          rwrd_in_sim=experiment.obstacle_reward,
    #                                                          out_boundaries_rwrd=arguments.rwrd,
    #                                                          dt_sim=arguments.dt,
    #                                                          n_vel=arguments.v,
    #                                                          n_angles=arguments.a)
    s0, _ = real_env.reset()
    trajectory = np.array(s0.x)
    config = real_env.config

    goal = s0.goal

    s = s0

    if experiment.action_expansion_policy is not voo_vo:
        for o in s0.obstacles:
            o.radius *= 1.05
        real_env.gym_env.state = s0
        sim_env.gym_env.state = s0

    obs = [s0.obstacles]
    if not experiment.discrete:
        planner = MctsApw(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env,
            computational_budget=100,
            k=arguments.k,
            alpha=arguments.alpha,
            discount=0.99,
            action_expansion_function=experiment.action_expansion_policy,
            rollout_policy=experiment.rollout_policy
        )
    else:
        planner = Mcts(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env,
            computational_budget=100,
            discount=0.99,
            rollout_policy=experiment.rollout_policy
        )
    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    infos = []
    actions = []
    step_n = 0
    controller = Controller()
    while not terminal:
        step_n += 1
        if step_n == 1000:
            break
        print(f"Step Number {step_n}")
        initial_time = time.time()
        # if step_n == 123:
        #     print(s.x)
        #     get_random_state()
        u, info = planner.plan(s)
        # del info['q_values']
        # del info['actions']
        # del info['visits']
        # gc.collect()

        actions.append(u)

        # Clip action
        u_copy = np.array(u, copy=True)
        # if arguments.algorithm in ["VOR", "VO2"]:
        u_copy = controller.plan(x=s.x, config=config, action=u_copy)
        #     # Extract obstacle information
        #     obstacles = s.obstacles
        #     obs_x = np.array([ob.x for ob in obstacles])
        #     obs_rad = np.array([ob.radius for ob in obstacles])
        #
        #     # Extract robot information
        #     x = s.x
        #     dt = config.dt
        #     ROBOT_RADIUS = config.robot_radius
        #     VMAX = 0.3
        #
        #     # Calculate velocities
        #     v = get_relative_velocity(VMAX, obs_x, x)
        #
        #     # Calculate radii
        #     r0 = np.linalg.norm(v, axis=1) * dt
        #     r1 = ROBOT_RADIUS + (obs_rad * 1.05)
        #     # increment by 5 percent
        #     # r1 *= 1.05
        #
        #     # Calculate intersection points
        #     intersection_points = [get_intersections(x[:2], obs_x[i][:2], r0[i], r1[i]) for i in range(len(obs_x))]
        #
        #     # If there are no intersection points
        #     if not any(intersection_points):
        #         u_copy[1] = max(min(u_copy[1], max_angle), min_angle)
        #         u_copy[1] = (u_copy[1] + math.pi) % (2 * math.pi) - math.pi
        #     else:
        #         # convert intersection points into ranges of available velocities/angles
        #         angle_space, _ = get_spaces(intersection_points, x, obs_x, r1, config)
        #         flag = False
        #         for aspace in angle_space:
        #             if aspace[0] <= u_copy[1] <= aspace[1]:
        #                 flag = True
        #         if not flag:
        #             angle_space = list(itertools.chain(*angle_space))
        #             u_copy[1] = min(angle_space, key=lambda e: abs(e - u_copy[1]))
        # else:
        #     u_copy[1] = max(min(u_copy[1], max_angle), min_angle)
        #     u_copy[1] = (u_copy[1] + math.pi) % (2 * math.pi) - math.pi
        final_time = time.time() - initial_time
        infos.append(info)

        times.append(final_time)
        s, r, terminal, truncated, env_info = real_env.step(s, u_copy)
        sim_env.gym_env.state = real_env.gym_env.state.copy()
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history
        obs.append(s.obstacles)
        gc.collect()

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

    dist_goal = dist_to_goal(s.x[:2], s.goal)
    reach_goal = dist_goal <= real_env.config.robot_radius

    data = {
        "cumRwrd": round(sum(rewards), 2),
        "nSteps": step_n,
        "MeanStepTime": np.round(mean(times), 2),
        "StdStepTime": np.round(std(times), 2),
        "reachGoal": int(reach_goal)
    }
    data = data | arguments.__dict__
    df = pd.Series(data)
    df.to_csv(f'{exp_name}_{exp_num}.csv')

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame,
            fargs=(goal, config, obs, trajectory, ax),
            frames=len(trajectory),
            save_count=None,
            cache_frame_data=False
        )
        ani.save(f"debug/trajectory_{exp_name}_{exp_num}.gif", fps=150)
        plt.close(fig)

    trajectories = [i["trajectories"] for i in infos]
    rollout_values = [i["rollout_values"] for i in infos]
    if DEBUG_DATA:
        print("Saving Debug Data...")
        q_vals = [i["q_values"] for i in infos]
        visits = [i["visits"] for i in infos]
        a = [[an.action for an in i["actions"]] for i in infos]
        with open(f"debug/trajectories_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(trajectories, f)
        with open(f"debug/rollout_values_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(rollout_values, f)
        with open(f"debug/visits_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(visits, f)
        with open(f"debug/q_values_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(q_vals, f)
        with open(f"debug/actions_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(a, f)
        with open(f"debug/trajectory_real_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(trajectory, f)
        with open(f"debug/chosen_a_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(actions, f)

    if DEBUG_ANIMATION:
        print("Creating Tree Trajectories Animation...")
        create_animation_tree_trajectory(goal, config, obs, exp_num, exp_name, rollout_values, trajectories)
        # create_animation_tree_trajectory_w_steps(goal, config, obs, exp_num)
    gc.collect()
    print("Done")


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithm', default="vanilla", type=str, help='The algorithm to run')
    parser.add_argument('--nsim', default=1000, type=int, help='The number of simulation the algorithm will run')
    parser.add_argument('--rwrd', default=-100, type=int, help='')
    parser.add_argument('--dt', default=0.2, type=float, help='')
    parser.add_argument('--std', default=0.38 * 2, type=float, help='')
    parser.add_argument('--amplitude', default=1, type=float, help='')
    parser.add_argument('--c', default=1, type=float, help='')
    parser.add_argument('--rollout', default="normal_towards_goal", type=str, help='')
    parser.add_argument('--alpha', default=0.1, type=float, help='')
    parser.add_argument('--k', default=50, type=float, help='')
    parser.add_argument('--a', default=10, type=int, help='number of discretization of angles')
    parser.add_argument('--v', default=10, type=int, help='number of discretization of velocities')
    parser.add_argument('--num', default=1, type=int, help='number of experiments to run')

    return parser


def get_experiment_data(arguments):
    # var_angle = 0.38 * 2
    std_angle_rollout = arguments.std

    if arguments.rollout == "normal_towards_goal":
        if arguments.algorithm == "VO2":
            rollout_policy = partial(towards_goal_vo, std_angle_rollout=std_angle_rollout)
        else:
            rollout_policy = partial(towards_goal, std_angle_rollout=std_angle_rollout)
    # elif arguments.rollout == "uniform_towards_goal":
    #     rollout_policy = partial(uniform_towards_goal, amplitude=math.radians(arguments.amplitude))
    elif arguments.rollout == "uniform":
        if arguments.algorithm == "VO2":
            rollout_policy = uniform_random_vo
        else:
            rollout_policy = uniform_random
    elif arguments.rollout == "epsilon_normal_uniform":
        if arguments.algorithm == "VO2":
            rollout_policy = partial(epsilon_normal_uniform_vo, std_angle_rollout=std_angle_rollout)
        else:
            rollout_policy = partial(epsilon_normal_uniform, std_angle_rollout=std_angle_rollout)
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
