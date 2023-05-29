import gc
import os
import pickle
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numpy import mean, std

from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import voo, towards_goal
from bettergym.agents.utils.vo import sample_centered_robot_arena, towards_goal_vo, voo_vo
from environment_creator import create_env_four_obs_difficult_continuous, create_env_five_small_obs_continuous
from experiment_utils import print_and_notify, plot_frame, plot_real_trajectory_information, \
    create_animation_tree_trajectory

DEBUG_DATA = True
ANIMATION = True


@dataclass(frozen=True)
class ExperimentData:
    rollout_policy: Callable
    action_expansion_policy: Callable
    discrete: bool
    obstacle_reward: bool
    variance: float


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    seed_numba(seed_value)


def run_experiment(seed_val, experiment: ExperimentData):
    global exp_num
    # input [forward speed, yaw_rate]
    # real_env, sim_env = create_env_four_obs_difficult_continuous(initial_pos=(1, 1), goal=(10, 10),
    #                                                              discrete=experiment.discrete)
    real_env, sim_env = create_env_five_small_obs_continuous(initial_pos=(1, 1), goal=(10, 10),
                                                             discrete=experiment.discrete, rwrd_in_sim=experiment.obstacle_reward)
    s0, _ = real_env.reset()
    seed_everything(seed_val)
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
    planner_apw = MctsApw(
        num_sim=1000,
        c=150,
        environment=sim_env,
        computational_budget=100,
        k=50,
        alpha=0.1,
        discount=0.99,
        action_expansion_function=experiment.action_expansion_policy,
        rollout_policy=experiment.rollout_policy
    )
    planner_mcts = Mcts(
        num_sim=1000,
        c=150,
        environment=sim_env,
        computational_budget=100,
        discount=0.99,
        rollout_policy=experiment.rollout_policy
    )
    planner = planner_apw if not experiment.discrete else planner_mcts

    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    infos = []
    actions = []
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
        s, r, terminal, truncated, env_info = real_env.step(s, u)
        sim_env.gym_env.state = real_env.gym_env.state.copy()
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history
        obs.append(s.obstacles)
        gc.collect()

    print_and_notify(
        f"Simulation Ended with Reward: {round(sum(rewards), 2)}\n"
        f"Discrete: {experiment.discrete}\n"
        f"Variance Angle: {experiment.variance}\n"
        f"Number of Steps: {step_n}\n"
        f"Avg Reward Step: {round(sum(rewards) / step_n, 2)}\n"
        f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n"
        f"Total Time: {sum(times)}",
        exp_num
    )

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame,
            fargs=(goal, config, obs, trajectory, ax),
            frames=len(trajectory)
        )
        ani.save(f"debug/trajectory_{exp_num}.gif", fps=150)
        plot_real_trajectory_information(trajectory, exp_num)
        plt.close()

    if DEBUG_DATA:
        trajectories = [i["trajectories"] for i in infos]
        rollout_values = [i["rollout_values"] for i in infos]

        print("Saving Debug Data...")
        q_vals = [i["q_values"] for i in infos]
        a = [[an.action for an in i["actions"]] for i in infos]
        with open(f"debug/trajectories_{exp_num}.pkl", 'wb') as f:
            pickle.dump(trajectories, f)
        with open(f"debug/rollout_values_{exp_num}.pkl", 'wb') as f:
            pickle.dump(rollout_values, f)
        with open(f"debug/q_values_{exp_num}.pkl", 'wb') as f:
            pickle.dump(q_vals, f)
        with open(f"debug/actions_{exp_num}.pkl", 'wb') as f:
            pickle.dump(a, f)
        with open(f"debug/trajectory_real_{exp_num}.pkl", 'wb') as f:
            pickle.dump(trajectory, f)
        with open(f"debug/chosen_a_{exp_num}.pkl", 'wb') as f:
            pickle.dump(actions, f)

        time.sleep(1)
        print("Creating Tree Trajectories Animation...")
        create_animation_tree_trajectory(goal, config, obs, exp_num)

    print("Done")


def main():
    global exp_num
    exp_num = 0
    var_angle = 0.38 * 2
    experiments = [
        # VORONOI + VO (albero + reward ostacoli)
        ExperimentData(
            action_expansion_policy=partial(voo_vo, eps=0.3, sample_centered=sample_centered_robot_arena),
            rollout_policy=partial(towards_goal, var_angle=var_angle),
            discrete=False,
            obstacle_reward=True,
            variance=0.38 * 2
        ),
        # VORONOI + VO (albero)
        ExperimentData(
            action_expansion_policy=partial(voo_vo, eps=0.3, sample_centered=sample_centered_robot_arena),
            rollout_policy=partial(towards_goal, var_angle=var_angle),
            discrete=False,
            obstacle_reward=False,
            variance=0.38 * 2
        ),
        # VORONOI
        ExperimentData(
            action_expansion_policy=partial(voo, eps=0.3, sample_centered=sample_centered_robot_arena),
            rollout_policy=partial(towards_goal, var_angle=var_angle),
            discrete=False,
            obstacle_reward=True,
            variance=0.38 * 2
        ),
        # VANILLA
        ExperimentData(
            action_expansion_policy=None,
            rollout_policy=partial(towards_goal, var_angle=var_angle),
            discrete=True,
            obstacle_reward=True,
            variance=0.38 * 2
        ),
        # VORONOI + VO (albero + rollout)
        ExperimentData(
            action_expansion_policy=partial(voo_vo, eps=0.3, sample_centered=sample_centered_robot_arena),
            rollout_policy=partial(towards_goal_vo, var_angle=var_angle),
            discrete=False,
            obstacle_reward=False,
            variance=0.38 * 2
        )
    ]
    for exp in experiments:
        run_experiment(seed_val=2, experiment=exp)
        exp_num += 1


if __name__ == '__main__':
    main()
