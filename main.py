import gc
import os
import random
import time
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numpy import mean, std

from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import voo, towards_goal
from bettergym.agents.utils.vo import sample_centered_robot_arena, voo_vo, towards_goal_vo
from environment_creator import create_env_four_obs_difficult_continuous
from experiment_utils import print_and_notify, plot_frame, plot_real_trajectory_information, \
    create_animation_tree_trajectory

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


def run_experiment(seed_val, num_actions=1, policy=None, discrete=False, var_angle: float = 0.38):
    global exp_num
    # input [forward speed, yaw_rate]
    real_env, sim_env = create_env_four_obs_difficult_continuous(initial_pos=(1, 1), goal=(10, 10))
    # real_env, sim_env = create_env_five_small_obs_continuous(initial_pos=(1, 1), goal=(10, 10))
    s0, _ = real_env.reset()
    seed_everything(seed_val)
    trajectory = np.array(s0.x)
    config = real_env.config

    goal = s0.goal

    s = s0
    # planner = RandomPlanner(real_env)
    if policy.func is voo:
        env = real_env
        for o in s0.obstacles:
            o.radius = o.radius * 1.05
        env.gym_env.state = s0
    else:
        env = real_env

    obs = [s0.obstacles]
    planner_apw = MctsApw(
        num_sim=1000,
        c=150,
        environment=env,
        computational_budget=100,
        k=50,
        alpha=0.1,
        discount=0.99,
        action_expansion_function=policy,
        # rollout_policy=uniform
        rollout_policy=partial(towards_goal_vo, var_angle=var_angle)
    )
    planner_mcts = Mcts(
        num_sim=1000,
        c=150,
        environment=sim_env,
        computational_budget=100,
        discount=0.99,
        rollout_policy=partial(towards_goal, var_angle=var_angle)
    )
    planner = planner_apw if not discrete else planner_mcts

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
        f"Discrete: {discrete}\n"
        f"Towards Goal Variance: {var_angle}\n"
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
        np.savez_compressed(f"debug/trajectories_{exp_num}", *trajectories)
        np.savez_compressed(f"debug/rollout_values_{exp_num}", *rollout_values)
        np.savez_compressed(f"debug/q_values_{exp_num}", *q_vals)
        np.savez_compressed(f"debug/actions_{exp_num}", *a)
        np.savez_compressed(f"debug/trajectory_real_{exp_num}", trajectory)
        np.savez_compressed(f"debug/chosen_a_{exp_num}", np.array(actions))

        print("Creating Tree Trajectories Animation...")
        create_animation_tree_trajectory(goal, config, obs)

    print("Done")


def main():
    global exp_num
    exp_num = 0
    # (partial(voo_vo, eps=0.3, sample_centered=sample_centered_robot_arena), 1, 0.38 * 2)
    for p, na, var in [(partial(voo_vo, eps=0.3, sample_centered=sample_centered_robot_arena), 1, 0.38 * 2)]:
        run_experiment(seed_val=2, policy=p, num_actions=na, discrete=False, var_angle=var)
        exp_num += 1


if __name__ == '__main__':
    main()
