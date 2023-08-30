import os
import os
import random
import time
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numpy import mean, std

from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import towards_goal, voo, voo_vo
from environment_creator import create_env_multiagent_five_small_obs_continuous
from experiment_utils import print_and_notify, plot_frame_multiagent
from mcts_utils import sample_centered_robot_arena

DEBUG_DATA = True
ANIMATION = True


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    seed_numba(seed_value)


def run_experiment(
    seed_val, num_actions=1, policy=None, discrete=False, var_angle: float = 0.38
):
    global exp_num
    # input [forward speed, yaw_rate]
    (
        real_env_1,
        real_env_2,
        sim_env_1,
        sim_env_2,
    ) = create_env_multiagent_five_small_obs_continuous(
        initial_pos=(1, 1), goal=(10, 10)
    )
    s0_1, _ = real_env_1.reset()
    s0_2, _ = real_env_2.reset()
    seed_everything(seed_val)
    trajectory_1 = np.array(s0_1.x)
    trajectory_2 = np.array(s0_2.x)
    # config is equal
    config = real_env_1.config
    goal_1 = s0_1.goal
    goal_2 = s0_2.goal
    s1 = s0_1
    s2 = s0_2

    if policy.func is voo:
        env1 = real_env_1
        env2 = real_env_2
        for o in s0_1.obstacles:
            o.radius += o.radius * 0.1

        for o in s0_2.obstacles:
            o.radius += o.radius * 0.1
        env1.gym_env.state = s0_1
        env2.gym_env.state = s0_2
    else:
        env1 = sim_env_1
        env2 = sim_env_2

    obs1 = [s0_1.obstacles]
    planner1 = MctsApw(
        num_sim=1000,
        c=150,
        environment=env1,
        computational_budget=100,
        k=50,
        alpha=0.1,
        discount=0.99,
        action_expansion_function=policy,
        rollout_policy=partial(towards_goal, var_angle=var_angle),
    )
    planner2 = MctsApw(
        num_sim=1000,
        c=150,
        environment=env2,
        computational_budget=100,
        k=50,
        alpha=0.1,
        discount=0.99,
        action_expansion_function=policy,
        rollout_policy=partial(towards_goal, var_angle=var_angle),
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

    print_and_notify(
        f"Simulation Ended with Reward: {round(sum(rewards), 2)}\n"
        f"Discrete: {discrete}\n"
        f"Towards Goal Variance: {var_angle}\n"
        f"Number of Steps: {step_n}\n"
        f"Avg Reward Step: {round(sum(rewards) / step_n, 2)}\n"
        f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n"
        f"Total Time: {sum(times)}",
        exp_num,
    )

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame_multiagent,
            fargs=(goal_1, goal_2, config, obs1, trajectory_1, trajectory_2, ax),
            frames=len(trajectory_1),
        )
        ani.save(f"debug/trajectory_{exp_num}.gif", fps=150)
        plt.close()


def main():
    global exp_num
    exp_num = 0
    for p, na, var in [
        (
            partial(voo, eps=0.3, sample_centered=sample_centered_robot_arena),
            1,
            0.38 * 2,
        )
    ]:
        run_experiment(
            seed_val=2, policy=p, num_actions=na, discrete=False, var_angle=var
        )
        exp_num += 1


if __name__ == "__main__":
    main()
