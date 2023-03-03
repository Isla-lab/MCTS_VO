import os
import random
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import mean, std

from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.utils.utils import uniform_discrete
from bettergym.environments.robot_arena import BetterRobotArena, Config
from utils import print_and_notify, plot_frame, plot_tree_trajectory, plot_action_evolution


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def run_experiment(seed_val, rollout_policy, num_actions):
    # input [forward speed, yaw_rate]
    c = Config()
    c.num_discrete_actions = num_actions
    real_env = BetterRobotArena(
        initial_position=(0, 0),
        gradient=True,
        discrete=True,
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
    # planner = MctsApw(
    #     num_sim=1000,
    #     c=4,
    #     environment=real_env,
    #     computational_budget=100,
    #     k=20,
    #     alpha=0,
    #     discount=0.99,
    #     action_expansion_function=uniform,
    #     rollout_policy=towards_goal
    # )
    planner = Mcts(
        num_sim=1000,
        c=4,
        environment=real_env,
        computational_budget=100,
        discount=0.99,
        rollout_policy=rollout_policy
    )

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

        actions.append(u)
        infos.append(info)

        final_time = time.time() - initial_time
        times.append(final_time)
        s, r, terminal, truncated, env_info = real_env.step(s, u)
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history

    print_and_notify(f"Simulation Ended with Reward: {sum(rewards)}")
    print_and_notify(f"Avg Step Time: {round(mean(times), 2)}±{round(std(times), 2)}")
    print_and_notify(f"Total Time: {sum(times)}")
    print("Creating Gif...")
    fig, ax = plt.subplots()
    ani = FuncAnimation(
        fig,
        plot_frame,
        fargs=(goal, config, trajectory, ax),
        frames=len(trajectory)
    )
    ani.save(f"debug/trajectory_{seed_val}_{rollout_policy.__name__}_actions {num_actions}.gif", dpi=300, fps=150)

    fig2, ax2 = plt.subplots()
    ani_tree_traj = FuncAnimation(
        fig2,
        plot_tree_trajectory,
        fargs=(infos, ax2),
        frames=len(infos)
    )
    ani_tree_traj.save(f"debug/tree_trajectory.mp4", fps=15)
    plot_action_evolution(np.array(actions))

    print("Done")


def main():
    for p, na in [(uniform_discrete, 5), (uniform_discrete, 10)]:
        run_experiment(0, p, na)


if __name__ == '__main__':
    main()
