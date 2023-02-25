import os
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.planner_random import RandomPlanner
from bettergym.agents.utils.action_expansion_functions import uniform
from bettergym.better_gym import BetterGym
from bettergym.environments.robot_arena import RobotArena, BetterRobotArena


def seed_everything(real_env: BetterGym, seed_value: int):
    real_env.seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def plot_robot(x, y, yaw, config, ax):
    circle = plt.Circle((x, y), config.robot_radius, color="b")
    ax.add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
    ax.plot([x, out_x], [y, out_y], "-k")


def plot_frame(i, goal, config, traj, ax):
    x = traj[i, :]
    ob = config.ob
    ax.clear()
    # ROBOT POSITION
    plt.plot(x[0], x[1], "xr")
    # GOAL POSITION
    ax.plot(goal[0], goal[1], "xb")
    # OBSTACLES
    ax.plot(ob[:, 0], ob[:, 1], "ok")
    # BOX AROUND ROBOT
    plot_robot(x[0], x[1], x[2], config, ax)
    # TRAJECTORY
    sub_traj = traj[:i]
    ax.plot(sub_traj[:, 0], sub_traj[:, 1], "--r")

    # ax.plot([70, 70], [100, 250], 'k-', lw=2)

    ax.set_xlim([config.left_limit, config.right_limit])
    ax.set_ylim([config.bottom_limit, config.upper_limit])
    # ax.axis("equal")
    ax.grid(True)


def main():
    # input [forward speed, yaw_rate]
    real_env = BetterRobotArena((0, 0))
    s0, _ = real_env.reset()
    seed_everything(real_env, 1)
    trajectory = np.array(s0.x)
    config = real_env.config
    goal = s0.goal

    s = s0
    planner = MctsApw(
        num_sim=1000,
        c=0,
        environment=real_env,
        computational_budget=200,
        k=20,
        alpha=0,
        discount=0.99,
        action_expansion_function=uniform
    )
    # planner = RandomPlanner(
    #     real_env
    # )

    print("Simulation Started")
    terminal = False
    rewards = []
    while not terminal:
        u = planner.plan(s)
        s, r, terminal, truncated, info = real_env.step(s, u)
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history

    fig, ax = plt.subplots()
    print(f"Simulation Ended with Reward: {sum(rewards)}")
    print(f"Max Reward: {max(rewards)}")
    print(f"Min Reward: {min(rewards)}")

    print("Creating Gif...")
    ani = FuncAnimation(
        fig,
        plot_frame,
        fargs=(goal, config, trajectory, ax),
        frames=len(trajectory)
    )
    ani.save("prova.gif", dpi=300, fps=15)
    print("Done")


main()
