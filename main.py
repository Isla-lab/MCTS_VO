import os
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.better_gym import BetterGym
from bettergym.environments.robot_arena import RobotArena, BetterRobotArena


def seed_everything(real_env: BetterGym, sim_env: BetterGym, seed_value: int):
    real_env.gym_env.seed(seed_value)
    sim_env.gym_env.seed(seed_value)
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

    ax.axis("equal")
    ax.grid(True)


def main():
    # input [forward speed, yaw_rate]
    real_env = BetterRobotArena((0, 0))
    s0, _ = real_env.reset()
    trajectory = np.array(s0.x)
    config = real_env.config
    goal = s0.goal

    s = s0
    planner = MctsApw(
        num_sim=10000,
        c=0.5,
        environment=real_env,
        computational_budget=200,
        k=15,
        alpha=0,
        action_expansion_function=None
    )

    print("Simulation Started")
    for _ in range(100):
        u = planner.plan(s)
        s, r, terminal, truncated, info = real_env.step(s, u)
        trajectory = np.vstack((trajectory, s.x))  # store state history
        if terminal:
            break

    fig, ax = plt.subplots()
    ax.set_xlim([config.left_limit, config.right_limit])
    ax.set_ylim([config.bottom_limit, config.upper_limit])
    print("Simulation Ended")

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
