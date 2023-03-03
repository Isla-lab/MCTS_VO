import os
import random
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import mean, std

from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import uniform, towards_goal
from bettergym.environments.robot_arena import BetterRobotArena


def seed_everything(seed_value: int):
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
    ax.plot(x[0], x[1], "xr")
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
    real_env = BetterRobotArena(
        initial_position=(0, 0),
        gradient=True,
        discrete=True
    )
    s0, _ = real_env.reset()
    seed_val = 3
    seed_everything(seed_val)
    trajectory = np.array(s0.x)
    config = real_env.config
    goal = s0.goal

    s = s0
    planner = MctsApw(
        num_sim=1000,
        c=4,
        environment=real_env,
        computational_budget=100,
        k=20,
        alpha=0,
        discount=0.99,
        action_expansion_function=uniform,
        rollout_policy=towards_goal
    )

    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    step_n = 0
    while not terminal:
        step_n += 1
        if step_n == 500:
            break
        print(f"Step Number {step_n}")
        initial_time = time.time()
        u = planner.plan(s)
        final_time = time.time() - initial_time
        times.append(final_time)
        s, r, terminal, truncated, info = real_env.step(s, u)
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history

    fig, ax = plt.subplots()
    print(f"Simulation Ended with Reward: {sum(rewards)}")
    print(f"Avg Step Time: {round(mean(times), 2)}±{round(std(times), 2)}")
    print(f"Total Time: {sum(times)}")
    print("Creating Gif...")
    ani = FuncAnimation(
        fig,
        plot_frame,
        fargs=(goal, config, trajectory, ax),
        frames=len(trajectory)
    )
    ani.save(f"prova_{seed_val}.gif", dpi=300, fps=150)
    print("Done")


main()
