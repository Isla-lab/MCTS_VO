import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from notify_run import Notify

notify = Notify()


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


def plot_tree_trajectory(i, infos, ax):
    trajectories = infos[i]["trajectories"]
    x = trajectories[0][0]
    ax.clear()
    plt.axis("equal")
    plt.grid(True)

    for t in trajectories:
        plt.plot(t[:, 0], t[:, 1], "--r")

    # STEP NUMBER
    plt.text(plt.gca().get_xlim()[0] - 0.3, plt.gca().get_ylim()[1] - 0.2, str(i), fontsize=20)
    # ROBOT POSITION
    ax.plot(x[0], x[1], "xg")


def plot_action_evolution(actions: np.ndarray):
    def plot(data):
        fig, axs = plt.subplots(2, sharex=True)
        sns.lineplot(data=data, x=data.index, y="Linear Velocity", ax=axs[0], color='#4c72b0')
        sns.lineplot(data=data, x=data.index, y="Angular Velocity", ax=axs[1], color='#c44e52')
        fig.savefig(f'debug/action_evolution_{len(data)}.svg')

    sns.set_theme()
    df = pd.DataFrame(
        {
            "Linear Velocity": actions[:, 0],
            "Angular Velocity": actions[:, 1]
        }
    )
    plot(df.iloc[:100])
    plot(df.iloc[:200])
    plot(df)
    np.save("actions", actions)
    sns.reset_orig()


def print_and_notify(message: str):
    print(message)
    notify.send(message)
