import os
import random
import time
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import mean, std

from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import uniform, binary_policy, epsilon_greedy
from bettergym.environments.robot_arena import BetterRobotArena, Config
from utils import print_and_notify, plot_frame, plot_action_evolution


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def run_experiment(seed_val, rollout_policy, num_actions):
    global exp_num
    # input [forward speed, yaw_rate]
    c = Config()
    c.num_discrete_actions = num_actions
    real_env = BetterRobotArena(
        initial_position=(8, 8),
        gradient=True,
        discrete=False,
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
    planner = MctsApw(
        num_sim=1000,
        c=4,
        environment=real_env,
        computational_budget=100,
        k=20,
        alpha=0,
        discount=0.99,
        action_expansion_function=uniform,
        rollout_policy=rollout_policy
    )
    # planner = Mcts(
    #     num_sim=1000,
    #     c=4,
    #     environment=real_env,
    #     computational_budget=100,
    #     discount=0.99,
    #     rollout_policy=rollout_policy
    # )

    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    step_n = 0
    while not terminal:
        step_n += 1
        if step_n == 200:
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

    print_and_notify(
        f"Simulation Ended with Reward: {sum(rewards)}\n" +
        f"Avg Step Time: {round(mean(times), 2)}±{round(std(times), 2)}\n" +
        f"Total Time: {sum(times)}"
    )
    print("Creating Gif...")
    fig, ax = plt.subplots()
    ani = FuncAnimation(
        fig,
        plot_frame,
        fargs=(goal, config, trajectory, ax),
        frames=len(trajectory)
    )
    ani.save(f"debug/trajectory_{exp_num}.gif", fps=150)
    plot_action_evolution(np.array(actions))
    # trjs = [info["trajectories"] for info in infos]
    # np.savez_compressed(f"debug/tree_trajectory/tree_trajectories", *trjs)

    # with imageio.get_writer('tree_trajectory.gif', mode='i') as writer:
    # for i in range(len(infos)):
    #     file_name = f'debug/tree_trajectory/{i}.png'
    #     plot_tree_trajectory(i, infos, file_name)
    # image = imageio.v3.imread(file_name)
    # writer.append_data(image)
    # os.remove(file_name)

    print("Done")


def main():
    global exp_num
    exp_num = 0
    for p, na in [(uniform, 1), (binary_policy, 1),
                  (partial(epsilon_greedy, eps=0.2, other_func=binary_policy), 1)]:
        run_experiment(0, p, na)


if __name__ == '__main__':
    main()
