import math
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
from bettergym.agents.utils.utils import towards_goal, voo
from bettergym.environments.robot_arena import RobotArenaState, Config, BetterRobotArena
from experiment_utils import print_and_notify, plot_frame, plot_real_trajectory_information, \
    create_animation_tree_trajectory
from mcts_utils import sample_centered_robot_arena

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


def create_env_five_small_obs_continuous(initial_pos: tuple, goal: tuple):
    c = Config()
    obstacles_positions = np.array([
        [4.0, 4.0],
        [4.0, 6.0],
        [5.0, 5.0],
        [6.0, 4.0],
        [6.0, 6.0]
    ])
    obs = [RobotArenaState(np.pad(ob, (0, 2), 'constant'), goal=None, obstacles=None, radius=c.obs_size) for ob in
           obstacles_positions]
    initial_state = RobotArenaState(
        x=np.array([initial_pos[0], initial_pos[1], math.pi / 8.0, 0.0]),
        goal=np.array([goal[0], goal[1]]),
        obstacles=obs,
        radius=c.robot_radius
    )
    c.num_discrete_actions = 1
    real_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete=False,
        config=c
    )
    return real_env


def create_env_four_obs_difficult_continuous(initial_pos: tuple, goal: tuple):
    c = Config()
    obstacles_positions = np.array([
        [3.0, 2.0],
        [1.0, 4.0],
        [4.0, 7.0],
        [9.0, 7.0],
    ])
    radiuses = [1, 1, 2, 2]
    obs = [RobotArenaState(np.pad(obstacles_positions[i], (0, 2), 'constant'), goal=None, obstacles=None,
                           radius=radiuses[i]) for i in
           range(len(obstacles_positions))]
    initial_state = RobotArenaState(
        x=np.array([initial_pos[0], initial_pos[1], math.pi / 8.0, 0.0]),
        goal=np.array([goal[0], goal[1]]),
        obstacles=obs,
        radius=c.robot_radius
    )
    c.num_discrete_actions = 1
    real_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete=False,
        config=c
    )
    return real_env


def run_experiment(seed_val, num_actions=1, policy=None, discrete=False, var_angle: float = 0.38):
    global exp_num
    # input [forward speed, yaw_rate]
    real_env = create_env_four_obs_difficult_continuous(initial_pos=(1, 1), goal=(10, 10))
    s0, _ = real_env.reset()
    seed_everything(seed_val)
    trajectory = np.array(s0.x)
    obs = [s0.obstacles]
    config = real_env.config
    infos = []
    actions = []

    goal = s0.goal

    s = s0
    # planner = RandomPlanner(real_env)
    planner_apw = MctsApw(
        num_sim=1000,
        c=101,
        environment=real_env,
        computational_budget=100,
        k=50,
        alpha=0.1,
        discount=0.99,
        action_expansion_function=policy,
        rollout_policy=partial(towards_goal, var_angle=var_angle)
    )
    planner_mcts = Mcts(
        num_sim=1000,
        c=6,
        environment=real_env,
        computational_budget=100,
        discount=0.99,
        rollout_policy=partial(towards_goal, var_angle=var_angle)
    )
    planner = planner_apw if not discrete else planner_mcts

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

        final_time = time.time() - initial_time
        actions.append(u)
        infos.append(info)

        times.append(final_time)
        s, r, terminal, truncated, env_info = real_env.step(s, u)
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history
        obs.append(s.obstacles)

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
        print("Saving Debug Data...")
        trajectories = [i["trajectories"] for i in infos]
        rollout_values = [i["rollout_values"] for i in infos]
        print("Creating Tree Trajectories Animation...")
        create_animation_tree_trajectory(goal, config, obs)
        q_vals = [i["q_values"] for i in infos]
        a = [[an.action for an in i["actions"]] for i in infos]
        np.savez_compressed(f"debug/trajectories_{exp_num}", *trajectories)
        np.savez_compressed(f"debug/rollout_values_{exp_num}", *rollout_values)
        np.savez_compressed(f"debug/q_values_{exp_num}", *q_vals)
        np.savez_compressed(f"debug/actions_{exp_num}", *a)
        np.savez_compressed(f"debug/trajectory_real_{exp_num}", trajectory)
        np.savez_compressed(f"debug/chosen_a_{exp_num}", np.array(actions))

    print("Done")


def main():
    global exp_num
    exp_num = 0

    for p, na, var in [(partial(voo, eps=0.3, sample_centered=sample_centered_robot_arena), 1, 0.38 * 2)]:
        run_experiment(seed_val=2, policy=p, num_actions=na, discrete=False, var_angle=var)
        exp_num += 1


if __name__ == '__main__':
    main()
