import math

import numpy as np

from bettergym.environments.robot_arena import BetterRobotArena, RobotArenaState, Config


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
        sim=False,
        config=c
    )
    sim_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete=False,
        sim=True,
        config=c
    )
    return real_env, sim_env


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
        sim=False,
        config=c
    )
    sim_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete=False,
        sim=True,
        config=c
    )
    return real_env, sim_env
