import math
from copy import deepcopy

import numpy as np

from bettergym.environments.robot_arena import BetterRobotArena, RobotArenaState, Config


def create_env_continuous(
    initial_pos,
    goal,
    obs,
    discrete: bool,
    rwrd_in_sim: bool,
    real_c: Config,
    sim_c: Config,
):
    initial_state = RobotArenaState(
        x=np.array([initial_pos[0], initial_pos[1], math.pi / 8.0, 0.0]),
        goal=np.array([goal[0], goal[1]]),
        obstacles=obs,
        radius=real_c.robot_radius,
    )
    real_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=real_c,
        collision_rwrd=True,
    )
    sim_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=sim_c,
        collision_rwrd=rwrd_in_sim,
    )
    return real_env, sim_env


def create_env_five_small_obs_continuous(
    initial_pos: tuple,
    goal: tuple,
    discrete: bool,
    rwrd_in_sim: bool,
    out_boundaries_rwrd: int,
    dt_sim: float,
    n_angles: int,
    n_vel: int,
):
    obstacles_positions = np.array(
        [[4.0, 4.0], [4.0, 6.0], [5.0, 5.0], [6.0, 4.0], [6.0, 6.0]]
    )
    dt_real = dt_sim
    real_c = Config(dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel)
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
    )
    obs = [
        RobotArenaState(
            np.pad(ob, (0, 2), "constant"),
            goal=None,
            obstacles=None,
            radius=real_c.obs_size,
        )
        for ob in obstacles_positions
    ]
    real_env, sim_env = create_env_continuous(
        initial_pos=initial_pos,
        goal=goal,
        obs=obs,
        discrete=discrete,
        rwrd_in_sim=rwrd_in_sim,
        real_c=real_c,
        sim_c=sim_c,
    )
    sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
    return real_env, sim_env


def create_env_four_obs_difficult_continuous(
    initial_pos: tuple,
    goal: tuple,
    discrete: bool,
    rwrd_in_sim: bool,
    out_boundaries_rwrd: int,
    dt_sim: float,
    n_angles: int,
    n_vel: int,
):
    obstacles_positions = np.array(
        [
            [3.4, 1.1],
            [1.0, 4.0],
            [4.0, 7.0],
            [9.5, 7.0],
        ]
    )
    radiuses = [1.8, 1, 2, 2]
    dt_real = dt_sim
    real_c = Config(dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel)
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
    )
    obs = [
        RobotArenaState(
            np.pad(obstacles_positions[i], (0, 2), "constant"),
            goal=None,
            obstacles=None,
            radius=radiuses[i],
        )
        for i in range(len(obstacles_positions))
    ]

    real_env, sim_env = create_env_continuous(
        initial_pos=initial_pos,
        goal=goal,
        obs=obs,
        discrete=discrete,
        rwrd_in_sim=rwrd_in_sim,
        real_c=real_c,
        sim_c=sim_c,
    )
    sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
    return real_env, sim_env


def create_env_multiagent_five_small_obs_continuous(
    initial_pos: tuple,
    goal: tuple,
    discrete: bool,
    rwrd_in_sim: bool,
    out_boundaries_rwrd: int,
    dt_sim: float,
    n_angles: int,
    n_vel: int,
):
    dt_real = dt_sim
    real_c = Config(dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel)
    sim_c = Config(dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel)
    obstacles_positions = np.array(
        [
            # add static obstacles
            [4.0, 4.0],
            [4.0, 6.0],
            [5.0, 5.0],
            [6.0, 4.0],
            [6.0, 6.0],
            # dynamic obstacle
            [goal[0], goal[1]],
        ]
    )
    obs = [
        RobotArenaState(
            np.pad(ob, (0, 2), "constant"),
            goal=None,
            obstacles=None,
            radius=real_c.obs_size,
        )
        for ob in obstacles_positions
    ]
    dynamic_obs = obs[-1]
    dynamic_obs.radius = real_c.robot_radius
    dynamic_obs.x[2] = -2.748893571891069  # opposite of pi/8
    dynamic_obs.goal = np.array([initial_pos[0], initial_pos[1]])

    initial_state = RobotArenaState(
        x=np.array([initial_pos[0], initial_pos[1], math.pi / 8.0, 0.0]),
        goal=np.array([goal[0], goal[1]]),
        obstacles=obs,
        radius=real_c.robot_radius,
    )
    obs_r2 = deepcopy(obs)
    obs_r2[-1] = initial_state
    dynamic_obs.obstacles = obs_r2

    real_env_1 = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=real_c,
        collision_rwrd=True,
        multiagent=True
    )
    sim_env_1 = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=sim_c,
        collision_rwrd=rwrd_in_sim,
        multiagent=True
    )
    real_env_2 = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=real_c,
        collision_rwrd=True,
        multiagent=True
    )
    sim_env_2 = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=sim_c,
        collision_rwrd=rwrd_in_sim,
        multiagent=True
    )

    sim_env_1.gym_env.WALL_REWARD = out_boundaries_rwrd
    sim_env_2.gym_env.WALL_REWARD = out_boundaries_rwrd

    return real_env_1, real_env_2, sim_env_1, sim_env_2
