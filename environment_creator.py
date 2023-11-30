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
        vo: bool,
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
        vo=vo,
    )
    sim_env = BetterRobotArena(
        initial_state=initial_state,
        gradient=True,
        discrete_env=discrete,
        config=sim_c,
        collision_rwrd=rwrd_in_sim,
        vo=vo,
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
        vo: bool,
):
    obstacles_positions = np.array(
        [[3, 3], [3, 7.0], [5.0, 5.0], [7, 3], [7, 7.0]]
    )
    # obstacles_positions = np.array(
    #     [[4.0, 4.0], [4.0, 6.0], [5.0, 5.0], [6.0, 4.0], [6.0, 6.0]]
    # )

    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel
        , obs_size=1.3
    )
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
        , obs_size=1.3
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
        vo=vo,
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
        vo: bool,
):
    obstacles_positions = np.array(
        [
            [3.4, 0.8],
            [0.8, 4.3],
            [4.0, 8.0],
            [9.5, 7.0],
            [6.5, 4.0],
            [4.0, 4.0]
        ]
    )
    radiuses = [1.5, 1.3, 2.1, 2.1, 0.6, 0.3]
    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel
    )
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
        vo=vo,
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
        vo: bool,
):
    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel
    )
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
    )
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
    # dynamic_obs.dynamic_obs = True
    dynamic_obs.radius = real_c.robot_radius
    dynamic_obs.x[2] = 0
    dynamic_obs.x[3] = 0.3
    dynamic_obs.obstacles = None
    # dynamic_obs.goal = np.array([initial_pos[0], initial_pos[1]])

    initial_state_1 = RobotArenaState(
        x=np.array([initial_pos[0], initial_pos[1], math.pi / 8.0, 0.0]),
        goal=np.array([goal[0], goal[1]]),
        obstacles=obs,
        radius=real_c.robot_radius,
    )
    obs_r2 = deepcopy(obs)
    obs_r2[-1] = deepcopy(initial_state_1)
    obs_r2[-1].x[2] = 0
    obs_r2[-1].x[3] = 0.3
    obs_r2[-1].obstacles = None
    # dynamic_obs.obstacles = obs_r2
    angle_2 = (math.pi / 8 + math.pi) % (2 * math.pi) - math.pi  # opposite of pi/8
    initial_state_2 = RobotArenaState(
        x=np.array([goal[0], goal[1], angle_2, 0.0]),
        goal=np.array([initial_pos[0], initial_pos[1]]),
        obstacles=obs_r2,
        radius=real_c.robot_radius,
    )

    real_env_1 = BetterRobotArena(
        initial_state=initial_state_1,
        gradient=True,
        discrete_env=discrete,
        config=real_c,
        collision_rwrd=True,
        multiagent=True,
        vo=vo,
    )
    sim_env_1 = BetterRobotArena(
        initial_state=initial_state_1,
        gradient=True,
        discrete_env=discrete,
        config=sim_c,
        collision_rwrd=rwrd_in_sim,
        multiagent=True,
        vo=vo,
    )
    real_env_2 = BetterRobotArena(
        initial_state=initial_state_2,
        gradient=True,
        discrete_env=discrete,
        config=real_c,
        collision_rwrd=True,
        multiagent=True,
        vo=vo,
    )
    sim_env_2 = BetterRobotArena(
        initial_state=initial_state_2,
        gradient=True,
        discrete_env=discrete,
        config=sim_c,
        collision_rwrd=rwrd_in_sim,
        multiagent=True,
        vo=vo,
    )

    sim_env_1.gym_env.WALL_REWARD = out_boundaries_rwrd
    sim_env_2.gym_env.WALL_REWARD = out_boundaries_rwrd

    return real_env_1, real_env_2, sim_env_1, sim_env_2


def create_env_4ag(
        discrete: bool,
        rwrd_in_sim: bool,
        out_boundaries_rwrd: int,
        dt_sim: float,
        n_angles: int,
        n_vel: int,
        vo: bool,
):
    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel
    )
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
    )
    real_envs = []
    sim_envs = []
    initial_states = np.array(
        [
            # dynamic obstacle
            [1.0, 1.0, 0.0, 0.3],
            [10.0, 10.0, 0.0, 0.3],
            [1.0, 10.0, 0.0, 0.3],
            [10.0, 1.0, 0.0, 0.3],
        ]
    )
    goal_positions = np.array(
        [
            # dynamic obstacle
            [10.0, 10.0],
            [1.0, 1.0],
            [10.0, 1.0],
            [1.0, 10.0],
        ]
    )
    angles = [
        math.pi / 8,
        (math.pi / 8 + math.pi) % (2 * math.pi) - math.pi,
        3,
        (3 + math.pi) % (2 * math.pi) - math.pi,
    ]

    for i, obs_pos in enumerate(initial_states):
        obs = [
            RobotArenaState(
                ob,
                goal=goal_positions[j],
                obstacles=None,
                radius=real_c.robot_radius,
            )
            for j, ob in enumerate(initial_states)
            if not np.array_equal(obs_pos, ob)
        ]
        initial_pos = np.array(obs_pos, copy=True)
        initial_pos[2] = angles[i]
        initial_pos[3] = 0.0
        initial_state = RobotArenaState(
            x=initial_pos,
            goal=goal_positions[i],
            obstacles=obs,
            radius=real_c.robot_radius,
        )
        real_env = BetterRobotArena(
            initial_state=initial_state,
            gradient=True,
            discrete_env=discrete,
            config=real_c,
            collision_rwrd=True,
            multiagent=True,
            vo=vo,
        )
        sim_env = BetterRobotArena(
            initial_state=initial_state,
            gradient=True,
            discrete_env=discrete,
            config=sim_c,
            collision_rwrd=rwrd_in_sim,
            multiagent=True,
            vo=vo,
        )
        sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
        real_envs.append(real_env)
        sim_envs.append(sim_env)
    return real_envs, sim_envs


def create_env_4ag_obs(
        discrete: bool,
        rwrd_in_sim: bool,
        out_boundaries_rwrd: int,
        dt_sim: float,
        n_angles: int,
        n_vel: int,
        vo: bool,
):
    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel
    )
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel
    )
    real_envs = []
    sim_envs = []
    initial_states = np.array(
        [
            # dynamic obstacle
            [1.0, 1.0, 0.0, 0.3],
            [10.0, 10.0, 0.0, 0.3],
            [1.0, 10.0, 0.0, 0.3],
            [10.0, 1.0, 0.0, 0.3],
        ]
    )
    goal_positions = np.array(
        [
            # dynamic obstacle
            [10.0, 10.0],
            [1.0, 1.0],
            [10.0, 1.0],
            [1.0, 10.0],
        ]
    )
    angles = [
        math.pi / 8,
        (math.pi / 8 + math.pi) % (2 * math.pi) - math.pi,
        3,
        (3 + math.pi) % (2 * math.pi) - math.pi,
    ]

    fixed_obs_positions = np.array(
        [
            # add static obstacles
            [4.0, 4.0],
            [4.0, 6.0],
            [5.0, 5.0],
            [6.0, 4.0],
            [6.0, 6.0],
        ]
    )
    fixed_obs = [
        RobotArenaState(
            np.pad(ob, (0, 2), "constant"),
            goal=None,
            obstacles=None,
            radius=real_c.obs_size,
        )
        for ob in fixed_obs_positions
    ]

    for i, obs_pos in enumerate(initial_states):
        obs = [
            RobotArenaState(
                ob,
                goal=goal_positions[j],
                obstacles=None,
                radius=real_c.robot_radius,
            )
            for j, ob in enumerate(initial_states)
            if not np.array_equal(obs_pos, ob)
        ]
        obs.extend(fixed_obs)
        initial_pos = np.array(obs_pos, copy=True)
        initial_pos[2] = angles[i]
        initial_pos[3] = 0.0
        initial_state = RobotArenaState(
            x=initial_pos,
            goal=goal_positions[i],
            obstacles=obs,
            radius=real_c.robot_radius,
        )
        real_env = BetterRobotArena(
            initial_state=initial_state,
            gradient=True,
            discrete_env=discrete,
            config=real_c,
            collision_rwrd=True,
            multiagent=True,
            vo=vo,
        )
        sim_env = BetterRobotArena(
            initial_state=initial_state,
            gradient=True,
            discrete_env=discrete,
            config=sim_c,
            collision_rwrd=rwrd_in_sim,
            multiagent=True,
            vo=vo,
        )
        sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
        real_envs.append(real_env)
        sim_envs.append(sim_env)
    return real_envs, sim_envs


def create_env_8ag_obs(
        discrete: bool,
        rwrd_in_sim: bool,
        out_boundaries_rwrd: int,
        dt_sim: float,
        n_angles: int,
        n_vel: int,
        vo: bool,
):
    dt_real = dt_sim
    real_c = Config(
        dt=dt_real, max_angle_change=1.9 * dt_real, n_angles=n_angles, n_vel=n_vel, obs_size=1.3
    )
    sim_c = Config(
        dt=dt_sim, max_angle_change=1.9 * dt_sim, n_angles=n_angles, n_vel=n_vel, obs_size=1.3
    )
    real_envs = []
    sim_envs = []
    initial_states = np.array(
        [
            # dynamic obstacle
            [1.0, 1.0, 0.0, 0.3],
            [10.0, 10.0, 0.0, 0.3],
            [1.0, 10.0, 0.0, 0.3],
            [10.0, 1.0, 0.0, 0.3],
            # new agents
            [5.0, 1.0, 0.0, 0.3],
            [5.0, 10.0, 0.0, 0.3],
            [1.0, 5.0, 0.0, 0.3],
            [10.0, 5.0, 0.0, 0.3]
        ]
    )
    goal_positions = np.array(
        [
            # dynamic obstacle
            [10.0, 10.0],
            [1.0, 1.0],
            [10.0, 1.0],
            [1.0, 10.0],
            # new agents
            [5.0, 10.0],
            [5.0, 1.0],
            [10.0, 5.0],
            [1.0, 5.0]
        ]
    )
    # angles = [
    #     math.pi / 8,
    #     (math.pi / 8 + math.pi) % (2 * math.pi) - math.pi,
    #     3,
    #     (3 + math.pi) % (2 * math.pi) - math.pi,
    # ]

    # fixed_obs_positions = np.array(
    #     [
    #         # add static obstacles
    #         [4.0, 4.0],
    #         [4.0, 6.0],
    #         [5.0, 5.0],
    #         [6.0, 4.0],
    #         [6.0, 6.0],
    #     ]
    # )
    fixed_obs_positions = np.array(
        [[3, 3], [3, 7.0], [5.0, 5.0], [7, 3], [7, 7.0]]
    )
    fixed_obs = [
        RobotArenaState(
            np.pad(ob, (0, 2), "constant"),
            goal=None,
            obstacles=None,
            radius=real_c.obs_size,
        )
        for ob in fixed_obs_positions
    ]

    for i, obs_pos in enumerate(initial_states):
        obs = [
            RobotArenaState(
                ob,
                goal=goal_positions[j],
                obstacles=None,
                radius=real_c.robot_radius,
            )
            for j, ob in enumerate(initial_states)
            if not np.array_equal(obs_pos, ob)
        ]
        obs.extend(fixed_obs)
        initial_pos = np.array(obs_pos, copy=True)
        # initial_pos[2] = angles[i]
        initial_pos[3] = 0.0
        initial_state = RobotArenaState(
            x=initial_pos,
            goal=goal_positions[i],
            obstacles=obs,
            radius=real_c.robot_radius,
        )
        real_env = BetterRobotArena(
            initial_state=initial_state,
            gradient=True,
            discrete_env=discrete,
            config=real_c,
            collision_rwrd=True,
            multiagent=True,
            vo=vo,
        )
        sim_env = BetterRobotArena(
            initial_state=initial_state,
            gradient=True,
            discrete_env=discrete,
            config=sim_c,
            collision_rwrd=rwrd_in_sim,
            multiagent=True,
            vo=vo,
        )
        sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
        real_envs.append(real_env)
        sim_envs.append(sim_env)
    return real_envs, sim_envs
