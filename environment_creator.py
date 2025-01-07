try:
    from MCTS_VO.bettergym.environments.env import BetterEnv, EnvConfig
except ModuleNotFoundError:
    from bettergym.environments.env import BetterEnv, EnvConfig


def create_pedestrian_env(
        discrete: bool,
        rwrd_in_sim: bool,
        out_boundaries_rwrd: int,
        n_angles: int,
        n_vel: int,
        vo: bool,
        n_obs: int,
        obs_pos: list = None,
        dt_real = 1.0
):
    real_c = EnvConfig(
        dt=dt_real, max_angle_change=1.9*dt_real, n_angles=n_angles, n_vel=n_vel, num_humans=n_obs
    )

    real_env = BetterEnv(discrete_env=discrete, vo=vo, config=real_c, collision_rwrd=rwrd_in_sim, sim_env=True,
                        obs_pos=None)
    sim_env = BetterEnv(discrete_env=discrete, vo=vo, config=real_c, collision_rwrd=rwrd_in_sim, sim_env=True,
                        obs_pos=None)
    sim_env.gym_env.WALL_REWARD = out_boundaries_rwrd
    real_env.gym_env.WALL_REWARD = out_boundaries_rwrd

    return real_env, sim_env
