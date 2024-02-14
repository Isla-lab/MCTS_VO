import numpy as np
from numba import njit


@njit
def check_coll_jit(x, obs, robot_radius, obs_size):
    for i, ob in enumerate(obs):
        dist_to_ob = np.linalg.norm(ob - x[:2])
        if dist_to_ob <= robot_radius + obs_size[i]:
            return True
    return False


@njit
def dist_to_goal(goal: np.ndarray, x: np.ndarray):
    return np.linalg.norm(x - goal)