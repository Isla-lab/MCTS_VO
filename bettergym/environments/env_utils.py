
import numpy as np
from numba import njit, jit


@njit
def check_coll_jit(x, obs, robot_radius, obs_size):
    for i, ob in enumerate(obs):
        dist_to_ob = np.linalg.norm(ob - x[:2])
        if dist_to_ob <= robot_radius + obs_size[i]:
            return True
    return False


@jit(cache=True, nopython=True)
def check_coll_vectorized(x, obs, robot_radius, obs_size):
    n = obs.shape[0]
    distances = np.empty(n)
    for i in range(n):
        distances[i] = np.sqrt(np.sum((obs[i] - x)**2))
    return np.any(distances <= robot_radius + obs_size)


@jit(cache=True, nopython=True)
def dist_to_goal(goal: np.ndarray, x: np.ndarray):
    return np.sqrt(np.sum((x-goal)**2))