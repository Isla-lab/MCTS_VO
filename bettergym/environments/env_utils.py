
import numpy as np
from numba import jit


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