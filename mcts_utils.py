import numpy as np


def sample_centered_robot_arena(center: np.ndarray):
    return np.random.multivariate_normal(
        center,
        np.diag([0.3 / 10, 1.9 / 10])
    )
