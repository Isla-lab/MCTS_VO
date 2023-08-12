import numpy as np

from bettergym.agents.utils.utils import get_robot_angles
from bettergym.environments.robot_arena import Config


class Controller:
    def plan(self, x: np.ndarray, config: Config, action: np.ndarray, dt_in: float = 1.0, dt_out: float = 0.2):
        robot_angles = get_robot_angles(x, config.max_angle_change)
        reduction_factor = dt_in/dt_out
        return action / reduction_factor
