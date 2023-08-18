import numpy as np

from bettergym.agents.utils.utils import get_robot_angles
from bettergym.environments.robot_arena import Config
import itertools


class Controller:

    def __init__(self, dt_in: float = 1.0, dt_out: float = 0.2):
        self.reduction_factor = dt_in / dt_out

    def plan(self, x: np.ndarray, config: Config, action: np.ndarray):
        robot_angles = get_robot_angles(x, config.max_angle_change)

        action_copy = np.array(action, copy=True)
        action_copy[1] = np.abs(x[2] - action[1])
        action_copy[0] = np.abs(x[3] - action[0])
        delta = action_copy / self.reduction_factor
        action_copy = np.array([x[3], x[2]])
        # if current velocity is higher than the target one(action) decrease it
        if x[3] > action[0]:
            action_copy[0] -= delta[0]
        else:
            action_copy[0] += delta[0]

        # if current angle is higher than the target one(action) decrease it
        if x[2] > action[1]:
            action_copy[1] -= delta[1]
        else:
            action_copy[1] += delta[1]

        in_range = any([rr[0] <= action_copy[1] <= rr[1] for rr in robot_angles])
        print(f"IN RANGE: {in_range}")
        if not in_range:
            print(f"x: {x}")
            print(f"action: {action}")
            # TODO: Do something

            dist_minus_pi = np.abs(x[2] - -np.pi)
            dist_pi = np.abs(x[2] - np.pi)
            minimum_is_dist_minus_pi = dist_minus_pi < dist_pi
            action_copy[1] = dist_minus_pi + np.abs(action[1] - np.pi) if minimum_is_dist_minus_pi \
                else dist_pi + np.abs(action[1] - -np.pi)
            action_copy[0] = np.abs(x[3] - action[0])
            delta = action_copy / self.reduction_factor
            action_copy = np.array([x[3], x[2]])

            action_copy[1] += -delta[1] if minimum_is_dist_minus_pi else delta[1]
            # if current velocity is higher than the target one(action) decrease it
            if x[3] > action[0]:
                action_copy[0] -= delta[0]
            else:
                action_copy[0] += delta[0]
            in_range = any([rr[0] <= action_copy[1] <= rr[1] for rr in robot_angles])
            print(f"IN RANGE: {in_range}")
        action_copy[1] = (action_copy[1] + np.pi) % (2 * np.pi) - np.pi
        return action_copy
