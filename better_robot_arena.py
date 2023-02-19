import numpy as np
from gymnasium.spaces import Box

from better_gym import BetterGym
from robot_arena_gym import RobotArenaState


class BetterRobotArena(BetterGym):
    def get_actions(self, state: RobotArenaState):
        config = self.gym_env.config
        return Box(
            low=np.array([config.min_speed, -config.max_yaw_rate]),
            high=np.array([config.max_speed, config.max_yaw_rate])
        )
        pass

    def set_state(self, state) -> None:
        self.gym_env.state = state
