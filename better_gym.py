from abc import abstractmethod
from typing import TypeVar, SupportsFloat, Any, Union
import gymnasium as gym

class BetterGym:
    def __init__(self, env: gym.Env):
        self.gym_env = env
    def step(self, state, action) -> tuple[Any, Union[float, int], bool, bool, Union[dict[str, Any]]]:
        """
        A transition function that works by
        :param state: the state of the environment
        :param action: action to perform into the environment to transition to next state
        :return:
        """
        self.set_state(state)
        return self.gym_env.step(action)

    @abstractmethod
    def set_state(self, state) -> None:
        """
        set the state of the environment
        :param state: the state of the environment
        :return:
        """
        pass
    @abstractmethod
    def get_actions(self, state):
        pass

    def __getattr__(self, attr):
        return getattr(self.gym_env, attr)

