from abc import ABC, abstractmethod
from typing import Any

try:
    from MCTS_VO.bettergym.better_gym import BetterGym
except ModuleNotFoundError:
    from bettergym.better_gym import BetterGym


class Planner(ABC):
    def __init__(self, environment: BetterGym):
        self.environment = environment

    @abstractmethod
    def plan(self, initial_state: Any):
        pass
