from abc import ABC, abstractmethod
from typing import Any


class Planner(ABC):
    @abstractmethod
    def plan(self, initial_state: Any):
        pass
