from dataclasses import dataclass
from typing import Union, Any

import gymnasium as gym
import numpy as np

from better_gym import BetterGym


class BetterFrozenLake(BetterGym):
    def set_state(self, state) -> None:
        self.gym_env.s = state

    def get_actions(self, state):
        return np.arange(0, self.gym_env.action_space.n)


@dataclass
class ActionNode:
    action: Any
    # num_visits_states: np.ndarray
    id_to_state = {}
    state_to_id = {}


@dataclass
class StateNode:
    state: Any
    id: int
    actions: tuple
    num_visits_actions: np.ndarray
    a_values: np.ndarray

    def __init__(self, environment, state, node_id):
        self.id = node_id
        actions = environment.get_actions(state)
        self.actions = tuple(ActionNode(a) for a in actions)
        self.num_visits_actions = np.array([np.zeros(len(actions), dtype=np.float64)])
        self.a_values = np.array([np.zeros(len(actions), dtype=np.float64)])


class Mcts:
    def __init__(self, num_sim: int, c: float | int, s0: Any, environment: BetterGym, computational_budget: int,
                 discount: float | int = 1):
        self.id_to_state_node: dict[int, StateNode] = {}
        self.num_sim: int = num_sim
        self.c: float | int = c
        self.s0: Any = s0
        self.environment: BetterGym = environment
        self.computational_budget: int = computational_budget
        self.discount: float | int = discount

        self.num_visits_actions = np.array([], dtype=np.float64)
        self.a_values = np.array([])

        self.num_visits_states = np.array([], dtype=np.float64)
        # self.id_to_state_node = {}
        # self.state_to_id = {}
        self.state_actions = {}

    def plan(self, initial_state: Any):
        root_id = 0
        root_node = StateNode(self.environment, initial_state, root_id)
        self.id_to_state_node[root_id] = root_node
        # self.state_to_id[initial_state] = root_id
        self.num_visits_states = np.append(self.num_visits_states, 0)
        for sn in range(self.num_sim):
            self.__simulate(state_id=root_id)

    def __simulate(self, state_id: int):
        node = self.id_to_state_node[state_id]
        self.num_visits_states[state_id] += 1
        current_state = node.state

        # UCB
        # Q + c * sqrt(ln(Parent_Visit)/Child_visit)
        ucb_scores = node.a_values / node.num_visits_actions + self.c * np.sqrt(
            np.log(self.num_visits_states[state_id]) / node.num_visits_actions
        )
        ucb_scores[np.isnan(ucb_scores)] = np.inf
        # randomly choose between actions which have the maximum ucb value
        action_idx = np.random.choice(np.flatnonzero(ucb_scores == np.max(ucb_scores)))
        # get action corresponding to the index
        action_node = node.actions[action_idx]
        action = action_node.action
        # increase action visits
        node.num_visits_actions[action_idx] += 1

        current_state, r, terminal, _, _ = env.step(current_state, action)
        new_state_id = action_node.state_to_id.get(current_state, None)

        if new_state_id is None:
            state_id += 1
            # Initialize State Data
            node = StateNode(self.environment, current_state, state_id)
            self.id_to_state_node[state_id] = node
            action_node.state_to_id[current_state] = state_id
            self.num_visits_states = np.append(self.num_visits_states, 0)
            # Do Rollout
            disc_rollout_value = self.discount * self.rollout(current_state)
            node.a_values[action_idx] += disc_rollout_value
            return disc_rollout_value
        else:
            state_id = new_state_id
            if terminal:
                return 0
            else:
                disc_value = self.discount * self.__simulate(state_id)
                node.a_values[state_id][action_idx] += disc_value
                return disc_value

    def rollout(self, current_state) -> Union[int, float]:
        terminal = False
        r = 0
        while terminal or self.computational_budget == 0:
            # random policy
            actions = self.environment.get_actions(current_state)
            chosen_action = np.random.choice(actions)
            current_state, r, terminal, _, _ = env.step(current_state, chosen_action)
        return r


if __name__ == '__main__':
    env = BetterFrozenLake(
        gym.make("FrozenLake-v1").unwrapped
    )
    s, _ = env.reset()
    planner = Mcts(
        num_sim=100,
        c=1,
        s0=s,
        environment=env,
        computational_budget=100,
        discount=0.9
    )
    planner.plan(s)
