from dataclasses import dataclass
from typing import Union, Any

import gymnasium as gym
import numpy as np

from environment.better_gym import BetterGym


class BetterFrozenLake(BetterGym):
    def set_state(self, state) -> None:
        self.gym_env.s = state

    def get_actions(self, state):
        return np.arange(0, self.gym_env.action_space.n)


@dataclass
class Tree:
    state: Any
    id: int
    num_visits_actions: np.ndarray
    a_values: np.ndarray


class Mcts:
    def __init__(self, num_sim: int, c: Union[float, int], s0: Any, environment: BetterGym, computational_budget: int,
                 discount: Union[int, float] = 1):
        self.num_sim = num_sim
        self.c = c
        self.s0 = s0
        self.environment = environment
        self.computational_budget = computational_budget
        self.discount = discount

        self.num_visits_actions = np.array([], dtype=np.float64)
        self.a_values = np.array([])

        self.num_visits_states = np.array([], dtype=np.float64)
        self.id_to_state = {}
        self.state_to_id = {}
        self.state_actions = {}

    def plan(self, initial_state: Any):
        root_id = 0
        self.id_to_state[root_id] = initial_state
        self.state_to_id[initial_state] = root_id
        self.num_visits_states = np.append(self.num_visits_states, 0)
        for sn in range(self.num_sim):
            self.__simulate(state_id=root_id)

    def __simulate(self, state_id: int):
        self.num_visits_states[state_id] += 1
        current_state = self.id_to_state[state_id]

        # if it's the first visit add visits of child
        if self.num_visits_states[state_id] == 1:
            actions = self.environment.get_actions(current_state)
            self.state_actions[state_id] = actions
            try:
                self.num_visits_actions = np.vstack((self.num_visits_actions, np.zeros(len(actions))))
                self.a_values = np.vstack((self.a_values, np.zeros(len(actions))))
            except ValueError:
                self.num_visits_actions = np.array([np.zeros(len(actions), dtype=np.float64)])
                self.a_values = np.array([np.zeros(len(actions), dtype=np.float64)])
        # UCB
        ucb_scores = self.a_values[state_id] + self.c * np.sqrt(
            np.divide(np.log(self.num_visits_states[state_id]), self.num_visits_actions[state_id])
        )
        ucb_scores[np.isnan(ucb_scores)] = np.inf
        # randomly choose between actions which have the maximum ucb value
        action_idx = np.random.choice(np.flatnonzero(ucb_scores == np.max(ucb_scores)))
        # get action corresponding to the index
        action = self.state_actions[state_id][action_idx]
        # increase action visits
        self.num_visits_actions[state_id][action] += 1

        current_state, r, terminal, _, _ = env.step(current_state, action)
        new_state_id = self.state_to_id.get(current_state, None)

        # if the number of visits of current state is 1 then the next state will be a new node
        # and hence we need a new id
        # otherwise the next state will be a node which id already exist
        # TODO: fix
        # PROBLEM: a state can be repeated within the tree so the mapping state -> id might be problematic

        if new_state_id is None:
            state_id += 1
            # Initialize State Data
            self.id_to_state[state_id] = current_state
            self.state_to_id[current_state] = state_id
            self.num_visits_states = np.append(self.num_visits_states, 0)
            # Do Rollout
            disc_rollout_value = self.discount * self.rollout(current_state)
            self.a_values[state_id][action_idx] += disc_rollout_value
            return disc_rollout_value
        else:
            state_id = new_state_id
            if terminal:
                return 0
            else:
                disc_value = self.discount * self.__simulate(state_id)
                self.a_values[state_id][action_idx] += disc_value
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
