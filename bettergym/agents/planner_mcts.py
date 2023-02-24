from dataclasses import dataclass, field
from typing import Union, Any, Dict

import numpy as np

from bettergym.agents.planner import Planner
from bettergym.better_gym import BetterGym


@dataclass
class ActionNode:
    action: Any
    state_to_id: Dict[Any, int] = field(default_factory=dict)


@dataclass
class StateNode:
    state: Any
    id: int
    actions: list
    num_visits_actions: np.ndarray
    a_values: np.ndarray
    num_visits: int = 0

    def __init__(self, environment, state, node_id):
        self.id = node_id
        self.state = state
        acts = environment.get_actions(state)
        self.actions = [ActionNode(a) for a in acts]
        self.num_visits_actions = np.zeros_like(self.actions, dtype=np.float64)
        self.a_values = np.zeros_like(self.actions, dtype=np.float64)


class MctsV2(Planner):
    def __init__(self, num_sim: int, c: float | int, environment: BetterGym, computational_budget: int,
                 discount: float | int = 1):
        self.id_to_state_node: dict[int, StateNode] = {}
        self.num_sim: int = num_sim
        self.c: float | int = c
        self.environment: BetterGym = environment
        self.computational_budget: int = computational_budget
        self.discount: float | int = discount

        self.num_visits_actions = np.array([], dtype=np.float64)
        self.a_values = np.array([])
        self.state_actions = {}
        self.last_id = -1

    def get_id(self):
        self.last_id += 1
        return self.last_id

    def plan(self, initial_state: Any):
        root_id = self.get_id()
        root_node = StateNode(self.environment, initial_state, root_id)
        self.id_to_state_node[root_id] = root_node
        for sn in range(self.num_sim):
            self.simulate(state_id=root_id)

        q_vals = root_node.a_values / root_node.num_visits_actions
        # randomly choose between actions which have the maximum ucb value
        action_idx = np.random.choice(np.flatnonzero(q_vals == np.max(q_vals)))
        action = root_node.actions[action_idx].action
        return action

    def simulate(self, state_id: int):
        node = self.id_to_state_node[state_id]
        node.num_visits += 1
        current_state = node.state

        # UCB
        # Q + c * sqrt(ln(Parent_Visit)/Child_visit)
        q_vals = node.a_values / node.num_visits_actions
        q_vals[np.isnan(q_vals)] = np.inf

        ucb_scores = q_vals + self.c * np.sqrt(
            np.log(node.num_visits) / node.num_visits_actions
        )
        ucb_scores[np.isnan(ucb_scores)] = np.inf
        # randomly choose between actions which have the maximum ucb value
        action_idx = np.random.choice(np.flatnonzero(ucb_scores == np.max(ucb_scores)))
        # get action corresponding to the index
        action_node = node.actions[action_idx]
        action = action_node.action
        # increase action visits
        node.num_visits_actions[action_idx] += 1

        current_state, r, terminal, _, _ = self.environment.step(current_state, action)
        new_state_id = action_node.state_to_id.get(current_state, None)

        prev_node = node
        if new_state_id is None:
            # Leaf Node
            state_id = self.get_id()
            # Initialize State Data
            node = StateNode(self.environment, current_state, state_id)
            self.id_to_state_node[state_id] = node
            action_node.state_to_id[current_state] = state_id
            node.num_visits += 1
            # Do Rollout
            disc_rollout_value = self.discount * self.rollout(current_state)
            prev_node.a_values[action_idx] += disc_rollout_value
            return disc_rollout_value
        else:
            # Node in the tree
            state_id = new_state_id
            if terminal:
                return 0
            else:
                disc_value = self.discount * self.simulate(state_id)
                # BackPropagate
                # since I only need action nodes for action selection I don't care about the value of State nodes
                prev_node.a_values[action_idx] += disc_value
                return disc_value

    def rollout(self, current_state) -> Union[int, float]:
        terminal = False
        r = 0
        budget = self.computational_budget
        while not terminal and budget != 0:
            # random policy
            actions = self.environment.get_actions(current_state)
            chosen_action = np.random.choice(actions)
            current_state, r, terminal, _, _ = self.environment.step(current_state, chosen_action)
            budget -= 1
        return r
