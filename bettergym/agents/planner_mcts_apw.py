import math
from typing import Union, Any, Dict, Callable

import numpy as np
from gymnasium import Space

from bettergym.agents.planner import Planner
from bettergym.better_gym import BetterGym


class ActionNode:
    def __init__(self, action: np.ndarray):
        self.action = action
        self.action_bytes = action.tobytes()
        self.state_to_id: Dict[Any, int] = {}

    def __hash__(self):
        return hash(self.action_bytes)

    def __repr__(self):
        return np.array2string(self.action)

    def __eq__(self, other):
        if isinstance(other, ActionNode) and hash(self) == hash(other):
            return True
        return False


class StateNode:
    def __init__(self, environment, state, node_id):
        self.id = node_id
        self.state = state
        self.actions = []
        self.num_visits_actions = np.array([], dtype=np.float64)
        self.a_values = np.array([], dtype=np.float64)
        self.num_visits: int = 0


class MctsApw(Planner):
    def __init__(self, num_sim: int, c: float | int, environment: BetterGym, computational_budget: int, k: float | int,
                 alpha: float | int, action_expansion_function: Callable, rollout_policy: Callable,
                 discount: float | int = 1):
        """
        Mcts algorithm with Action Progressive Widening
        :param num_sim: number of simulations
        :param c: exploration constant of the UCB
        :param environment: the simulated environments
        :param computational_budget: maximum rollout depth
        :param k: first parameter of the action progressive widening
        :param alpha: second parameter of the action progressive widening
        :param action_expansion_function: function to choose which action to add to the tree
        :param discount: the discount factor of the mdp
        """
        super().__init__(environment)
        self.id_to_state_node: dict[int, StateNode] = {}
        self.num_sim: int = num_sim
        self.c: float | int = c
        self.computational_budget: int = computational_budget
        self.discount: float | int = discount
        self.k = k
        self.alpha = alpha
        self.action_expansion_function = action_expansion_function
        self.rollout_policy = rollout_policy

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

        if len(node.actions) == 0:
            # Since we don't have actions in the node, we'll sample one at random
            available_actions: Space = self.environment.get_actions(current_state)
            new_action: np.ndarray = available_actions.sample()
            # add child
            new_action_node = ActionNode(new_action)
            node.actions.append(new_action_node)
            node.num_visits_actions = np.append(node.num_visits_actions, 0)
            node.a_values = np.append(node.a_values, 0)

        elif len(node.actions) <= math.ceil(self.k * (node.num_visits ** self.alpha)):
            new_action: np.ndarray = self.action_expansion_function(current_state, self)

            # add child
            new_action_node = ActionNode(new_action)
            node.actions.append(new_action_node)
            # remove duplicate nodes
            node.actions = list(dict.fromkeys(node.actions))

            if len(node.num_visits_actions) != len(node.actions):
                # the node we added was a duplicate node
                node.num_visits_actions = np.append(node.num_visits_actions, 0)
                node.a_values = np.append(node.a_values, 0)

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
            chosen_action = self.rollout_policy(current_state, self)
            current_state, r, terminal, _, _ = self.environment.step(current_state, chosen_action)
            budget -= 1
        return r
