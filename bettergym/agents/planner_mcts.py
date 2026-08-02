from __future__ import annotations

import time
from typing import Union, Any, Dict, Callable

import numpy as np

try:
    from MCTS_VO.bettergym.agents.planner import Planner
    from MCTS_VO.bettergym.better_gym import BetterGym
    from MCTS_VO.bettergym.compiled_utils import fused_rollout
except ModuleNotFoundError:
    from bettergym.agents.planner import Planner
    from bettergym.better_gym import BetterGym
    from bettergym.compiled_utils import fused_rollout

# Passed to fused_rollout to switch collision checking off: it loops over the
# obstacles it is given, so an empty set is step_no_check_coll semantics.
_NO_OBSTACLES = np.empty((0, 2), dtype=np.float64)

class ActionNode:
    def __init__(self, action: Any):
        self.action: Any = action
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
        # if node_id == 0:
        #     plot_vo(state, environment.gym_env.config)
        acts = environment.get_actions(state)
        self.actions = [ActionNode(a) for a in acts]
        self.num_visits_actions = np.zeros_like(self.actions, dtype=np.float64)
        self.a_values = np.zeros_like(self.actions, dtype=np.float64)
        self.num_visits: int = 0


class RolloutStateNode:
    def __init__(self, state):
        self.state = state

class Mcts(Planner):
    def __init__(
            self,
            num_sim: int,
            c: float,
            environment: BetterGym,
            computational_budget: int,
            rollout_policy: Callable,
            discount: float = 1.0,
            logger=None,
            rollout_eps: float = None,
            rollout_collision_check: bool = True,
    ):
        super().__init__(environment)
        self.num_sim: int = num_sim
        self.exploration_constant: float | int = c
        self.c = None
        self.computational_budget: int = computational_budget
        self.discount: float | int = discount
        self.rollout_policy = rollout_policy
        # `fused_rollout` inlines the epsilon_uniform_uniform policy, so it needs
        # the same eps that rollout_policy was built with. Passing it explicitly
        # rather than digging it out of the partial keeps the two in one place -
        # and if it is not given, the compiled path is simply not used.
        self.rollout_eps = rollout_eps
        self.rollout_collision_check = rollout_collision_check

        self.id_to_state_node = None
        self.num_visits_actions = None
        self.a_values = None
        self.state_actions = None
        self.last_id = None
        self.info = None
        self.logger = logger

    def initialize_variables(self):
        self.id_to_state_node: dict[int, StateNode] = {}
        self.last_id = -1
        self.info = {
            "trajectories": [],
            "q_values": [],
            "actions": [],
            "visits": [],
            "rollout_values": [],
        }
        # Depth statistics. These are plain int attributes rather than entries of
        # self.info: they are updated only where the tree actually grows (node
        # creation) and at the end of a rollout, never inside simulate()'s per
        # visit path, so they cost a single integer comparison on events that are
        # orders of magnitude rarer than node visits.
        # - max_tree_depth: depth of the deepest node of the search tree
        #   (the root is at depth 0)
        # - max_rollout_depth: length in steps of the longest rollout
        # - max_total_depth: deepest state ever reached, i.e. the depth at which
        #   a rollout started plus its length
        self.max_tree_depth = 0
        self.max_rollout_depth = 0
        self.max_total_depth = 0

    def get_id(self):
        self.last_id += 1
        return self.last_id

    def plan(self, initial_state: Any, available_time: float):
        initial_time = time.time()

        self.initialize_variables()
        root_id = self.get_id()
        root_node = StateNode(self.environment, initial_state, root_id)

        self.id_to_state_node[root_id] = root_node
        simulate = True
        sn = 1
        while simulate:
            sim_time = time.time()
            self.info["trajectories"].append(np.array([initial_state.x]))
            # root should be at depth 0
            total_reward = self.simulate(state_id=root_id, depth=0)
            self.info["rollout_values"].append(total_reward)
            final_time = time.time() - initial_time
            # self.logger.info(f"Sim Time: {time.time() - sim_time}")
            sn += 1
            simulate = final_time < available_time

        q_vals = np.divide(
            root_node.a_values,
            root_node.num_visits_actions,
            out=np.full_like(root_node.a_values, -np.inf),
            where=root_node.num_visits_actions != 0,
        )
        # DEBUG INFORMATION
        self.info["q_values"] = q_vals
        self.info["actions"] = root_node.actions
        self.info["visits"] = root_node.num_visits_actions
        self.info["simulations"] = sn
        self.info["max_tree_depth"] = self.max_tree_depth
        self.info["max_rollout_depth"] = self.max_rollout_depth
        self.info["max_total_depth"] = self.max_total_depth
        
        # randomly choose between actions which have the maximum q value
        action_idx = np.random.choice(np.flatnonzero(q_vals == np.max(q_vals)))
        action = root_node.actions[action_idx].action
        return action, self.info

    def simulate(self, state_id: int, depth: int):
        node = self.id_to_state_node[state_id]
        node.num_visits += 1
        current_state = node.state

        # UCB
        # Q + c * sqrt(ln(Parent_Visit)/Child_visit)
        q_vals = np.divide(
            node.a_values,
            node.num_visits_actions,
            out=np.full_like(node.a_values, np.inf),
            where=node.num_visits_actions != 0,
        )

        ucb_scores = q_vals + self.exploration_constant * np.sqrt(
            np.divide(
                np.log(node.num_visits),
                node.num_visits_actions,
                out=np.full_like(node.num_visits_actions, np.inf),
                where=node.num_visits_actions != 0,
            )
        )

        # randomly choose between actions which have the maximum ucb value
        action_idx = np.random.choice(np.flatnonzero(ucb_scores == np.max(ucb_scores)))

        # get action corresponding to the index
        action_node = node.actions[action_idx]
        action = action_node.action
        # increase action visits
        node.num_visits_actions[action_idx] += 1

        current_state, r, terminal, _, _ = self.environment.step(current_state, action)
        new_state_id = action_node.state_to_id.get(current_state, None)
        self.info["trajectories"][-1] = np.vstack(
            (
                self.info["trajectories"][-1],
                current_state.x,
            )
        )

        prev_node = node
        if (
                new_state_id is None
                and depth + 1 < self.computational_budget
                and not terminal
        ):
            # Leaf Node
            # The tree grows exactly here, so the deepest node is tracked at node
            # creation instead of at every visit: this runs once per new node,
            # next to a StateNode construction that already enumerates actions
            # and allocates two arrays.
            if depth + 1 > self.max_tree_depth:
                self.max_tree_depth = depth + 1
            state_id = self.get_id()
            # Initialize State Data
            node = StateNode(self.environment, current_state, state_id)
            self.id_to_state_node[state_id] = node
            action_node.state_to_id[current_state] = state_id
            node.num_visits += 1
            # Do Rollout
            # the value returned by the rollout is already discounted
            total_rwrd = r + self.discount * self.rollout(current_state, depth + 1)
            prev_node.a_values[action_idx] += total_rwrd
            return total_rwrd
        else:
            # Node in the tree
            state_id = new_state_id
            if terminal or depth + 1 >= self.computational_budget:
                prev_node.a_values[action_idx] += r
                return r
            else:
                total_rwrd = r + self.discount * self.simulate(state_id, depth + 1)
                # BackPropagate
                # since I only need action nodes for action selection I don't care about the value of State nodes
                prev_node.a_values[action_idx] += total_rwrd
                return total_rwrd

    def rollout(self, current_state, curr_depth) -> Union[int, float]:
        """
        Roll out to the computational budget and return the discounted return.

        Dispatches to the compiled `fused_rollout` when it can - which is
        whenever the eps of the rollout policy was declared - and to the Python
        implementation otherwise. `rollout_python` is kept as the readable
        reference the compiled version is checked against, and as the path that
        still records the per-step trajectory.

        The two agree distributionally rather than bit for bit: the Python
        version draws its epsilon coin from Python's `random` and its speeds
        from numba's generator, while the fused one draws both from numba's.
        """
        if self.rollout_eps is None:
            return self.rollout_python(current_state, curr_depth)

        depth = self.computational_budget - curr_depth
        if depth <= 0:
            return 0.0

        env = self.environment.gym_env
        config = env.config
        obstacles = current_state.obstacles
        if self.rollout_collision_check and len(obstacles[0]) != 0:
            obs_xy = np.ascontiguousarray(obstacles[0][:, :2], dtype=np.float64)
        else:
            obs_xy = _NO_OBSTACLES

        total_reward = fused_rollout(
            current_state.x,
            current_state.goal,
            obs_xy,
            config.dt,
            config.max_angle_change,
            config.max_speed,
            config.robot_radius,
            env.max_eudist,
            depth,
            self.discount,
            self.rollout_eps,
        )

        # The depth statistics cannot see inside the compiled call, so they
        # record the budget the rollout was given. It is an upper bound: a
        # rollout that reaches the goal or hits an obstacle stops early.
        if depth > self.max_rollout_depth:
            self.max_rollout_depth = depth
        total_depth = curr_depth + depth
        if total_depth > self.max_total_depth:
            self.max_total_depth = total_depth

        return total_reward

    def rollout_python(self, current_state, curr_depth) -> Union[int, float]:
        terminal = False
        trajectory = []
        total_reward = 0
        starting_depth = 0
        while not terminal and curr_depth + starting_depth != self.computational_budget:
            chosen_action = self.rollout_policy(RolloutStateNode(current_state), self)
            current_state, r, terminal, _, _ = self.environment.step(
                current_state, chosen_action
            )
            total_reward += r * pow(self.discount, starting_depth)
            trajectory.append(current_state.x)  # store state history
            starting_depth += 1

        # starting_depth is already maintained by the loop above, so the rollout
        # length costs nothing extra: it is only read here, once per rollout.
        if starting_depth > self.max_rollout_depth:
            self.max_rollout_depth = starting_depth
        total_depth = curr_depth + starting_depth
        if total_depth > self.max_total_depth:
            self.max_total_depth = total_depth

        self.info["trajectories"][-1] = np.vstack(
            (self.info["trajectories"][-1], np.array(trajectory))
        )
        return total_reward
