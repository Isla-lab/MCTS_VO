import math
import random
from functools import partial
from typing import Any, Callable

# import graphviz
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

from bettergym.agents.planner import Planner


def uniform(node: Any, planner: Planner):
    current_state = node.state
    available_actions = planner.environment.get_actions(current_state)
    return available_actions.sample()


def uniform_discrete(node: Any, planner: Planner):
    current_state = node.state
    actions = planner.environment.get_actions(current_state)
    return random.choice(actions)


@njit
def compute_towards_goal_jit(x: np.ndarray, goal: np.ndarray, max_angle_change: float, std_angle_rollout: float,
                             min_speed: float, max_speed: float):
    mean_angle = np.arctan2(goal[1] - x[1], goal[0] - x[0])
    angle = np.random.normal(mean_angle, std_angle_rollout)
    linear_velocity = np.random.uniform(
        low=min_speed,
        high=max_speed
    )
    # Make sure angle is within range of -π to π
    min_angle = x[2] - max_angle_change
    max_angle = x[2] + max_angle_change
    angle = max(min(angle, max_angle), min_angle)
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return np.array([linear_velocity, angle])


def towards_goal(node: Any, planner: Planner, std_angle_rollout: float):
    config = planner.environment.config
    return compute_towards_goal_jit(node.state.x, node.state.goal, config.max_angle_change, std_angle_rollout,
                                    config.min_speed,
                                    config.max_speed)


def towards_goal_discrete(node: Any, planner: Planner, std_angle_rollout: float):
    config = planner.environment.config
    action = compute_towards_goal_jit(node.state.x, node.state.goal, config.max_angle_change, std_angle_rollout,
                                      config.min_speed, config.max_speed)
    discrete_actions = planner.environment.get_actions(node.state)
    return bin_action(action, discrete_actions)


@njit
def compute_uniform_towards_goal_jit(x: np.ndarray, goal: np.ndarray, max_angle_change: float, min_speed: float,
                                     max_speed: float, amplitude: float):
    mean_angle = np.arctan2(goal[1] - x[1], goal[0] - x[0])
    linear_velocity = np.random.uniform(
        low=min_speed,
        high=max_speed
    )
    # Make sure angle is within range of -π to π
    min_angle = x[2] - max_angle_change
    max_angle = x[2] + max_angle_change
    angle = np.random.uniform(
        low=mean_angle - amplitude,
        high=mean_angle + amplitude
    )

    angle = max(min(angle, max_angle), min_angle)
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return np.array([linear_velocity, angle])


def uniform_towards_goal(node: Any, planner: Planner, amplitude: float):
    config = planner.environment.config
    return compute_uniform_towards_goal_jit(node.state.x, node.state.goal, config.max_angle_change, config.min_speed,
                                            config.max_speed, amplitude)


def bin_action(action, bins):
    diff_vector = np.linalg.norm(action - bins, axis=1)
    idx = diff_vector.argmin()
    return bins[idx]


def uniform_towards_goal_discrete(node: Any, planner: Planner, amplitude: float):
    config = planner.environment.config
    action = compute_uniform_towards_goal_jit(node.state.x, node.state.goal, config.max_angle_change, config.min_speed,
                                              config.max_speed, amplitude)
    discrete_actions = planner.environment.get_actions(node.state)
    return bin_action(action, discrete_actions)


def epsilon_greedy(eps: float, other_func: Callable, node: Any, planner: Planner):
    """
    :param node:
    :param eps: defines the probability of acting according to other_func
    :param other_func:
    :param planner:
    :return:
    """
    prob = random.random()
    if prob <= 1 - eps:
        return other_func(node, planner)
    else:
        return uniform(node, planner)


def binary_policy(node: Any, planner: Planner):
    if len(node.actions) == 1:
        return uniform(node, planner)
    else:
        sorted_actions = [a for _, a in sorted(zip(node.a_values, node.actions), key=lambda pair: pair[0])]
        return np.mean([sorted_actions[0].action, sorted_actions[1].action], axis=0)


def voronoi(actions: np.ndarray, q_vals: np.ndarray, sample_centered: Callable):
    N_SAMPLE = 1000
    valid = False
    n_iter = 0
    # find the index of the action with the highest Q-value
    best_action_index = np.argmax(q_vals)

    # get the action with the highest Q-value
    best_action = actions[best_action_index]
    tmp_best = None
    tmp_dist = np.inf
    while not valid:
        if n_iter >= 100:
            return tmp_best

        # generate random points centered around the best action
        points = sample_centered(center=best_action, number=N_SAMPLE)

        # compute the Euclidean distances between each point and each action
        # column -> actions
        # rows -> points
        dists = cdist(points, actions, 'euclidean')

        # find the distances between each point and the best action
        best_action_distances = dists[:, best_action_index]

        # repeat the distances for each action except the best action (necessary for doing `<=` later)
        best_action_distances_rep = np.tile(best_action_distances, (dists.shape[1] - 1, 1)).T

        # remove the column for the best action from the distance matrix
        # dists = np.hstack((dists[:, :best_action_index], dists[:, best_action_index + 1:]))
        dists = np.delete(dists, best_action_index, axis=1)

        # find the closest action to each point
        closest = best_action_distances_rep <= dists

        # find the rows where all distances to other actions are greater than the distance to the best action
        all_true_rows = np.where(np.all(closest, axis=1))[0]

        # find the index of the point closest to the best action among the valid rows
        valid_points = best_action_distances[all_true_rows]
        if len(valid_points >= 0):
            closest_point_idx = np.argmin(valid_points)
            # return the closest point to the best action
            return points[closest_point_idx]
        else:
            closest_point_idx = np.argmin(best_action_distances)
            if d := best_action_distances[closest_point_idx] < tmp_dist:
                # return the closest point to the best action
                tmp_best = points[closest_point_idx]
                tmp_dist = d
        n_iter += 1
        del points, dists, best_action_distances, best_action_distances_rep, closest, all_true_rows, valid_points


@njit
def clip_act(chosen: np.ndarray, max_angle_change: float, x: np.ndarray, allow_negative: bool):
    if allow_negative:
        chosen[:, 0] = (chosen[:, 0] % 0.4) - 0.1
    else:
        chosen[:, 0] = chosen[:, 0] % 0.3
    min_available_angle = x[2] - max_angle_change
    max_available_angle = x[2] + max_angle_change
    # Make sure angle is within range of -min_angle to max_angle
    chosen[:, 1] = chosen[:, 1] % (max_available_angle - min_available_angle) + min_available_angle
    # Make sure angle is within range of -π to π
    chosen[:, 1] = (chosen[:, 1] + math.pi) % (2 * math.pi) - math.pi
    return chosen


def voo(eps: float, sample_centered: Callable, node: Any, planner: Planner):
    prob = random.random()
    if prob <= 1 - eps and len(node.actions) != 0:
        config = planner.environment.gym_env.config
        return voronoi(
            np.array([node.action for node in node.actions]),
            node.a_values,
            partial(sample_centered, clip_fn=partial(clip_act, max_angle_change=config.max_angle_change, x=node.state.x,
                                                     allow_negative=True))
        )
    else:
        return uniform(node, planner)

# def visualize_state_node(node: StateNode, father: str | None, g: graphviz.Digraph, n: int, planner):
#     # add the node its self
#     g.attr('node', shape='circle')
#     name = f"node{n}"
#     g.node(name, f"ID={node.id}\n{node.state.x}\nn={node.num_visits}")
#     g.attr('node', fillcolor='white', style='filled')
#     # for root node father is None
#     if father is not None:
#         g.edge(father, name)
#     n += 1
#     angles = [a.action[1] for a in node.actions]
#     min_angle = np.min(angles)
#     max_angle = np.max(angles)
#     # add its child nodes
#     for idx, action_node in enumerate(node.actions):
#         if action_node.action[1] == max_angle or action_node.action[1] == min_angle:
#             # add the node its self
#             g.attr('node', shape='box')
#             child_name = f"node{n}"
#             g.node(child_name, f"{action_node.action}\nn={node.num_visits_actions[idx]}\nQ={(node.a_values[idx] / node.num_visits_actions[idx]):.3f}")
#             # connect to father node
#             g.edge(name, child_name)
#             n += 1
#             # add its child nodes
#             for state_node_id in action_node.state_to_id.values():
#                 father = child_name
#                 state_node = planner.id_to_state_node[state_node_id]
#                 n = visualize_state_node(state_node, father, g, n, planner)
#
#     # to avoid losing the updated n value every time the function end returns the most updated n value
#     return n
#
#
# def visualize_tree(planner, n):
#     filename = f'mcts_{n}'
#     # g = graphviz.Digraph('g', filename=f'{filename}.svg', directory='debug/tree')
#     g = graphviz.Digraph('g', engine='dot')
#     root: StateNode = planner.id_to_state_node[0]
#     node_counter = 0
#     visualize_state_node(root, None, g, node_counter, planner)
#
#     # save gv file
#     g.save(directory='debug/tree', filename=f'{filename}.gv')
#     # g.render(f'{filename}', format="svgz")
#     # # render gv file to an svg
#     with open(f'debug/tree/{filename}.svg', 'w') as f:
#         subprocess.Popen(['dot', '-Tsvg', f'debug/tree/{filename}.gv'], stdout=f)
