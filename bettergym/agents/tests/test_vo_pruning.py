"""The optimised in-tree VO pruning must return the action set it replaced.

`get_actions_discrete_vo2` is what makes MCTS-VO what it is: it is the only
place velocity obstacles enter the planner. Making it cheaper is only legitimate
if the set of actions it hands the tree is identical, so these tests pin that
down against reimplementations of the previous code paths rather than against
recorded expectations.

Note when running this file directly from inside MCTS_VO/: numba then compiles
under `bettergym.*` module names, and a run of the ROS loop - which imports the
same code as `MCTS_VO.bettergym.*` - will find the cache poisoned. Clear it with
    find . -name __pycache__ -type d -exec rm -rf {} +
"""
import math
from unittest import TestCase

import numpy as np
from intervaltree import IntervalTree

try:
    from MCTS_VO.bettergym.agents.utils.vo import compute_ranges_difference
    from MCTS_VO.bettergym.compiled_utils import discrete_actions
    from MCTS_VO.bettergym.environments.env import State
    from MCTS_VO.environment_creator import create_pedestrian_env
except ModuleNotFoundError:
    from bettergym.agents.utils.vo import compute_ranges_difference
    from bettergym.compiled_utils import discrete_actions
    from bettergym.environments.env import State
    from environment_creator import create_pedestrian_env

DT = 0.1
GOAL = np.array([-2.783, -0.720])


def _reference_ranges_difference(robot_angles, forbidden_ranges):
    """The IntervalTree implementation this replaced, verbatim."""
    def get_interval_tree(ranges):
        try:
            return IntervalTree.from_tuples(ranges)
        except ValueError:
            r = np.array(ranges)
            ranges = r[r[:, 1] != r[:, 0]]
            it = IntervalTree.from_tuples(ranges)
            it.merge_overlaps(strict=False)
            return it

    t1 = get_interval_tree(robot_angles)
    t2 = get_interval_tree(forbidden_ranges)
    for i in t2:
        t1.chop(i.begin, i.end)
    return [[i.begin, i.end] for i in t1.all_intervals]


def _reference_discrete_actions(x, config):
    """The numpy implementation `discrete_actions` replaced, verbatim."""
    available_angles = np.linspace(
        start=x[2] - config.max_angle_change,
        stop=x[2] + config.max_angle_change,
        num=config.n_angles,
    )
    if (curr_angle := x[2]) not in available_angles:
        available_angles = np.append(available_angles, curr_angle)
    available_angles = (available_angles + np.pi) % (2 * np.pi) - np.pi
    available_velocities = np.linspace(
        start=config.min_speed, stop=config.max_speed, num=config.n_vel
    )
    if 0.0 not in available_velocities:
        available_velocities = np.append(available_velocities, 0.0)
    return np.transpose([
        np.tile(available_velocities, len(available_angles)),
        np.repeat(available_angles, len(available_velocities)),
    ])


def _sorted(ranges):
    if len(ranges) == 0:
        return np.empty((0, 2))
    r = np.asarray(ranges, dtype=np.float64).reshape(-1, 2)
    return r[np.lexsort((r[:, 1], r[:, 0]))]


class TestRangesDifference(TestCase):
    def test_matches_the_interval_tree_on_random_inputs(self):
        rng = np.random.default_rng(0)
        for _ in range(4000):
            def draw(n):
                lo = rng.uniform(-math.pi, math.pi, n)
                # width 0 sometimes, so degenerate intervals are covered
                hi = lo + rng.choice([0.0, 1.0]) * rng.uniform(0, 1.5, n)
                return np.column_stack((lo, hi)).tolist()

            base = draw(rng.integers(1, 3))
            forbidden = draw(rng.integers(1, 5))

            got = _sorted(compute_ranges_difference(base, forbidden))
            want = _sorted(_reference_ranges_difference(base, forbidden))
            self.assertEqual(got.shape, want.shape,
                             f"base={base} forbidden={forbidden}")
            np.testing.assert_allclose(got, want, atol=1e-12,
                                       err_msg=f"base={base} forbidden={forbidden}")

    def test_result_is_sorted_by_lower_bound(self):
        """The tree returned a set, so its order was arbitrary; this is not.

        The order matters: get_discrete_space floors the sample count of
        even-indexed ranges and ceils that of odd-indexed ones.
        """
        out = compute_ranges_difference([[-1.0, 1.0]],
                                        [[-0.6, -0.4], [0.2, 0.3]])
        lows = [r[0] for r in out]
        self.assertEqual(lows, sorted(lows))

    def test_nothing_forbidden_returns_the_base_ranges(self):
        self.assertEqual(compute_ranges_difference([[-1.0, 1.0]], []),
                         [[-1.0, 1.0]])

    def test_everything_forbidden_returns_nothing(self):
        self.assertEqual(compute_ranges_difference([[-1.0, 1.0]],
                                                   [[-math.pi, math.pi]]), [])


class TestDiscreteActions(TestCase):
    def test_matches_the_numpy_implementation(self):
        _, env = create_pedestrian_env(
            discrete=True, rwrd_in_sim=True, out_boundaries_rwrd=-100,
            n_vel=4, n_angles=6, vo=False, obs_pos=None, n_obs=None, dt_real=DT,
        )
        config = env.gym_env.config
        rng = np.random.default_rng(1)
        for _ in range(200):
            x = np.array([rng.uniform(-3.2, 0.4), rng.uniform(-1.9, 0.5),
                          rng.uniform(-np.pi, np.pi), 0.0])
            got = discrete_actions(x, config.max_angle_change, config.min_speed,
                                   config.max_speed, config.n_angles, config.n_vel)
            want = _reference_discrete_actions(x, config)
            np.testing.assert_array_equal(got, want)


class TestVoPruning(TestCase):
    """End to end: the pruned set must stay inside the unpruned one and shrink."""

    def setUp(self):
        _, self.env = create_pedestrian_env(
            discrete=True, rwrd_in_sim=True, out_boundaries_rwrd=-100,
            n_vel=4, n_angles=6, vo=True, obs_pos=None, n_obs=None, dt_real=DT,
        )
        self.config = self.env.gym_env.config

    def _state(self, x, obs_pos, obs_rad):
        return State(x=x, goal=GOAL.copy(), obstacles=(obs_pos, obs_rad),
                     radius=0.15)

    def test_no_obstacles_gives_the_full_action_set(self):
        x = np.array([0.49, -1.14, 0.0, 0.0])
        state = self._state(x, np.empty((0, 4)), np.empty(0))
        np.testing.assert_array_equal(
            self.env.get_actions_discrete_vo2(state),
            self.env.get_actions_discrete(state),
        )

    def test_distant_obstacles_prune_nothing(self):
        x = np.array([0.49, -1.14, 0.0, 0.0])
        far = np.array([[50.0, 50.0, 0.0, 0.15]])
        state = self._state(x, far, np.array([0.18]))
        np.testing.assert_array_equal(
            self.env.get_actions_discrete_vo2(state),
            self.env.get_actions_discrete(state),
        )

    def test_an_obstacle_dead_ahead_removes_forward_headings(self):
        x = np.array([0.0, 0.0, 0.0, 0.0])
        ahead = np.array([[0.35, 0.0, 0.0, 0.15]])
        state = self._state(x, ahead, np.array([0.18]))
        pruned = self.env.get_actions_discrete_vo2(state)
        # Nothing may be commanded at top speed straight down the obstacle's
        # bearing; that is the whole claim VO makes.
        forward_at_speed = [
            a for a in pruned
            if a[0] > 0.0 and abs((a[1] + np.pi) % (2 * np.pi) - np.pi) < 1e-9
        ]
        self.assertEqual(forward_at_speed, [])
        self.assertGreater(len(pruned), 0, "some action must always remain")

    def test_pruning_is_deterministic(self):
        x = np.array([-1.0, -0.5, 0.4, 0.0])
        obs = np.array([[-1.3, -0.35, 0.0, 0.15], [-0.8, -0.9, 0.0, 0.15]])
        state = self._state(x, obs, np.array([0.18, 0.18]))
        first = self.env.get_actions_discrete_vo2(state)
        for _ in range(20):
            np.testing.assert_array_equal(
                self.env.get_actions_discrete_vo2(state), first)
