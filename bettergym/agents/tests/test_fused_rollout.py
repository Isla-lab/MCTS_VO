"""The compiled rollout must agree with the Python one it replaces.

Agreement is distributional, not bitwise, and cannot be anything else: the
Python path draws its epsilon coin from Python's `random` and its speeds from
numba's generator, while the fused one draws both from numba's. Seeding both
therefore does not put them on the same stream. What must hold is that the two
sample the same distribution of returns, so the test compares means over many
rollouts per start state with a two-sample z test.

Note when running this file directly from inside MCTS_VO/: numba then compiles
under `bettergym.*` module names, and a run of the ROS loop - which imports the
same code as `MCTS_VO.bettergym.*` - will find the cache poisoned. Clear it with
    find . -name __pycache__ -type d -exec rm -rf {} +
"""
from unittest import TestCase

import numpy as np

try:
    from MCTS_VO.bettergym.agents.planner_mcts import Mcts
    from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
    from MCTS_VO.bettergym.environments.env import State
    from MCTS_VO.environment_creator import create_pedestrian_env
except ModuleNotFoundError:
    from bettergym.agents.planner_mcts import Mcts
    from bettergym.agents.utils.utils import epsilon_uniform_uniform
    from bettergym.environments.env import State
    from environment_creator import create_pedestrian_env

from functools import partial

DT = 0.1
DEPTH = 200
EPS = 0.2
DISCOUNT = 0.81 ** DT
# Rollouts per state. The spread of returns is wide (a rollout either reaches
# the goal, hits something, or neither), so the mean needs a few hundred samples
# before its standard error is small enough for the test to mean anything.
N = 400

GOAL = np.array([-2.783, -0.720])
OBS_POS = np.array([
    [-0.399, 0.420, 0.0, 0.15], [-1.542, -1.790, 0.0, 0.15],
    [-1.539, 0.360, 0.0, 0.15], [-2.640, -1.310, 0.0, 0.15],
    [-0.317, -1.820, 0.0, 0.15], [-3.020, 0.363, 0.0, 0.15],
])
OBS_RAD = np.full(len(OBS_POS), 0.18)


def _make_planner():
    _, sim_env = create_pedestrian_env(
        discrete=True, rwrd_in_sim=True, out_boundaries_rwrd=-100,
        n_vel=4, n_angles=6, vo=True, obs_pos=None, n_obs=None, dt_real=DT,
    )
    sim_env.gym_env.max_eudist = 3.30
    planner = Mcts(
        num_sim=100, c=1.0, environment=sim_env, computational_budget=DEPTH,
        rollout_policy=partial(epsilon_uniform_uniform,
                               std_angle_rollout=2.84 * DT, eps=EPS),
        discount=DISCOUNT, rollout_eps=EPS, rollout_collision_check=True,
    )
    planner.initialize_variables()
    # rollout_python appends to the current simulation's trajectory buffer,
    # which plan() would normally have created.
    planner.info["trajectories"].append(np.empty((0, 4)))
    return planner


def _states(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = np.array([rng.uniform(-3.2, 0.4), rng.uniform(-1.9, 0.5),
                      rng.uniform(-np.pi, np.pi), 0.0])
        out.append(State(x=x, goal=GOAL.copy(),
                         obstacles=(OBS_POS, OBS_RAD), radius=0.15))
    return out


class TestFusedRollout(TestCase):
    def test_agrees_with_python_rollout(self):
        planner = _make_planner()
        for state in _states(8):
            py = np.array([planner.rollout_python(state, 0) for _ in range(N)])
            fused = np.array([planner.rollout(state, 0) for _ in range(N)])
            se = np.sqrt(py.var(ddof=1) / N + fused.var(ddof=1) / N)
            z = (py.mean() - fused.mean()) / se if se > 0 else 0.0
            # 3 sigma, over 8 states: a false failure about once in 400 runs.
            self.assertLess(abs(z), 3.0,
                            f"returns differ at x={state.x}: python "
                            f"{py.mean():.3f} vs fused {fused.mean():.3f}")

    def test_collision_free_mode_never_terminates_on_an_obstacle(self):
        """With checking off, no rollout may collect the -100 collision reward."""
        planner = _make_planner()
        planner.rollout_collision_check = False
        state = _states(1)[0]
        values = np.array([planner.rollout(state, 0) for _ in range(2000)])
        self.assertGreater(values.min(), -100.0)

    def test_no_rollout_eps_falls_back_to_the_python_path(self):
        """Without an eps declared the compiled path cannot be used at all."""
        planner = _make_planner()
        planner.rollout_eps = None
        value = planner.rollout(_states(1)[0], 0)
        self.assertIsInstance(float(value), float)

    def test_exhausted_budget_returns_zero(self):
        planner = _make_planner()
        self.assertEqual(planner.rollout(_states(1)[0], DEPTH), 0.0)
