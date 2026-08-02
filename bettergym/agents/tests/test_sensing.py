"""The compiled sensing kernel must fit what skimage's RANSAC fits.

The pipeline still has its two stages - cluster the returns, then fit each
cluster with RANSAC - so only the implementations changed. The RANSAC fit is
checked directly against skimage on the same clusters; the clustering is checked
on scans whose grouping is unambiguous, since the two criteria are different in
kind (HDBSCAN in two dimensions against the adaptive breakpoint over the ordered
ranges) and cannot be expected to agree case by case.

Note when running this file directly from inside MCTS_VO/: numba then compiles
under `bettergym.*` module names, and a run of the ROS loop - which imports the
same code as `MCTS_VO.bettergym.*` - will find the cache poisoned. Clear it with
    find . -name __pycache__ -type d -exec rm -rf {} +
"""
from unittest import TestCase

import numpy as np
from skimage.measure import CircleModel, ransac

try:
    from MCTS_VO.bettergym.compiled_utils import cluster_and_fit_circles, ransac_circle
except ModuleNotFoundError:
    from bettergym.compiled_utils import cluster_and_fit_circles, ransac_circle

ANGLE_INCREMENT = np.deg2rad(1.0)
LAMBDA = np.deg2rad(30.0)
SIGMA = 0.01
MIN_POINTS = 3
MAX_RADIUS = 0.5
RESIDUAL_THRESHOLD = 0.1
MAX_TRIALS = 100
STOP_PROBABILITY = 0.99


def _arc(cx, cy, r, a0, a1, n, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    th = np.linspace(a0, a1, n)
    rr = r + rng.normal(0, noise, n) if noise else r
    return np.column_stack((cx + rr * np.cos(th), cy + rr * np.sin(th)))


def _fit(points):
    px = np.ascontiguousarray(points[:, 0])
    py = np.ascontiguousarray(points[:, 1])
    return ransac_circle(px, py, 0, len(points), MAX_TRIALS,
                         RESIDUAL_THRESHOLD, STOP_PROBABILITY)


class TestRansacCircle(TestCase):
    def test_recovers_a_noiseless_arc_exactly(self):
        model = _fit(_arc(-1.0, 0.5, 0.1, 0.3, 1.5, 9))
        np.testing.assert_allclose(model[:3], [-1.0, 0.5, 0.1], atol=1e-9)
        self.assertLess(model[3], 1e-9)

    def test_agrees_with_skimage_on_noisy_arcs(self):
        """Same clusters, same parameters: the fits must land in the same place.

        RANSAC is randomised, so this is agreement to within the noise, not to
        the last bit.
        """
        rng = np.random.default_rng(0)
        for k in range(40):
            cx, cy = rng.uniform(-3, 0), rng.uniform(-2, 1)
            r = rng.uniform(0.08, 0.2)
            a0 = rng.uniform(-np.pi, np.pi)
            pts = _arc(cx, cy, r, a0, a0 + rng.uniform(0.8, 2.0),
                       rng.integers(6, 20), noise=SIGMA, seed=k)

            np.random.seed(k)
            mine = _fit(pts)
            theirs, _ = ransac(pts, CircleModel, max_trials=MAX_TRIALS,
                               min_samples=3, residual_threshold=RESIDUAL_THRESHOLD,
                               stop_probability=STOP_PROBABILITY)
            self.assertIsNotNone(theirs)

            centre_err = np.linalg.norm(mine[:2] - theirs.params[:2])
            self.assertLess(centre_err, 0.05,
                            f"centres differ by {centre_err:.3f} m on cluster {k}")
            self.assertLess(abs(mine[2] - theirs.params[2]), 0.05,
                            f"radii differ on cluster {k}")

    def test_collinear_points_yield_no_circle(self):
        pts = np.column_stack((np.linspace(0, 1, 8), np.zeros(8)))
        self.assertLess(_fit(pts)[2], 0.0)

    def test_too_few_points_yield_no_circle(self):
        self.assertLess(_fit(np.array([[0.0, 0.0], [1.0, 0.0]]))[2], 0.0)


def _scan(clusters, gaps=True):
    """Build (dist, idx, points) for arcs at given bearings, robot at the origin."""
    dist, idx, pts = [], [], []
    for start_beam, centre, r in clusters:
        n = 8
        for k in range(n):
            beam = start_beam + k
            a = beam * ANGLE_INCREMENT
            # Range to the near surface of the circle along this bearing
            d = np.hypot(*centre) - r
            dist.append(d)
            idx.append(beam)
            pts.append([d * np.cos(a), d * np.sin(a)])
    return (np.array(dist), np.array(idx, dtype=np.int64),
            np.ascontiguousarray(np.array(pts)))


class TestClusterAndFit(TestCase):
    def _run(self, dist, idx, points, radius_scale=1.0, max_residual=np.inf):
        return cluster_and_fit_circles(
            np.ascontiguousarray(dist, dtype=np.float64),
            np.ascontiguousarray(idx, dtype=np.int64), points,
            ANGLE_INCREMENT, LAMBDA, SIGMA, MIN_POINTS, radius_scale,
            MAX_RADIUS, RESIDUAL_THRESHOLD, MAX_TRIALS, STOP_PROBABILITY,
            max_residual)

    def test_empty_scan(self):
        centres, radii = self._run(np.empty(0), np.empty(0, dtype=np.int64),
                                   np.empty((0, 2)))
        self.assertEqual(len(centres), 0)
        self.assertEqual(len(radii), 0)

    def test_two_well_separated_objects_give_two_clusters(self):
        # Two arcs, far apart in both bearing and range
        a = _arc(1.0, 0.0, 0.1, -0.3, 0.3, 8)
        b = _arc(0.0, 2.5, 0.1, 1.3, 1.9, 8)
        points = np.ascontiguousarray(np.vstack((a, b)))
        dist = np.linalg.norm(points, axis=1)
        idx = np.concatenate((np.arange(8), np.arange(100, 108))).astype(np.int64)
        centres, radii = self._run(dist, idx, points)
        self.assertEqual(len(centres), 2)

    def test_a_dropped_return_breaks_a_cluster(self):
        """Non-consecutive scan indices are not neighbours, whatever the ranges."""
        points = np.ascontiguousarray(_arc(1.0, 0.0, 0.1, -0.4, 0.4, 12))
        dist = np.linalg.norm(points, axis=1)
        contiguous = np.arange(12).astype(np.int64)
        split = np.concatenate((np.arange(6), np.arange(200, 206))).astype(np.int64)
        self.assertEqual(len(self._run(dist, contiguous, points)[0]), 1)
        self.assertEqual(len(self._run(dist, split, points)[0]), 2)

    def test_clusters_below_min_points_are_dropped(self):
        points = np.ascontiguousarray(_arc(1.0, 0.0, 0.1, -0.1, 0.1, 2))
        dist = np.linalg.norm(points, axis=1)
        centres, _ = self._run(dist, np.arange(2).astype(np.int64), points)
        self.assertEqual(len(centres), 0)

    def test_radius_scale_is_applied(self):
        points = np.ascontiguousarray(_arc(1.0, 0.0, 0.1, -0.4, 0.4, 10))
        dist = np.linalg.norm(points, axis=1)
        idx = np.arange(10).astype(np.int64)
        _, r1 = self._run(dist, idx, points, radius_scale=1.0)
        _, r2 = self._run(dist, idx, points, radius_scale=1.8)
        np.testing.assert_allclose(r2, r1 * 1.8, rtol=1e-9)

    def test_a_flat_surface_is_rejected_by_the_radius_filter(self):
        """A wall fits an enormous circle, which MAX_RADIUS throws away."""
        n = 12
        y = np.linspace(-0.5, 0.5, n)
        points = np.ascontiguousarray(np.column_stack((np.full(n, 2.0), y)))
        dist = np.linalg.norm(points, axis=1)
        centres, _ = self._run(dist, np.arange(n).astype(np.int64), points)
        self.assertEqual(len(centres), 0)
