from __future__ import annotations

import unittest

import numpy as np

from src.calib_laser_plane import (
    _exclude_laser_occluded_corners,
    _fit_frame_laser_line,
    _filter_stripe_centers,
    _frame_consensus_plane,
)
from src.charuco import CharucoDetection


class LaserPlaneFrameConsensusTests(unittest.TestCase):
    def test_filters_weak_and_off_stripe_image_centers(self) -> None:
        stripe = np.column_stack((
            20.0 + np.linspace(0.0, 10.0, 30),
            np.linspace(5.0, 34.0, 30),
        ))
        outliers = np.array([[50.0, 8.0], [60.0, 20.0], [45.0, 32.0]])
        centers = np.vstack((stripe, outliers))
        score = np.zeros((40, 80), dtype=np.float32)
        uv = np.rint(stripe).astype(int)
        score[uv[:, 1], uv[:, 0]] = 100.0
        score[8, 50] = 100.0
        score[20, 60] = 20.0
        score[32, 45] = 100.0

        filtered = _filter_stripe_centers(
            centers, score, min_score=40.0, max_line_dist_px=2.5)

        self.assertEqual(len(filtered), len(stripe))
        self.assertLess(float(np.max(filtered[:, 0])), 31.0)

    def test_excludes_board_corners_covered_by_laser(self) -> None:
        corners = np.array([
            [10.0, 10.0], [20.0, 10.0], [30.0, 10.0],
            [10.0, 30.0], [20.0, 30.0], [30.0, 30.0],
        ])
        detection = CharucoDetection(
            corners=corners,
            ids=np.arange(6, dtype=np.int32).reshape(-1, 1),
            obj_points=np.column_stack((corners, np.zeros(6))),
        )
        centers = np.array([[20.0, 0.0], [20.0, 40.0]])

        filtered = _exclude_laser_occluded_corners(
            detection, centers, exclusion_px=11.0, min_corners=4)

        self.assertEqual(filtered.count, 4)
        self.assertFalse(np.any(np.isclose(filtered.corners[:, 0], 20.0)))

    def test_fits_and_projects_single_frame_laser_line(self) -> None:
        rng = np.random.default_rng(2)
        x = np.linspace(-0.1, 0.1, 80)
        points = np.column_stack((
            x,
            0.02 + rng.normal(0.0, 0.0001, len(x)),
            0.3 + 0.4 * x + rng.normal(0.0, 0.0001, len(x)),
        ))
        points[::10, 1] += 0.008

        projected, inliers, rms = _fit_frame_laser_line(
            points, threshold=0.001)

        self.assertGreater(int(inliers.sum()), 65)
        self.assertLess(len(projected), len(points))
        self.assertLess(rms, 0.0002)

    def test_rejects_entire_inconsistent_pose_lines(self) -> None:
        rng = np.random.default_rng(7)
        frames = []
        for index in range(8):
            x = np.linspace(-0.12, 0.12, 60)
            y = np.full_like(x, -0.08 + index * 0.02)
            z = 0.30 + 0.40 * x + rng.normal(0.0, 0.0001, len(x))
            frames.append(np.column_stack((x, y, z)))
        for index in range(4):
            x = np.linspace(-0.12, 0.12, 60)
            y = np.full_like(x, 0.10 + index * 0.01)
            z = 0.31 + 0.40 * x + index * 0.003
            frames.append(np.column_stack((x, y, z)))

        _, inliers, good_frames, rms = _frame_consensus_plane(
            frames, threshold=0.001, iters=500)

        np.testing.assert_array_equal(
            good_frames,
            np.array([True] * 8 + [False] * 4),
        )
        self.assertGreater(float(np.mean(inliers)), 0.95)
        self.assertLess(rms, 0.0002)


if __name__ == "__main__":
    unittest.main()
