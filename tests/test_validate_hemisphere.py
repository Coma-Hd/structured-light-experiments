from __future__ import annotations

import unittest

import numpy as np

from scripts.validate_hemisphere import (
    analyze_common_overlap,
    fit_hemisphere,
    render_markdown,
    summarize_repeats,
)


def synthetic_hemisphere(
    radius: float,
    seed: int,
    count: int = 5000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    cos_polar = rng.uniform(0.12, 1.0, count)
    sin_polar = np.sqrt(1.0 - cos_polar ** 2)
    points = np.column_stack((
        radius * sin_polar * np.cos(azimuth),
        radius * sin_polar * np.sin(azimuth),
        radius * cos_polar,
    ))
    points += np.array([0.18, 0.14, -0.05])
    points += rng.normal(0.0, 0.00015, points.shape)
    points[:50] += rng.normal(0.0, 0.006, (50, 3))
    return points


class HemisphereValidationTests(unittest.TestCase):
    def test_robust_fit_recovers_noisy_hemisphere_diameter(self) -> None:
        points = synthetic_hemisphere(0.04945, seed=1)

        result = fit_hemisphere(points, inlier_threshold_m=0.0025)

        self.assertAlmostEqual(result["diameter_mm"], 98.9, delta=0.15)
        self.assertGreater(result["inlier_ratio"], 0.97)
        self.assertLess(result["residual_rms_mm"], 0.3)

    def test_repeatability_and_markdown_report(self) -> None:
        clouds = [
            synthetic_hemisphere(0.04945, seed=2),
            synthetic_hemisphere(0.04950, seed=3),
            synthetic_hemisphere(0.04940, seed=4),
        ]
        scans = []
        for index, cloud in enumerate(clouds):
            sphere = fit_hemisphere(cloud, 0.0025)
            scans.append({
                "name": f"run{index + 1}",
                "sphere": sphere,
                "diameter_error_mm": sphere["diameter_mm"] - 98.9,
                "diameter_error_percent": (
                    (sphere["diameter_mm"] - 98.9) / 98.9 * 100.0
                ),
                "tracking": {},
            })
        repeat = summarize_repeats(scans, clouds, 98.9, 3000)
        report = {
            "title": "测试报告",
            "generated_at": "2026-07-24T00:00:00+08:00",
            "true_diameter_mm": 98.9,
            "settings": {"inlier_threshold_mm": 2.5},
            "scans": scans,
            "repeatability": repeat,
            "calibration": {},
        }

        markdown = render_markdown(report)

        self.assertLess(repeat["diameter_std_mm"], 0.15)
        self.assertEqual(len(repeat["pairwise"]), 3)
        self.assertIn("完整点云重复性", markdown)
        self.assertIn("不做 ICP", markdown)

    def test_common_overlap_uses_shared_board_region(self) -> None:
        full = synthetic_hemisphere(0.04945, seed=8, count=8000)
        clouds = [
            full[full[:, 0] < np.percentile(full[:, 0], 96)],
            full[full[:, 0] > np.percentile(full[:, 0], 4)],
        ]
        scans = []
        for index, cloud in enumerate(clouds):
            sphere = fit_hemisphere(cloud, 0.0025)
            scans.append({
                "name": f"scan{index + 1}",
                "sphere": sphere,
            })

        common = analyze_common_overlap(
            scans, clouds, 98.9, 0.0025, percentile=2.0,
            maximum_pair_points=3000)

        self.assertEqual(len(common["scans"]), 2)
        self.assertTrue(all(
            scan["retained_point_ratio"] < 1.0
            for scan in common["scans"]
        ))
        self.assertLess(
            common["repeatability"]["diameter_std_mm"], 0.15)


if __name__ == "__main__":
    unittest.main()
