from __future__ import annotations

import unittest

import numpy as np

from src.laser_center import (
    _filter_center_continuity,
    _passes_steger_quality,
    extract_laser_centers,
)


class LaserQualityFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.quality = {
            "enabled": True,
            "width_level": 0.5,
            "min_width_px": 1,
            "max_width_px": 6,
            "saturation_threshold": 250,
            "max_saturated_width_px": 3,
            "max_secondary_peak_ratio": 0.8,
            "secondary_exclusion_px": 5,
        }

    def test_rejects_broad_and_ambiguous_peaks(self) -> None:
        thin = np.zeros(31, dtype=np.float32)
        thin[14:17] = [110, 200, 110]
        intensity = thin.copy()
        self.assertTrue(
            _passes_steger_quality(thin, intensity, 15, 20, self.quality)
        )

        broad = np.zeros(31, dtype=np.float32)
        broad[10:21] = 200
        self.assertFalse(
            _passes_steger_quality(broad, broad, 15, 20, self.quality)
        )

        double = thin.copy()
        double[24] = 180
        self.assertFalse(
            _passes_steger_quality(double, double, 15, 20, self.quality)
        )

    def test_rejects_isolated_center_jump(self) -> None:
        centers = np.array([
            [20.0, 0.0],
            [20.2, 1.0],
            [35.0, 2.0],
            [20.1, 3.0],
            [20.0, 4.0],
        ])
        filtered = _filter_center_continuity(
            centers, "row", window=5, max_deviation=3.0)
        self.assertEqual(len(filtered), 4)
        self.assertFalse(np.any(np.isclose(filtered[:, 0], 35.0)))

    def test_board_mask_is_applied_before_row_argmax(self) -> None:
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[4:36, 19:22, 0] = 130
        image[4:36, 59:62, 0] = 240
        mask = np.zeros((40, 80), dtype=bool)
        mask[:, :40] = True
        cfg = {
            "laser": {
                "method": "steger",
                "score_mode": "blue_minus_green",
                "score_threshold": 20,
                "min_intensity": 1,
                "scan_axis": "row",
                "steger_sigma": 1.0,
                "quality_filter": {"enabled": False},
            }
        }
        centers = extract_laser_centers(
            image, cfg, image_mask_override=mask)
        self.assertGreater(len(centers), 20)
        self.assertAlmostEqual(float(np.median(centers[:, 0])), 20.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
