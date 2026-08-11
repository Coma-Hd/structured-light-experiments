from __future__ import annotations

import csv
import os
import tempfile
import unittest

import numpy as np

from scripts.calibrate_turntable_axis import (
    PoseSample,
    estimate_axis,
    evaluate_quality,
)
from src.keyframe_roi import (
    load_keyframe_roi_file,
    roi_override_for_angle_deg,
    save_keyframe_roi_file,
)
from src.turntable_poses import (
    load_turntable_angles,
    rotation_matrix,
    transform_cam_by_turntable_rotation,
)


class TurntablePoseTests(unittest.TestCase):
    def test_undoes_rotation_about_offset_axis(self) -> None:
        axis_point = np.array([0.1, -0.2, 0.5])
        axis_direction = np.array([0.0, 0.0, 1.0])
        reference_points = np.array([
            [0.2, -0.2, 0.55],
            [0.1, 0.0, 0.60],
        ])
        rotation = rotation_matrix(axis_direction, 37.0)
        current_points = (
            (reference_points - axis_point) @ rotation.T + axis_point
        )

        recovered = transform_cam_by_turntable_rotation(
            current_points,
            angle_deg=47.0,
            reference_angle_deg=10.0,
            axis_point_m=axis_point,
            axis_direction=axis_direction,
        )

        np.testing.assert_allclose(recovered, reference_points, atol=1e-12)

    def test_clockwise_negative_angles(self) -> None:
        reference = np.array([[1.0, 0.0, 0.0]])
        current = np.array([[0.0, -1.0, 0.0]])
        recovered = transform_cam_by_turntable_rotation(
            current,
            angle_deg=-90.0,
            reference_angle_deg=0.0,
            axis_point_m=[0.0, 0.0, 0.0],
            axis_direction=[0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(recovered, reference, atol=1e-12)

    def test_loads_utf8_angles_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "angles.csv")
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image", "angle_deg", "elapsed_s"])
                writer.writerow(["img_000000.png", "12.5", "0.0"])
            result = load_turntable_angles(path)
            self.assertEqual(result, {"img_000000.png": 12.5})

    def test_estimates_synthetic_axis(self) -> None:
        axis = np.array([0.1, -0.2, 0.97], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        axis_point = np.array([0.03, -0.04, 0.55])
        board_origin = np.array([0.13, 0.02, 0.58])
        samples = []
        for index, angle in enumerate((0.0, 20.0, 45.0, 80.0, 120.0, 160.0)):
            rotation = rotation_matrix(axis, angle)
            origin = (
                axis_point + rotation @ (board_origin - axis_point)
            )
            samples.append(PoseSample(
                path=f"img_{index:06d}.png",
                angle_deg=angle,
                rotation_board_to_camera=rotation,
                origin_camera_m=origin,
                reprojection_error_px=0.1,
                corner_count=30,
            ))

        result = estimate_axis(samples)
        estimated_axis = np.asarray(result["axis_direction"])
        estimated_point = np.asarray(result["axis_point_m"])

        self.assertGreater(float(np.dot(estimated_axis, axis)), 0.999999)
        point_to_true_axis = np.linalg.norm(
            np.cross(estimated_point - axis_point, axis)
        )
        self.assertLess(point_to_true_axis, 1e-9)
        self.assertLess(result["center_model_rms_m"], 1e-9)

        checks = evaluate_quality(
            result,
            min_frames=5,
            min_span_deg=90.0,
            limits={},
        )
        self.assertTrue(all(check["passed"] for check in checks))
        strict_checks = evaluate_quality(
            result,
            min_frames=10,
            min_span_deg=180.0,
            limits={"max_mean_reprojection_error_px": 0.05},
        )
        failed_names = {
            check["name"] for check in strict_checks if not check["passed"]
        }
        self.assertIn("有效位姿数量", failed_names)
        self.assertIn("角度覆盖", failed_names)
        self.assertIn("PnP 平均重投影误差", failed_names)

    def test_angle_keyframe_roi_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "roi.json")
            save_keyframe_roi_file(path, [
                {
                    "image": "img_000000.png",
                    "angle_deg": 0.0,
                    "roi": {
                        "x_min": 0.1, "x_max": 0.3,
                        "y_min": 0.2, "y_max": 0.4,
                    },
                },
                {
                    "image": "img_000001.png",
                    "angle_deg": 90.0,
                    "roi": {
                        "x_min": 0.3, "x_max": 0.5,
                        "y_min": 0.4, "y_max": 0.6,
                    },
                },
            ])
            loaded = load_keyframe_roi_file(path)
            roi = roi_override_for_angle_deg(loaded, 45.0)

            self.assertEqual(loaded["parameter_key"], "angle_deg")
            self.assertAlmostEqual(roi["x_min"], 0.2)
            self.assertAlmostEqual(roi["x_max"], 0.4)
            self.assertAlmostEqual(roi["y_min"], 0.3)
            self.assertAlmostEqual(roi["y_max"], 0.5)


if __name__ == "__main__":
    unittest.main()
