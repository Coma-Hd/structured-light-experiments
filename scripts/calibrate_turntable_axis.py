"""由旋转中的 ChArUco 板和已知角度标定转台轴在相机坐标系中的位置。"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calib_intrinsic import list_images  # noqa: E402
from src.charuco import CharucoTarget  # noqa: E402
from src.config import CharucoConfig, load_config, resolve_path  # noqa: E402
from src.io_utils import imread_color, load_intrinsic  # noqa: E402
from src.turntable_poses import (  # noqa: E402
    load_turntable_angles,
    lookup_angle,
    resolve_angles_path,
    rotation_matrix,
)


@dataclass
class PoseSample:
    path: str
    angle_deg: float
    rotation_board_to_camera: np.ndarray
    origin_camera_m: np.ndarray
    reprojection_error_px: float
    corner_count: int


DEFAULT_QUALITY_LIMITS = {
    "max_center_rms_mm": 1.0,
    "max_center_max_mm": 2.0,
    "max_rotation_rms_deg": 1.0,
    "max_rotation_max_deg": 2.0,
    "max_mean_reprojection_error_px": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用旋转 ChArUco 板标定转台旋转轴"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--angles", default="angles.csv")
    parser.add_argument("--intrinsic", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--min-span-deg", type=float, default=90.0)
    return parser.parse_args()


def wrapped_delta_deg(angle: float, reference: float) -> float:
    return (float(angle) - float(reference) + 180.0) % 360.0 - 180.0


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def collect_samples(
    files: list[str],
    angles: dict[str, float],
    target: CharucoTarget,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    min_corners: int,
    max_reprojection_error: float,
) -> list[PoseSample]:
    samples: list[PoseSample] = []
    for path in files:
        angle = lookup_angle(angles, path)
        if angle is None:
            continue
        image = imread_color(path)
        if image is None:
            continue
        detection = target.detect(image)
        if detection is None or detection.count < min_corners:
            continue
        pose = target.estimate_pose(detection, camera_matrix, distortion)
        if pose is None:
            continue
        rvec, tvec = pose
        reprojection_error = target.reproj_error(
            detection, rvec, tvec, camera_matrix, distortion
        )
        if reprojection_error > max_reprojection_error:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        samples.append(PoseSample(
            path=path,
            angle_deg=float(angle),
            rotation_board_to_camera=rotation,
            origin_camera_m=np.asarray(tvec, dtype=np.float64).reshape(3),
            reprojection_error_px=float(reprojection_error),
            corner_count=int(detection.count),
        ))
    return samples


def estimate_axis(samples: list[PoseSample]) -> dict:
    reference = samples[0]
    axis_candidates = []
    for sample in samples[1:]:
        delta = wrapped_delta_deg(sample.angle_deg, reference.angle_deg)
        if abs(delta) < 5.0 or abs(delta) > 175.0:
            continue
        observed_relative = (
            sample.rotation_board_to_camera
            @ reference.rotation_board_to_camera.T
        )
        rotation_vector, _ = cv2.Rodrigues(observed_relative)
        vector = rotation_vector.reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            continue
        candidate = vector / norm
        if delta < 0:
            candidate = -candidate
        axis_candidates.append(candidate)
    if len(axis_candidates) < 3:
        raise RuntimeError(
            "可用于估计轴方向的姿态不足；需要覆盖 5°～175° 的多个角度"
        )

    axis = np.sum(np.asarray(axis_candidates), axis=0)
    axis /= np.linalg.norm(axis)
    aligned_candidates = np.asarray([
        candidate if np.dot(candidate, axis) >= 0 else -candidate
        for candidate in axis_candidates
    ])
    axis = aligned_candidates.mean(axis=0)
    axis /= np.linalg.norm(axis)

    matrices = []
    targets = []
    reference_origin = reference.origin_camera_m
    for sample in samples[1:]:
        delta = wrapped_delta_deg(sample.angle_deg, reference.angle_deg)
        if abs(delta) < 1e-6:
            continue
        expected_rotation = rotation_matrix(axis, delta)
        matrices.append(np.eye(3) - expected_rotation)
        targets.append(
            sample.origin_camera_m - expected_rotation @ reference_origin
        )
    # 转轴上的点沿轴方向不唯一；附加 n^T C=0，选择距离相机光心最近的轴点。
    matrices.append(axis.reshape(1, 3))
    targets.append(np.zeros(1, dtype=np.float64))
    design = np.vstack(matrices)
    observation = np.concatenate(targets)
    axis_point, _, _, _ = np.linalg.lstsq(design, observation, rcond=None)

    center_residuals = []
    rotation_residuals = []
    for sample in samples:
        delta = wrapped_delta_deg(sample.angle_deg, reference.angle_deg)
        expected_rotation = rotation_matrix(axis, delta)
        predicted_origin = (
            axis_point
            + expected_rotation @ (reference_origin - axis_point)
        )
        center_residuals.append(
            float(np.linalg.norm(sample.origin_camera_m - predicted_origin))
        )
        observed_relative = (
            sample.rotation_board_to_camera
            @ reference.rotation_board_to_camera.T
        )
        rotation_residuals.append(
            rotation_error_deg(observed_relative, expected_rotation)
        )

    center_residuals_array = np.asarray(center_residuals)
    rotation_residuals_array = np.asarray(rotation_residuals)
    radius_m = float(np.linalg.norm(
        reference_origin
        - axis_point
        - axis * np.dot(axis, reference_origin - axis_point)
    ))
    return {
        "axis_point_m": axis_point.tolist(),
        "axis_direction": axis.tolist(),
        "positive_angle_rule": (
            "angles.csv 中角度增加时，按 axis_direction 右手定则旋转"
        ),
        "reference_image": os.path.basename(reference.path),
        "reference_angle_deg": float(reference.angle_deg),
        "board_origin_radius_m": radius_m,
        "valid_pose_frames": len(samples),
        "angle_span_deg": float(
            max(sample.angle_deg for sample in samples)
            - min(sample.angle_deg for sample in samples)
        ),
        "center_model_rms_m": float(np.sqrt(np.mean(center_residuals_array ** 2))),
        "center_model_max_m": float(center_residuals_array.max()),
        "rotation_model_rms_deg": float(
            np.sqrt(np.mean(rotation_residuals_array ** 2))
        ),
        "rotation_model_max_deg": float(rotation_residuals_array.max()),
        "mean_reprojection_error_px": float(np.mean([
            sample.reprojection_error_px for sample in samples
        ])),
        "mean_corner_count": float(np.mean([
            sample.corner_count for sample in samples
        ])),
        "frame": "camera",
    }


def evaluate_quality(
    result: dict,
    *,
    min_frames: int,
    min_span_deg: float,
    limits: dict,
) -> list[dict]:
    """Return transparent per-metric pass/fail checks."""
    merged = {**DEFAULT_QUALITY_LIMITS, **(limits or {})}
    checks = [
        {
            "name": "有效位姿数量",
            "value": int(result["valid_pose_frames"]),
            "limit": int(min_frames),
            "operator": ">=",
            "unit": "帧",
            "passed": int(result["valid_pose_frames"]) >= int(min_frames),
        },
        {
            "name": "角度覆盖",
            "value": float(result["angle_span_deg"]),
            "limit": float(min_span_deg),
            "operator": ">=",
            "unit": "°",
            "passed": float(result["angle_span_deg"]) >= float(min_span_deg),
        },
        {
            "name": "中心模型 RMS",
            "value": float(result["center_model_rms_m"]) * 1000.0,
            "limit": float(merged["max_center_rms_mm"]),
            "operator": "<=",
            "unit": "mm",
            "passed": (
                float(result["center_model_rms_m"]) * 1000.0
                <= float(merged["max_center_rms_mm"])
            ),
        },
        {
            "name": "中心模型最大值",
            "value": float(result["center_model_max_m"]) * 1000.0,
            "limit": float(merged["max_center_max_mm"]),
            "operator": "<=",
            "unit": "mm",
            "passed": (
                float(result["center_model_max_m"]) * 1000.0
                <= float(merged["max_center_max_mm"])
            ),
        },
        {
            "name": "旋转模型 RMS",
            "value": float(result["rotation_model_rms_deg"]),
            "limit": float(merged["max_rotation_rms_deg"]),
            "operator": "<=",
            "unit": "°",
            "passed": (
                float(result["rotation_model_rms_deg"])
                <= float(merged["max_rotation_rms_deg"])
            ),
        },
        {
            "name": "旋转模型最大值",
            "value": float(result["rotation_model_max_deg"]),
            "limit": float(merged["max_rotation_max_deg"]),
            "operator": "<=",
            "unit": "°",
            "passed": (
                float(result["rotation_model_max_deg"])
                <= float(merged["max_rotation_max_deg"])
            ),
        },
        {
            "name": "PnP 平均重投影误差",
            "value": float(result["mean_reprojection_error_px"]),
            "limit": float(merged["max_mean_reprojection_error_px"]),
            "operator": "<=",
            "unit": "px",
            "passed": (
                float(result["mean_reprojection_error_px"])
                <= float(merged["max_mean_reprojection_error_px"])
            ),
        },
    ]
    return checks


def print_quality_report(checks: list[dict]) -> bool:
    print("")
    print("[质量判定] 转台轴标定")
    for check in checks:
        status = "合格" if check["passed"] else "不合格"
        value = check["value"]
        value_text = (
            str(value) if isinstance(value, int) else f"{float(value):.3f}"
        )
        print(
            f"[质量判定] {check['name']}：{value_text} {check['unit']} "
            f"（要求 {check['operator']} {check['limit']:g} "
            f"{check['unit']}）— {status}"
        )
    passed = all(check["passed"] for check in checks)
    print(f"[质量结论] {'合格，可以继续扫描' if passed else '不合格，请先排查并重新标定'}")
    return passed


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    intrinsic_path = (
        args.intrinsic
        or resolve_path(cfg, cfg["paths"]["camera_intrinsic"])
    )
    camera_matrix, distortion = load_intrinsic(intrinsic_path)
    files = list_images(args.images)
    if not files:
        raise FileNotFoundError(f"目录中没有标定图片：{args.images}")
    angles_path = resolve_angles_path(args.images, args.angles)
    angles = load_turntable_angles(angles_path)
    target = CharucoTarget(CharucoConfig.from_cfg(cfg))
    gating = cfg.get("gating", {}) or {}
    samples = collect_samples(
        files,
        angles,
        target,
        camera_matrix,
        distortion,
        min_corners=int(gating.get("min_charuco_corners", 6)),
        max_reprojection_error=float(gating.get("max_reproj_error", 2.0)),
    )
    span = (
        max(sample.angle_deg for sample in samples)
        - min(sample.angle_deg for sample in samples)
        if samples
        else 0.0
    )
    try:
        result = estimate_axis(samples)
    except (IndexError, RuntimeError, ValueError) as error:
        # Even when geometry is insufficient for an axis fit, preserve and
        # print the measurable diagnostics instead of failing without a result.
        result = {
            "valid_pose_frames": len(samples),
            "angle_span_deg": float(span),
            "quality_passed": False,
            "calibration_error": str(error),
            "frame": "camera",
        }
        output = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                result, handle, allow_unicode=True, sort_keys=False
            )
        print(f"[转轴标定] 有效位姿：{len(samples)}")
        print(f"[转轴标定] 角度覆盖：{span:.3f}°")
        print(f"[转轴标定] 无法计算转轴模型：{error}")
        print(f"[转轴标定] 诊断结果已保存：{output}")
        print("[质量结论] 不合格，数据不足以计算完整转轴")
        return 3

    quality_checks = evaluate_quality(
        result,
        min_frames=args.min_frames,
        min_span_deg=args.min_span_deg,
        limits=cfg.get("turntable_axis_quality", {}) or {},
    )
    quality_passed = all(check["passed"] for check in quality_checks)
    result["quality_passed"] = quality_passed
    result["quality_checks"] = quality_checks
    output = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            result, handle, allow_unicode=True, sort_keys=False
        )

    print(f"[转轴标定] 有效位姿：{result['valid_pose_frames']}")
    print(f"[转轴标定] 角度覆盖：{result['angle_span_deg']:.3f}°")
    print(f"[转轴标定] axis_point_m={result['axis_point_m']}")
    print(f"[转轴标定] axis_direction={result['axis_direction']}")
    print(
        "[转轴标定] 中心模型 RMS="
        f"{result['center_model_rms_m'] * 1000.0:.3f} mm，"
        "最大值="
        f"{result['center_model_max_m'] * 1000.0:.3f} mm；"
        "旋转模型 RMS="
        f"{result['rotation_model_rms_deg']:.3f}°，"
        "最大值="
        f"{result['rotation_model_max_deg']:.3f}°"
    )
    print(
        "[转轴标定] PnP 平均重投影误差="
        f"{result['mean_reprojection_error_px']:.3f} px"
    )
    print(f"[转轴标定] 已保存：{output}")
    print_quality_report(quality_checks)
    return 0 if quality_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
