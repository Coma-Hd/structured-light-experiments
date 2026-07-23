"""用固定 ChArUco 板标定导轨在相机坐标系中的运动轴和位移比例。

采集时标定板固定不动，相机+激光器沿导轨移动并在多个已知位置停稳拍照。
脚本用每张图的 ChArUco 位姿计算相机中心，然后拟合：

    C_board(s_command) = C0 + beta * s_command

其中 normalize(beta) 是板坐标系中的导轨方向，|beta| 是实际位移 /
指令位移的比例。再通过板到相机旋转，将方向转换为扫描重建所需的
相机坐标系 rail.axis。
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calib_intrinsic import list_images  # noqa: E402
from src.charuco import CharucoTarget  # noqa: E402
from src.config import CharucoConfig, load_config  # noqa: E402
from src.io_utils import imread_color, load_intrinsic  # noqa: E402
from src.rail_poses import load_rail_positions  # noqa: E402


@dataclass
class PoseSample:
    image: str
    command_m: float
    rotation_board_to_camera: np.ndarray
    camera_center_board_m: np.ndarray
    reproj_error_px: float
    corners: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用固定 ChArUco 板标定任意三维导轨轴向量和位移比例"
    )
    parser.add_argument("--images", required=True, help="导轨标定图片目录")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--positions",
        help="位置表 CSV，格式 image,distance_mm；优先用于非等距位置",
    )
    source.add_argument(
        "--step-mm",
        type=float,
        help="图片按文件名排序后的等指令步距，例如 20",
    )
    parser.add_argument("--start-mm", type=float, default=0.0,
                        help="配合 --step-mm 使用的首图指令位置")
    parser.add_argument("--config", default="config.yaml", help="ChArUco 配置")
    parser.add_argument(
        "--intrinsic",
        default="output/camera_intrinsic.yaml",
        help="相机内参 YAML",
    )
    parser.add_argument(
        "--out",
        default="output/rail_axis.yaml",
        help="标定结果 YAML",
    )
    parser.add_argument("--min-frames", type=int, default=5,
                        help="最少有效位置数量")
    parser.add_argument("--min-span-mm", type=float, default=60.0,
                        help="最小标定行程")
    parser.add_argument("--max-reproj-px", type=float, default=None,
                        help="单帧最大重投影误差；默认读取配置")
    return parser.parse_args()


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def _command_positions(
    files: Sequence[str],
    positions_path: str | None,
    step_mm: float | None,
    start_mm: float,
) -> dict[str, float]:
    if positions_path:
        return load_rail_positions(positions_path, distance_unit="mm")
    if step_mm is None or step_mm <= 0:
        raise ValueError("--step-mm 必须大于 0")
    return {
        os.path.basename(path): (float(start_mm) + i * float(step_mm)) * 1e-3
        for i, path in enumerate(files)
    }


def _collect_samples(
    cfg: dict,
    files: Sequence[str],
    positions_m: dict[str, float],
    intrinsic_path: str,
    max_reproj_px: float,
) -> tuple[list[PoseSample], list[dict]]:
    K, dist = load_intrinsic(intrinsic_path)
    target = CharucoTarget(CharucoConfig.from_cfg(cfg))
    min_corners = int(cfg.get("gating", {}).get("min_charuco_corners", 6))
    samples: list[PoseSample] = []
    rejected: list[dict] = []

    for path in files:
        name = os.path.basename(path)
        if name not in positions_m:
            rejected.append({"image": name, "reason": "positions 中无对应位置"})
            continue
        image = imread_color(path)
        if image is None:
            rejected.append({"image": name, "reason": "图像读取失败"})
            continue
        det = target.detect(image)
        if det is None or det.count < min_corners:
            rejected.append({"image": name, "reason": "ChArUco 角点不足"})
            continue
        pose = target.estimate_pose(det, K, dist)
        if pose is None:
            rejected.append({"image": name, "reason": "PnP 位姿求解失败"})
            continue
        rvec, tvec = pose
        reproj = target.reproj_error(det, rvec, tvec, K, dist)
        if reproj > max_reproj_px:
            rejected.append({
                "image": name,
                "reason": f"重投影误差 {reproj:.3f}px 超过 {max_reproj_px:.3f}px",
            })
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        center_board = -rotation.T @ translation
        samples.append(PoseSample(
            image=name,
            command_m=float(positions_m[name]),
            rotation_board_to_camera=rotation,
            camera_center_board_m=center_board,
            reproj_error_px=float(reproj),
            corners=int(det.count),
        ))
    return samples, rejected


def fit_rail_motion(samples: Sequence[PoseSample]) -> dict:
    """拟合导轨直线；独立函数便于用合成数据测试。"""
    if len(samples) < 2:
        raise ValueError("至少需要两个有效位置")
    ordered = sorted(samples, key=lambda row: row.command_m)
    command = np.asarray([row.command_m for row in ordered], dtype=np.float64)
    centers = np.vstack([row.camera_center_board_m for row in ordered])
    ds = command - float(command.mean())
    denom = float(np.dot(ds, ds))
    if denom <= 1e-12:
        raise ValueError("所有指令位置相同，无法拟合导轨")

    center_mean = centers.mean(axis=0)
    beta = np.sum(ds[:, None] * (centers - center_mean), axis=0) / denom
    scale = float(np.linalg.norm(beta))
    if scale <= 1e-9:
        raise ValueError("拟合位移接近 0，请检查位置表和 ChArUco 位姿")
    axis_board = beta / scale
    predicted = center_mean[None, :] + ds[:, None] * beta[None, :]
    residuals = centers - predicted
    residual_norm = np.linalg.norm(residuals, axis=1)

    axis_camera_samples = np.vstack([
        row.rotation_board_to_camera @ axis_board for row in ordered
    ])
    axis_camera = axis_camera_samples.mean(axis=0)
    axis_camera /= np.linalg.norm(axis_camera)
    axis_spread_deg = [
        float(np.degrees(np.arccos(np.clip(np.dot(v, axis_camera), -1.0, 1.0))))
        for v in axis_camera_samples
    ]

    reference_rotation = ordered[0].rotation_board_to_camera
    rotation_change_deg = [
        _rotation_angle_deg(row.rotation_board_to_camera @ reference_rotation.T)
        for row in ordered
    ]

    return {
        "ordered": ordered,
        "axis_board": axis_board,
        "axis_camera": axis_camera,
        "position_scale": scale,
        "centers": centers,
        "predicted": predicted,
        "residual_norm": residual_norm,
        "axis_spread_deg": axis_spread_deg,
        "rotation_change_deg": rotation_change_deg,
        "span_m": float(command.max() - command.min()),
    }


def _write_result(
    out_path: str,
    fit: dict,
    rejected: list[dict],
    max_reproj_px: float,
) -> None:
    ordered: list[PoseSample] = fit["ordered"]
    residual_mm = np.asarray(fit["residual_norm"]) * 1000.0
    reproj = np.asarray([row.reproj_error_px for row in ordered])
    result = {
        "axis_camera": np.asarray(fit["axis_camera"]).astype(float).tolist(),
        "axis_board": np.asarray(fit["axis_board"]).astype(float).tolist(),
        "position_scale_actual_over_commanded": float(fit["position_scale"]),
        "recommended_velocity_multiplier": float(fit["position_scale"]),
        "command_span_mm": float(fit["span_m"] * 1000.0),
        "line_fit": {
            "rms_mm": float(np.sqrt(np.mean(residual_mm ** 2))),
            "max_mm": float(residual_mm.max()),
        },
        "camera_rotation_change": {
            "max_deg": float(max(fit["rotation_change_deg"])),
            "mean_deg": float(np.mean(fit["rotation_change_deg"])),
        },
        "axis_camera_spread": {
            "max_deg": float(max(fit["axis_spread_deg"])),
            "mean_deg": float(np.mean(fit["axis_spread_deg"])),
        },
        "reprojection": {
            "gate_px": float(max_reproj_px),
            "mean_px": float(reproj.mean()),
            "max_px": float(reproj.max()),
        },
        "used_frames": len(ordered),
        "rejected_frames": rejected,
        "samples": [
            {
                "image": row.image,
                "command_mm": float(row.command_m * 1000.0),
                "camera_center_board_m": row.camera_center_board_m.astype(float).tolist(),
                "reproj_error_px": float(row.reproj_error_px),
                "corners": int(row.corners),
                "line_residual_mm": float(fit["residual_norm"][index] * 1000.0),
                "rotation_change_deg": float(fit["rotation_change_deg"][index]),
            }
            for index, row in enumerate(ordered)
        ],
        "usage": {
            "rail_axis_yaml": (
                "将 axis_camera 写入 face1_scan.yaml 和 face2_scan.yaml 的 rail.axis"
            ),
            "velocity": (
                "实际速度 = 采集脚本中的名义速度 × recommended_velocity_multiplier"
            ),
            "sign": "最终扫描运动方向必须与本次标定图片位置递增方向相同",
        },
    }
    path = Path(out_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    files = list_images(args.images)
    if not files:
        print(f"[错误] 图片目录为空：{args.images}")
        return 2
    try:
        cfg = load_config(args.config)
        max_reproj_px = (
            float(args.max_reproj_px)
            if args.max_reproj_px is not None
            else float(cfg.get("gating", {}).get("max_reproj_error", 2.0))
        )
        positions_m = _command_positions(
            files, args.positions, args.step_mm, args.start_mm
        )
        samples, rejected = _collect_samples(
            cfg, files, positions_m, args.intrinsic, max_reproj_px
        )
        if len(samples) < int(args.min_frames):
            raise RuntimeError(
                f"有效位置只有 {len(samples)} 个，需要至少 {args.min_frames} 个"
            )
        fit = fit_rail_motion(samples)
        if fit["span_m"] * 1000.0 < float(args.min_span_mm):
            raise RuntimeError(
                f"有效行程只有 {fit['span_m']*1000.0:.1f} mm，"
                f"要求至少 {args.min_span_mm:.1f} mm"
            )
        _write_result(args.out, fit, rejected, max_reproj_px)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[错误] {error}")
        return 2

    axis = np.asarray(fit["axis_camera"])
    residual_mm = np.asarray(fit["residual_norm"]) * 1000.0
    rotation_max = float(max(fit["rotation_change_deg"]))
    spread_max = float(max(fit["axis_spread_deg"]))
    print("")
    print("[导轨轴标定完成]")
    print(
        "  axis_camera = "
        f"[{axis[0]:.9f}, {axis[1]:.9f}, {axis[2]:.9f}]"
    )
    print(
        "  实际/指令位移比例 = "
        f"{fit['position_scale']:.9f}"
    )
    print(
        "  实际速度 = 采集名义速度 × "
        f"{fit['position_scale']:.9f}"
    )
    print(
        f"  直线拟合 RMS = {np.sqrt(np.mean(residual_mm**2)):.3f} mm, "
        f"max = {residual_mm.max():.3f} mm"
    )
    print(f"  相机姿态最大变化 = {rotation_max:.3f}°")
    print(f"  相机系轴方向最大离散 = {spread_max:.3f}°")
    print(f"  使用 {len(fit['ordered'])} 张，拒绝 {len(rejected)} 张")
    print(f"  已保存：{Path(args.out).resolve()}")
    if rotation_max > 1.0:
        print("[警告] 运动中相机姿态变化超过 1°，请检查滑块刚性、振动或板位姿质量。")
    if residual_mm.max() > 2.0:
        print("[警告] 导轨直线残差超过 2 mm，请增加位置数量并检查标定板识别。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
