"""检查手持/机械臂扫描帧的 ChArUco 位姿质量。

不依赖导轨行程或转台角度：对每张图做检测 + PnP + 重投影误差，
并估计相邻有效帧的相机平移，用于判断姿态覆盖是否足够。

用法（在测试包根目录）:
    .\\.venv\\Scripts\\python.exe ceshi/上机械臂/check_poses.py \\
      --config ceshi/上机械臂/arm_scan.yaml \\
      --images ceshi/上机械臂/data/scan \\
      --intrinsic ceshi/上机械臂/calibration/camera_intrinsic.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.charuco import CharucoTarget  # noqa: E402
from src.config import CharucoConfig, load_config, resolve_path  # noqa: E402
from src.geometry import scale_intrinsic  # noqa: E402
from src.io_utils import imread_color, load_intrinsic, load_intrinsic_size  # noqa: E402


_IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def list_images(folder: str):
    files = []
    for ext in _IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(set(files))


def camera_center_in_board(rvec, tvec) -> np.ndarray:
    """相机光心在板坐标系下的位置。"""
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return (-R.T @ t).reshape(3)


def main():
    ap = argparse.ArgumentParser(description="机械臂扫描：逐帧 ChArUco 位姿检查")
    ap.add_argument("--config", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--intrinsic", default=None)
    ap.add_argument("--out", default=None, help="报告 yaml/json 输出路径前缀")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(_HERE, "arm_scan.yaml")
    cfg = load_config(cfg_path)
    image_dir = args.images or resolve_path(cfg, cfg["paths"]["scan_images"])
    intrinsic_path = args.intrinsic or resolve_path(
        cfg, cfg["paths"]["camera_intrinsic"])
    out_prefix = args.out or os.path.join(
        resolve_path(cfg, cfg["paths"]["output"]), "pose_check")

    if not os.path.isfile(intrinsic_path):
        raise FileNotFoundError(
            f"找不到内参: {intrinsic_path}\n"
            "请先在本目录完成相机内参标定。"
        )

    files = list_images(image_dir)
    if not files:
        raise FileNotFoundError(f"扫描目录没有图片: {image_dir}")

    K, dist = load_intrinsic(intrinsic_path)
    calib_size = load_intrinsic_size(intrinsic_path)
    probe = imread_color(files[0])
    if probe is not None and calib_size[0] > 0:
        scan_size = (probe.shape[1], probe.shape[0])
        if scan_size != tuple(calib_size):
            K = scale_intrinsic(K, calib_size, scan_size)
            print(
                f"[警告] 内参分辨率 {calib_size} ≠ 扫描 {scan_size}，已缩放 K。"
            )

    gating = cfg.get("gating", {}) or {}
    min_corners = int(gating.get("min_charuco_corners", 8))
    max_reproj = float(gating.get("max_reproj_error", 2.0))
    target = CharucoTarget(CharucoConfig.from_cfg(cfg))

    rows = []
    centers = []
    ok_count = 0
    for path in files:
        name = os.path.basename(path)
        img = imread_color(path)
        if img is None:
            rows.append({
                "file": name,
                "ok": False,
                "reason": "read_failed",
            })
            continue
        det = target.detect(img)
        if det is None or det.count < min_corners:
            n = 0 if det is None else det.count
            rows.append({
                "file": name,
                "ok": False,
                "reason": "few_corners",
                "corners": int(n),
            })
            continue
        pose = target.estimate_pose(det, K, dist)
        if pose is None:
            rows.append({
                "file": name,
                "ok": False,
                "reason": "pnp_failed",
                "corners": int(det.count),
            })
            continue
        rvec, tvec = pose
        err = float(target.reproj_error(det, rvec, tvec, K, dist))
        c_board = camera_center_in_board(rvec, tvec)
        accepted = err <= max_reproj
        if accepted:
            ok_count += 1
            centers.append(c_board)
        rows.append({
            "file": name,
            "ok": bool(accepted),
            "reason": None if accepted else "reproj_too_large",
            "corners": int(det.count),
            "reproj_error_px": err,
            "camera_center_board_m": c_board.tolist(),
            "tvec_m": tvec.reshape(3).tolist(),
            "rvec": rvec.reshape(3).tolist(),
        })
        tag = "OK" if accepted else "跳过"
        print(
            f"  [{tag}] {name}: corners={det.count}, "
            f"reproj={err:.3f}px, cam_board={c_board}"
        )

    step_mm = []
    for i in range(1, len(centers)):
        step_mm.append(float(np.linalg.norm(centers[i] - centers[i - 1]) * 1000.0))

    span_mm = 0.0
    if len(centers) >= 2:
        arr = np.asarray(centers, dtype=np.float64)
        span_mm = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)) * 1000.0)

    report = {
        "mode": "arm_handheld_charuco_per_frame",
        "image_dir": image_dir,
        "intrinsic": intrinsic_path,
        "total_frames": len(files),
        "accepted_frames": ok_count,
        "rejected_frames": len(files) - ok_count,
        "accept_ratio": float(ok_count / max(len(files), 1)),
        "min_charuco_corners": min_corners,
        "max_reproj_error_px": max_reproj,
        "camera_path_span_mm": span_mm,
        "mean_step_mm": float(np.mean(step_mm)) if step_mm else 0.0,
        "median_step_mm": float(np.median(step_mm)) if step_mm else 0.0,
        "mean_reproj_error_px": float(np.mean([
            r["reproj_error_px"] for r in rows if r.get("ok")
        ])) if ok_count else None,
        "frames": rows,
        "usable": bool(ok_count >= 3 and span_mm >= 15.0),
        "notes": [
            "位姿来自每帧 ChArUco PnP，不需要机械臂编码器。",
            "usable=true 表示至少 3 个有效位姿且相机光心包络跨度 >= 15 mm。",
            "若大量 few_corners：移动时保持板在画面内，避免物体完全挡住板。",
            "若 reproj_too_large：检查内参、板参数、模糊与过曝。",
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    yaml_path = out_prefix + ".yaml"
    json_path = out_prefix + ".json"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(
        f"[位姿检查] 有效 {ok_count}/{len(files)}  "
        f"accept_ratio={report['accept_ratio']:.3f}  "
        f"path_span={span_mm:.1f} mm"
    )
    if report["mean_reproj_error_px"] is not None:
        print(f"[位姿检查] 平均重投影误差 {report['mean_reproj_error_px']:.3f} px")
    print(f"[位姿检查] 已保存: {yaml_path}")
    if not report["usable"]:
        print("[警告] 当前有效位姿覆盖不足，重建点云会很稀疏或失败。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
