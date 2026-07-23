"""相机内参标定（阶段 3）。

流程：遍历图片目录 -> 检测 ChArUco 角点 -> 收集 obj/img 点 ->
cv2.calibrateCamera -> 保存 K, dist，报告重投影误差。
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

import cv2
import numpy as np

from .charuco import CharucoTarget
from .config import CharucoConfig
from .io_utils import save_intrinsic

_IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def list_images(folder: str) -> List[str]:
    files: List[str] = []
    for ext in _IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(set(files))


def calibrate_intrinsic(cfg: Dict, image_dir: str, out_path: str,
                        min_corners: int = 6, verbose: bool = True) -> float:
    """执行内参标定，返回重投影误差 (px)。"""
    target = CharucoTarget(CharucoConfig.from_cfg(cfg))
    files = list_images(image_dir)
    if not files:
        raise FileNotFoundError(f"目录中没有图片: {image_dir}")

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    image_size = None
    used = 0

    for f in files:
        img = cv2.imread(f)
        if img is None:
            if verbose:
                print(f"  跳过(读不了): {os.path.basename(f)}")
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        det = target.detect(img)
        if det is None or det.count < min_corners:
            if verbose:
                n = 0 if det is None else det.count
                print(f"  跳过(角点不足 {n}): {os.path.basename(f)}")
            continue
        obj_points.append(det.obj_points.astype(np.float32))
        img_points.append(det.corners.astype(np.float32).reshape(-1, 1, 2))
        used += 1
        if verbose:
            print(f"  使用({det.count} 角点): {os.path.basename(f)}")

    if used < 3:
        raise RuntimeError(f"有效标定图片过少({used})，至少需要 3 张。")

    flags = 0
    ret, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags)

    save_intrinsic(out_path, K, dist, image_size, ret)
    if verbose:
        print(f"\n标定完成: 使用 {used} 张图, 重投影误差 = {ret:.4f} px")
        print(f"K =\n{K}")
        print(f"dist = {np.asarray(dist).ravel()}")
        print(f"已保存: {out_path}")
        if ret > 1.0:
            print("警告: 重投影误差 > 1px，建议重新采集标定图。")
    return float(ret)
