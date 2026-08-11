"""曲面物体逐帧掩码的保存、加载与跟踪辅助函数。"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def resolve_object_mask_manifest(
    cfg: Dict,
    project_root: Optional[str] = None,
) -> Optional[str]:
    setting = (cfg.get("laser") or {}).get("object_mask") or {}
    if not bool(setting.get("enabled", False)):
        return None
    path = str(setting.get("manifest", "")).strip()
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(
        os.path.join(project_root, path) if project_root else path)


def load_object_mask_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    frames = data.get("frames") or {}
    if not isinstance(frames, dict) or not frames:
        raise ValueError(f"物体掩码清单没有帧: {path}")
    data["path"] = path
    return data


def load_object_mask_for_image(
    manifest: Dict[str, Any],
    image_path: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    name = os.path.basename(image_path)
    record = (manifest.get("frames") or {}).get(name)
    if not isinstance(record, dict):
        raise KeyError(f"物体掩码清单缺少图片: {name}")
    stat = os.stat(image_path)
    expected_size = record.get("image_size_bytes")
    expected_mtime = record.get("image_mtime_ns")
    if (
        expected_size is not None
        and int(expected_size) != int(stat.st_size)
    ):
        raise RuntimeError(
            f"物体掩码属于另一批图片（文件大小变化）: {name}")
    if (
        expected_mtime is not None
        and int(expected_mtime) != int(stat.st_mtime_ns)
    ):
        raise RuntimeError(
            f"物体掩码属于另一批图片（修改时间变化）: {name}")
    mask_path = str(record.get("mask", ""))
    if not os.path.isabs(mask_path):
        mask_path = os.path.join(
            os.path.dirname(manifest["path"]), mask_path)
    data = np.fromfile(mask_path, dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"读不了物体掩码: {mask_path}")
    return mask > 0, record


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(binary)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def cleanup_object_mask(
    mask: np.ndarray,
    morphology_px: int = 7,
    smoothing_px: int = 9,
) -> np.ndarray:
    """删除窄小板面粘连并平滑 GrabCut 锯齿，不做最终安全膨胀。"""
    cleaned = largest_component(mask)
    morphology_px = max(0, int(morphology_px))
    if morphology_px >= 3:
        if morphology_px % 2 == 0:
            morphology_px += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology_px, morphology_px),
        )
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel)
    smoothing_px = max(0, int(smoothing_px))
    if smoothing_px >= 3:
        if smoothing_px % 2 == 0:
            smoothing_px += 1
        cleaned = (
            cv2.medianBlur(cleaned * 255, smoothing_px) > 127
        ).astype(np.uint8)
    return largest_component(cleaned)


def grabcut_from_rectangle(
    image: np.ndarray,
    rect: Tuple[int, int, int, int],
    iterations: int = 5,
    morphology_px: int = 7,
    smoothing_px: int = 9,
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    gc_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image, gc_mask, (int(x0), int(y0), width, height),
        bg_model, fg_model, max(1, int(iterations)),
        cv2.GC_INIT_WITH_RECT,
    )
    foreground = np.isin(
        gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    return cleanup_object_mask(
        foreground, morphology_px, smoothing_px)


def track_and_refine_mask(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_mask: np.ndarray,
    grabcut_iterations: int = 1,
    morphology_px: int = 7,
    smoothing_px: int = 9,
) -> Tuple[np.ndarray, float, int]:
    """光流估计物体仿射运动，再用 GrabCut 在当前帧细化。"""
    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    feature_mask = cv2.dilate(
        (previous_mask > 0).astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    old_points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=4,
        mask=feature_mask * 255,
    )
    if old_points is None or len(old_points) < 6:
        raise RuntimeError("物体掩码内可跟踪特征不足")
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, old_points, None,
        winSize=(21, 21), maxLevel=3)
    if new_points is None or status is None:
        raise RuntimeError("物体光流跟踪失败")
    keep = status.reshape(-1) > 0
    old = old_points.reshape(-1, 2)[keep]
    new = new_points.reshape(-1, 2)[keep]
    if len(old) < 6:
        raise RuntimeError("物体有效光流点不足")
    matrix, inliers = cv2.estimateAffinePartial2D(
        old, new, method=cv2.RANSAC,
        ransacReprojThreshold=2.5, maxIters=1000)
    if matrix is None:
        raise RuntimeError("物体仿射跟踪失败")
    inlier_ratio = (
        float(np.mean(inliers.reshape(-1) > 0))
        if inliers is not None else 0.0
    )
    inlier_count = (
        int(np.count_nonzero(inliers.reshape(-1) > 0))
        if inliers is not None else 0
    )
    height, width = previous_mask.shape
    predicted = cv2.warpAffine(
        (previous_mask > 0).astype(np.uint8),
        matrix, (width, height),
        flags=cv2.INTER_NEAREST,
    )
    predicted = largest_component(predicted)
    if int(predicted.sum()) < 50:
        raise RuntimeError("跟踪后的物体掩码面积过小")

    # 对低纹理物体放在高对比度标定板上的场景，逐帧 GrabCut 容易把被
    # 激光照亮的棋盘区域吸入前景，并在长序列中持续累积。配置为 0 时
    # 只使用光流预测后的轮廓，保持参考帧人工确认过的物体形状。
    if int(grabcut_iterations) <= 0:
        refined = cleanup_object_mask(
            predicted, morphology_px, smoothing_px)
        if int(refined.sum()) < 50:
            raise RuntimeError("清理后的光流物体掩码面积过小")
        return refined, inlier_ratio, inlier_count

    kernel = np.ones((7, 7), dtype=np.uint8)
    eroded = cv2.erode(predicted, kernel, iterations=1)
    dilated = cv2.dilate(predicted, kernel, iterations=2)
    gc_mask = np.full(predicted.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[dilated > 0] = cv2.GC_PR_BGD
    gc_mask[predicted > 0] = cv2.GC_PR_FGD
    gc_mask[eroded > 0] = cv2.GC_FGD
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        current_image, gc_mask, None, bg_model, fg_model,
        max(1, int(grabcut_iterations)), cv2.GC_INIT_WITH_MASK)
    refined = np.isin(
        gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    refined = cleanup_object_mask(
        refined, morphology_px, smoothing_px)
    if int(refined.sum()) < 50:
        raise RuntimeError("GrabCut 细化后的物体掩码面积过小")
    return refined, inlier_ratio, inlier_count


def dilate_mask(mask: np.ndarray, margin_px: int) -> np.ndarray:
    if margin_px <= 0:
        return (mask > 0).astype(np.uint8)
    size = 2 * int(margin_px) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(
        (mask > 0).astype(np.uint8), kernel, iterations=1)
