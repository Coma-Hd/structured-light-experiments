"""蓝色线激光中心提取。

提供三种方法：
- centroid: 灰度质心法（默认，快，第一版够用）
- steger:   基于 Hessian 的亚像素脊线提取（高精度）
- components: 蓝紫响应连通线提取，允许同一行/列存在多段激光

支持纯蓝色差和“亮度加权蓝色差”等响应。
返回 (N,2) 的亚像素中心点 (u, v)。
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


def blue_laser_score(image_bgr: np.ndarray,
                     mode: str = "blue_minus_max",
                     blue_gate_threshold: float = 20.0) -> np.ndarray:
    """计算蓝色激光响应图 (float32)。输入 BGR 图。

    blue_minus_max: B - max(R,G)，只保留纯蓝响应，抗白光但会漏掉紫白激光。
    blue_minus_green: B - G，保留蓝色和偏紫激光，适合木材/曲面。
    blue_minus_mean: B - (R+G)/2，两者之间的折中。
    blue_weighted_intensity: mean(B,G,R) * max(B-G,0) / 255。
        仍要求像素含蓝色增量，但会把中心从纯蓝光晕移向高亮白紫核心；
        不会像纯灰度最大值那样把普通白色物体/标定板当成激光。
    blue_gated_intensity: B-G 只形成会饱和的软门控，通过后以
        mean(B,G,R) 定位最亮核心。避免 white/purple core 与纯蓝光晕
        的 B-G 差异在不同材质上造成系统性横向偏移。
    """
    img = image_bgr.astype(np.float32)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    mode = str(mode).lower()
    if mode == "blue_gated_intensity":
        intensity = (b + g + r) / 3.0
        gate_scale = max(float(blue_gate_threshold), 1e-6)
        blue_gate = np.clip(
            np.maximum(b - g, 0.0) / gate_scale,
            0.0,
            1.0,
        )
        score = intensity * blue_gate
    elif mode == "blue_weighted_intensity":
        intensity = (b + g + r) / 3.0
        score = intensity * np.maximum(b - g, 0.0) / 255.0
    elif mode == "blue_minus_green":
        score = b - g
    elif mode == "blue_minus_mean":
        score = b - 0.5 * (r + g)
    else:
        score = b - np.maximum(r, g)
    return np.clip(score, 0, 255)


def _centroid_along_axis(score: np.ndarray, threshold: float,
                         min_intensity: float, axis: str) -> np.ndarray:
    """沿 column(每列一个中心) 或 row(每行一个中心) 求亮度质心。"""
    s = np.where(score >= threshold, score, 0.0)
    pts = []
    if axis == "column":
        h = s.shape[0]
        rows = np.arange(h, dtype=np.float32)
        for u in range(s.shape[1]):
            col = s[:, u]
            tot = col.sum()
            if tot < min_intensity:
                continue
            v = float((rows * col).sum() / tot)
            pts.append((float(u), v))
    else:  # row
        w = s.shape[1]
        cols = np.arange(w, dtype=np.float32)
        for v in range(s.shape[0]):
            row = s[v, :]
            tot = row.sum()
            if tot < min_intensity:
                continue
            u = float((cols * row).sum() / tot)
            pts.append((u, float(v)))
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _steger_centers(score: np.ndarray, sigma: float, threshold: float,
                    axis: str) -> np.ndarray:
    """Steger 脊线法：用 Hessian 的最大特征向量方向做亚像素修正。

    在每列/每行找强响应作为初值，沿 Hessian 主方向做二次泰勒展开定位极值。
    """
    s = cv2.GaussianBlur(score.astype(np.float32), (0, 0), sigma)
    # 一阶、二阶导数
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.Sobel(s, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(s, cv2.CV_32F, 0, 2, ksize=3)
    gxy = cv2.Sobel(s, cv2.CV_32F, 1, 1, ksize=3)

    h, w = s.shape
    pts = []

    def refine(v0: int, u0: int):
        # Hessian 主方向（最大 |特征值| 对应特征向量）
        a = gxx[v0, u0]
        b = gxy[v0, u0]
        c = gyy[v0, u0]
        tr = a + c
        det = a * c - b * b
        disc = max((tr * 0.5) ** 2 - det, 0.0)
        lam = tr * 0.5 - np.sqrt(disc)  # 更负的特征值 -> 脊线法向
        # 特征向量 (nx, ny)
        if abs(b) > 1e-9:
            nx = lam - c
            ny = b
        else:
            nx, ny = (1.0, 0.0) if a <= c else (0.0, 1.0)
        norm = np.hypot(nx, ny)
        if norm < 1e-9:
            return None
        nx, ny = nx / norm, ny / norm
        # 沿法向的一二阶导，求亚像素偏移 t
        num = gx[v0, u0] * nx + gy[v0, u0] * ny
        den = (a * nx * nx + 2 * b * nx * ny + c * ny * ny)
        if abs(den) < 1e-9:
            return None
        t = -num / den
        if abs(t) > 1.0:  # 偏移过大，认为初值不在脊上
            return None
        return (u0 + t * nx, v0 + t * ny)

    if axis == "column":
        for u in range(w):
            col = s[:, u]
            v0 = int(np.argmax(col))
            if col[v0] < threshold:
                continue
            r = refine(v0, u)
            if r is not None:
                pts.append(r)
    else:
        for v in range(h):
            row = s[v, :]
            u0 = int(np.argmax(row))
            if row[u0] < threshold:
                continue
            r = refine(v, u0)
            if r is not None:
                pts.append(r)

    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _component_centers(score: np.ndarray, threshold: float,
                       min_intensity: float, axis: str,
                       min_area: int, min_length: int,
                       close_size: int, keep_largest: int) -> np.ndarray:
    """从完整蓝紫响应图提取连通激光线中心。

    与逐行全局 argmax 不同，本方法先找所有连通线段，再在每个线段内部
    逐行/列计算加权中心。因此标定板和物体在同一行出现时不会互相覆盖。
    """
    mask = (score >= threshold).astype(np.uint8) * 255
    if close_size > 1:
        k = max(1, int(close_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area or max(w, h) < min_length:
            continue
        candidates.append((int(area), label))

    candidates.sort(reverse=True)
    if keep_largest > 0:
        candidates = candidates[:keep_largest]

    pts = []
    for _, label in candidates:
        x, y, w, h, area = stats[label]
        if axis == "column":
            for u in range(x, x + w):
                rows = np.where(labels[y:y + h, u] == label)[0] + y
                if rows.size == 0:
                    continue
                weights = score[rows, u]
                total = float(weights.sum())
                if total < min_intensity:
                    continue
                v = float(np.sum(rows * weights) / total)
                pts.append((float(u), v))
        else:
            for v in range(y, y + h):
                cols = np.where(labels[v, x:x + w] == label)[0] + x
                if cols.size == 0:
                    continue
                weights = score[v, cols]
                total = float(weights.sum())
                if total < min_intensity:
                    continue
                u = float(np.sum(cols * weights) / total)
                pts.append((u, float(v)))

    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _smooth_centers(centers: np.ndarray, axis: str, window: int) -> np.ndarray:
    """Median-smooth laser centers along the stripe to reduce left-right jitter."""
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 5 or window < 3:
        return pts
    w = int(window)
    if w % 2 == 0:
        w += 1
    # For row-scan, points are ordered by v; smooth u. For column-scan, smooth v.
    order_idx = 1 if axis == "row" else 0
    value_idx = 0 if axis == "row" else 1
    order = np.argsort(pts[:, order_idx])
    sorted_pts = pts[order].copy()
    vals = sorted_pts[:, value_idx]
    pad = w // 2
    padded = np.pad(vals, (pad, pad), mode="edge")
    smoothed = np.empty_like(vals)
    for i in range(len(vals)):
        smoothed[i] = np.median(padded[i:i + w])
    sorted_pts[:, value_idx] = smoothed
    return sorted_pts


def extract_laser_centers(
    image_bgr: np.ndarray,
    cfg: Dict,
    image_roi_override: Optional[Dict] = None,
) -> np.ndarray:
    """按配置提取激光中心点，返回 (N,2) 亚像素 (u,v)。

    image_roi_override: 若提供，优先于 cfg['laser']['image_roi']
    （用于关键帧 ROI 按行程插值后的单帧盒）。
    """
    lz = cfg["laser"]
    method = str(lz.get("method", "centroid")).lower()
    axis = str(lz.get("scan_axis", "column")).lower()
    # Accept short aliases used in configs.
    if axis in ("col", "cols", "column", "columns", "x"):
        axis = "column"
    elif axis in ("row", "rows", "y"):
        axis = "row"
    else:
        raise ValueError(f"unsupported laser.scan_axis: {lz.get('scan_axis')}")
    threshold = float(lz.get("score_threshold", 40))
    min_intensity = float(lz.get("min_intensity", 30))
    score_mode = str(lz.get("score_mode", "blue_minus_max"))
    blue_gate_threshold = float(lz.get("blue_gate_threshold", 20.0))

    score = blue_laser_score(
        image_bgr,
        score_mode,
        blue_gate_threshold=blue_gate_threshold,
    )
    blur_sigma = float(lz.get("score_blur_sigma", 0.0))
    if blur_sigma > 0:
        score = cv2.GaussianBlur(score, (0, 0), blur_sigma)

    if image_roi_override is not None:
        image_roi = image_roi_override or {}
    else:
        image_roi = lz.get("image_roi", {}) or {}
    if image_roi.get("enabled", False):
        h, w = score.shape
        normalized = bool(image_roi.get("normalized", True))
        x_min = image_roi.get("x_min", 0.0)
        x_max = image_roi.get("x_max", 1.0 if normalized else w)
        y_min = image_roi.get("y_min", 0.0)
        y_max = image_roi.get("y_max", 1.0 if normalized else h)
        if normalized:
            x_min, x_max = float(x_min) * w, float(x_max) * w
            y_min, y_max = float(y_min) * h, float(y_max) * h
        x0 = max(0, min(w, int(round(float(x_min)))))
        x1 = max(0, min(w, int(round(float(x_max)))))
        y0 = max(0, min(h, int(round(float(y_min)))))
        y1 = max(0, min(h, int(round(float(y_max)))))
        roi_score = np.zeros_like(score)
        if x1 > x0 and y1 > y0:
            roi_score[y0:y1, x0:x1] = score[y0:y1, x0:x1]
        score = roi_score

    if method == "components":
        centers = _component_centers(
            score,
            threshold,
            min_intensity,
            axis,
            int(lz.get("component_min_area", 12)),
            int(lz.get("component_min_length", 8)),
            int(lz.get("morph_close_size", 3)),
            int(lz.get("component_keep_largest", 0)),
        )
    elif method == "steger":
        sigma = float(lz.get("steger_sigma", 2.0))
        centers = _steger_centers(score, sigma, threshold, axis)
    else:
        centers = _centroid_along_axis(score, threshold, min_intensity, axis)

    smooth_window = int(lz.get("smooth_window", 0))
    if smooth_window >= 3 and centers.shape[0] > 0:
        centers = _smooth_centers(centers, axis, smooth_window)
    return centers


def draw_centers(image_bgr: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """把提取的中心点画到图上，便于调试。"""
    vis = image_bgr.copy()
    for u, v in centers:
        cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0, 0, 255), -1)
    return vis
