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
                     blue_gate_threshold: float = 20.0,
                     blue_gate_expand_px: int = 0) -> np.ndarray:
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
    blue_guided_intensity: 把蓝色门控向邻域扩展后再乘亮度，使饱和白色
        激光亮核继承两侧蓝色光晕的门控，适合白芯明显的新相机图像。
    """
    img = image_bgr.astype(np.float32)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    mode = str(mode).lower()
    if mode in ("blue_gated_intensity", "blue_guided_intensity"):
        intensity = (b + g + r) / 3.0
        gate_scale = max(float(blue_gate_threshold), 1e-6)
        blue_gate = np.clip(
            np.maximum(b - g, 0.0) / gate_scale,
            0.0,
            1.0,
        )
        if mode == "blue_guided_intensity":
            radius = max(0, int(blue_gate_expand_px))
            if radius > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * radius + 1, 2 * radius + 1),
                )
                blue_gate = cv2.dilate(blue_gate, kernel)
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


def _contiguous_width(profile: np.ndarray, index: int,
                      level: float, floor: float) -> int:
    """Return the contiguous peak width at a relative response level."""
    peak = float(profile[index])
    cut = max(float(floor), peak * float(level))
    left = int(index)
    right = int(index)
    while left > 0 and profile[left - 1] >= cut:
        left -= 1
    while right + 1 < len(profile) and profile[right + 1] >= cut:
        right += 1
    return right - left + 1


def _passes_steger_quality(
    profile: np.ndarray,
    intensity_profile: np.ndarray,
    index: int,
    threshold: float,
    quality: Dict,
) -> bool:
    """Reject broad, saturated, or ambiguous stripe peaks."""
    if not bool(quality.get("enabled", False)):
        return True

    width_level = float(quality.get("width_level", 0.5))
    width = _contiguous_width(
        profile, index, width_level, threshold)
    min_width = float(quality.get("min_width_px", 0.0))
    max_width = float(quality.get("max_width_px", 0.0))
    if min_width > 0 and width < min_width:
        return False
    if max_width > 0 and width > max_width:
        return False

    saturation_threshold = float(
        quality.get("saturation_threshold", 250.0))
    max_saturated_width = int(
        quality.get("max_saturated_width_px", 0))
    if max_saturated_width > 0:
        saturated = intensity_profile >= saturation_threshold
        saturated_width = 0
        if saturated[index]:
            left = int(index)
            right = int(index)
            while left > 0 and saturated[left - 1]:
                left -= 1
            while right + 1 < len(saturated) and saturated[right + 1]:
                right += 1
            saturated_width = right - left + 1
        if saturated_width > max_saturated_width:
            return False

    max_secondary_ratio = float(
        quality.get("max_secondary_peak_ratio", 0.0))
    if 0 < max_secondary_ratio < 1.0:
        exclusion = max(
            int(quality.get("secondary_exclusion_px", 8)),
            width // 2 + 2,
        )
        left = max(0, int(index) - exclusion)
        right = min(len(profile), int(index) + exclusion + 1)
        secondary = 0.0
        if left > 0:
            secondary = max(secondary, float(np.max(profile[:left])))
        if right < len(profile):
            secondary = max(secondary, float(np.max(profile[right:])))
        if secondary > float(profile[index]) * max_secondary_ratio:
            return False
    return True


def _filter_center_continuity(
    centers: np.ndarray,
    axis: str,
    window: int,
    max_deviation: float,
) -> np.ndarray:
    """Reject isolated center jumps without moving accepted centers."""
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3 or window < 3 or max_deviation <= 0:
        return pts
    if window % 2 == 0:
        window += 1
    order_idx = 1 if axis == "row" else 0
    value_idx = 0 if axis == "row" else 1
    pts = pts[np.argsort(pts[:, order_idx])]
    keep = np.ones(len(pts), dtype=bool)
    half = window // 2

    # Process only consecutive scan-line groups; do not bridge ROI gaps.
    starts = [0]
    gaps = np.where(np.diff(pts[:, order_idx]) > 1.5)[0]
    starts.extend((gaps + 1).tolist())
    ends = (gaps + 1).tolist() + [len(pts)]
    for start, end in zip(starts, ends):
        if end - start < 3:
            continue
        values = pts[start:end, value_idx]
        for local_index in range(end - start):
            lo = max(0, local_index - half)
            hi = min(end - start, local_index + half + 1)
            median = float(np.median(values[lo:hi]))
            if abs(float(values[local_index]) - median) > max_deviation:
                keep[start + local_index] = False
    return pts[keep]


def _steger_centers(score: np.ndarray, sigma: float, threshold: float,
                    axis: str, intensity: Optional[np.ndarray] = None,
                    quality: Optional[Dict] = None) -> np.ndarray:
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
    intensity = (
        np.asarray(intensity, dtype=np.float32)
        if intensity is not None else np.asarray(score, dtype=np.float32)
    )
    quality = quality or {}
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
            if not _passes_steger_quality(
                    col, intensity[:, u], v0, threshold, quality):
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
            if not _passes_steger_quality(
                    row, intensity[v, :], u0, threshold, quality):
                continue
            r = refine(v, u0)
            if r is not None:
                pts.append(r)

    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    centers = np.asarray(pts, dtype=np.float64)
    return _filter_center_continuity(
        centers,
        axis,
        int(quality.get("continuity_window", 0)),
        float(quality.get("max_continuity_deviation_px", 0.0)),
    )


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


def _filter_dominant_center_track(
    centers: np.ndarray,
    axis: str,
    setting: Dict,
) -> np.ndarray:
    """保留主激光平滑轨迹，删除掩码内短小错误响应分支。"""
    points = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if not bool(setting.get("enabled", False)) or len(points) < 6:
        return points
    order_index = 1 if axis == "row" else 0
    value_index = 0 if axis == "row" else 1
    points = points[np.argsort(points[:, order_index])]
    max_order_gap = float(setting.get("max_order_gap_px", 3.0))
    max_jump = float(setting.get("max_center_jump_px", 8.0))
    min_anchor = int(setting.get("min_anchor_points", 10))
    max_residual = float(setting.get("max_residual_px", 4.0))
    min_secondary_points = int(
        setting.get("min_secondary_segment_points", 0))
    max_secondary_offset = float(
        setting.get("max_secondary_segment_offset_px", 0.0))
    polynomial_order = max(
        1, min(int(setting.get("polynomial_order", 2)), 3))

    segments = []
    start = 0
    for index in range(1, len(points)):
        order_gap = (
            points[index, order_index]
            - points[index - 1, order_index]
        )
        value_jump = abs(
            points[index, value_index]
            - points[index - 1, value_index]
        )
        if order_gap > max_order_gap or value_jump > max_jump:
            segments.append(points[start:index])
            start = index
    segments.append(points[start:])
    anchor = max(segments, key=len)
    if len(anchor) < max(min_anchor, polynomial_order + 2):
        return points

    anchor_order = anchor[:, order_index]
    anchor_value = anchor[:, value_index]
    coefficients = np.polyfit(
        anchor_order, anchor_value, polynomial_order)
    residual = (
        points[:, value_index]
        - np.polyval(coefficients, points[:, order_index])
    )
    keep = np.abs(residual) <= max_residual
    # 用首轮内点再拟合，避免最长段局部噪声轻微拉偏整条轨迹。
    if int(keep.sum()) >= max(min_anchor, polynomial_order + 2):
        coefficients = np.polyfit(
            points[keep, order_index],
            points[keep, value_index],
            polynomial_order,
        )
        residual = (
            points[:, value_index]
            - np.polyval(coefficients, points[:, order_index])
        )
        keep = np.abs(residual) <= max_residual
    # 曲面反光可能把一条真实激光分成数个间断段。除主拟合轨迹外，保留
    # 长度足够且横向位置接近主分支的片段；短小、远离主线的分支仍删除。
    if min_secondary_points > 0 and max_secondary_offset > 0:
        anchor_median = float(np.median(anchor[:, value_index]))
        for segment in segments:
            if len(segment) < min_secondary_points:
                continue
            segment_median = float(np.median(segment[:, value_index]))
            if abs(segment_median - anchor_median) > max_secondary_offset:
                continue
            for point in segment:
                match = np.all(np.isclose(points, point), axis=1)
                keep[match] = True
    return points[keep]


def _roi_has_laser_line(
    score: np.ndarray,
    axis: str,
    min_peak_score: float,
    min_hit_lines: int,
) -> bool:
    """判断 ROI 掩码后的 score 中是否已经出现足够长的激光线。

    激光尚未扫入 ROI 时，框内常有弱蓝噪声；逐行/列 argmax 仍会出红点。
    要求至少 ``min_hit_lines`` 条扫描线的峰值达到 ``min_peak_score``。
    """
    if score.size == 0:
        return False
    min_hit_lines = max(1, int(min_hit_lines))
    min_peak_score = float(min_peak_score)
    if axis == "column":
        peaks = score.max(axis=0)
    else:
        peaks = score.max(axis=1)
    return int(np.count_nonzero(peaks >= min_peak_score)) >= min_hit_lines


def extract_laser_centers(
    image_bgr: np.ndarray,
    cfg: Dict,
    image_roi_override: Optional[Dict] = None,
    image_mask_override: Optional[np.ndarray] = None,
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
    blue_gate_expand_px = int(lz.get("blue_gate_expand_px", 0))

    score = blue_laser_score(
        image_bgr,
        score_mode,
        blue_gate_threshold=blue_gate_threshold,
        blue_gate_expand_px=blue_gate_expand_px,
    )
    intensity = np.max(image_bgr, axis=2).astype(np.float32)
    blur_sigma = float(lz.get("score_blur_sigma", 0.0))
    if blur_sigma > 0:
        score = cv2.GaussianBlur(score, (0, 0), blur_sigma)

    roi_active = False
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
        roi_intensity = np.zeros_like(intensity)
        if x1 > x0 and y1 > y0:
            roi_score[y0:y1, x0:x1] = score[y0:y1, x0:x1]
            roi_intensity[y0:y1, x0:x1] = intensity[y0:y1, x0:x1]
            roi_active = True
        score = roi_score
        intensity = roi_intensity
    if image_mask_override is not None:
        mask = np.asarray(image_mask_override, dtype=bool)
        if mask.shape != score.shape:
            raise ValueError(
                "image_mask_override shape must match the image")
        score = np.where(mask, score, 0.0)
        intensity = np.where(mask, intensity, 0.0)
        roi_active = True

    # 可选门控：ROI/掩码内尚未出现足够长激光线时不抓中心，避免框内噪声出红点。
    gate = lz.get("roi_laser_gate", {}) or {}
    if bool(gate.get("enabled", False)) and roi_active:
        min_peak = float(gate.get("min_peak_score", max(threshold, 40.0)))
        min_hits = int(gate.get("min_hit_lines", 12))
        if not _roi_has_laser_line(score, axis, min_peak, min_hits):
            return np.empty((0, 2), dtype=np.float64)

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
        centers = _steger_centers(
            score,
            sigma,
            threshold,
            axis,
            intensity=intensity,
            quality=lz.get("quality_filter", {}) or {},
        )
    else:
        centers = _centroid_along_axis(score, threshold, min_intensity, axis)

    smooth_window = int(lz.get("smooth_window", 0))
    if smooth_window >= 3 and centers.shape[0] > 0:
        centers = _smooth_centers(centers, axis, smooth_window)
    centers = _filter_dominant_center_track(
        centers, axis, lz.get("dominant_track_filter", {}) or {})
    return centers


def draw_centers(image_bgr: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """把提取的中心点画到图上，便于调试。"""
    vis = image_bgr.copy()
    for u, v in centers:
        cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0, 0, 255), -1)
    return vis
