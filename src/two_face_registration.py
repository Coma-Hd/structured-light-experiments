"""无标定板导轨点云：两相邻侧面刚体配准。

约定：
  - target = 第一次扫描（F1 主面 + F2 条带），定义最终坐标系。
  - source = 第二次扫描（F2 主面 + F1 条带）。
  - 输出 T_source_to_target，使 p_target = T @ p_source_h。

正式算法：
  1) 在两云中分别拟合两个主平面；
  2) 枚举平面对应关系和法向符号，构造完整三维旋转；
  3) 用共享棱锚点确定平移，并按实际重叠率选择初值；
  4) 中/细层 point-to-plane ICP 精化；
  5) 变换完整 source，合并时不做最大簇/最大面提取。

保留约 90 度候选方法用于旧数据兼容，但机械转角不应直接视为点云坐标转角。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:  # pragma: no cover
    _HAS_O3D = False


def _require_o3d() -> None:
    if not _HAS_O3D:
        raise ImportError("两面 ICP 需要 open3d，请先 pip install open3d")


@dataclass(frozen=True)
class ICPLevel:
    voxel: float
    max_correspondence: float
    iterations: int
    method: str = "point_to_plane"


@dataclass
class PlaneFeature:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_count: int

    def to_dict(self) -> Dict:
        return {
            "normal": self.normal.astype(float).tolist(),
            "d": float(self.d),
            "centroid_m": self.centroid.astype(float).tolist(),
            "inlier_count": int(self.inlier_count),
        }


def load_cloud(path: str) -> "o3d.geometry.PointCloud":
    """读取并校验输入点云。单位必须为米。"""
    _require_o3d()
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if len(pts) < 50:
        raise ValueError(f"点云点数过少 ({len(pts)}): {path}")
    if not np.isfinite(pts).all():
        idx = np.where(np.isfinite(pts).all(axis=1))[0]
        pcd = pcd.select_by_index(idx)
        pts = np.asarray(pcd.points)
    span = np.ptp(pts, axis=0)
    if float(span.max()) > 10.0:
        raise ValueError(
            f"点云跨度 {span.max():.3f} m，疑似单位不是米或存在极远飞点: {path}"
        )
    if float(span.max()) < 0.005:
        raise ValueError(f"点云跨度不足 5 mm，无法可靠配准: {path}")
    return pcd


def cloud_stats(pcd: "o3d.geometry.PointCloud") -> Dict:
    pts = np.asarray(pcd.points)
    return {
        "points": int(len(pts)),
        "centroid_m": pts.mean(axis=0).astype(float).tolist(),
        "min_m": pts.min(axis=0).astype(float).tolist(),
        "max_m": pts.max(axis=0).astype(float).tolist(),
        "span_m": np.ptp(pts, axis=0).astype(float).tolist(),
    }


def normalize_axis(axis: Sequence[float]) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("rotation_axis 不能是零向量")
    return a / n


def axis_angle_rotation(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    """Rodrigues：右手系轴角旋转矩阵。"""
    x, y, z = normalize_axis(axis)
    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ], dtype=np.float64)


def centroid_rotation_init(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    axis: Sequence[float],
    angle_deg: float,
) -> np.ndarray:
    """绕原点旋转 source，再通过质心差给出粗平移。"""
    src_c = np.asarray(source.points).mean(axis=0)
    tgt_c = np.asarray(target.points).mean(axis=0)
    R = axis_angle_rotation(axis, angle_deg)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tgt_c - R @ src_c
    return T


def anchor_rotation_init(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    axis: Sequence[float],
    angle_deg: float,
    source_anchors: np.ndarray | None = None,
    target_anchors: np.ndarray | None = None,
) -> np.ndarray:
    """已知约90°旋转后，用对应共享棱锚点估计平移；无锚点则回退质心。"""
    if source_anchors is None or target_anchors is None:
        return centroid_rotation_init(source, target, axis, angle_deg)
    src = np.asarray(source_anchors, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_anchors, dtype=np.float64).reshape(-1, 3)
    if len(src) < 1 or len(src) != len(tgt):
        raise ValueError("source_anchors/target_anchors 数量必须相同且至少 1 个")
    R = axis_angle_rotation(axis, angle_deg)
    translations = tgt - (R @ src.T).T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.median(translations, axis=0)
    return T


def rotation_difference_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """两个旋转矩阵之间的最小三维转角。"""
    delta = np.asarray(R_a).reshape(3, 3).T @ np.asarray(R_b).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _down_and_normals(
    pcd: "o3d.geometry.PointCloud",
    voxel: float,
) -> "o3d.geometry.PointCloud":
    down = pcd.voxel_down_sample(float(voxel)) if voxel > 0 else copy.deepcopy(pcd)
    radius = max(float(voxel) * 3.0, 0.0015)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
    )
    return down


def _registration_estimation(method: str):
    method = str(method).strip().lower()
    if method in ("point_to_point", "p2p"):
        return o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    if method in ("point_to_plane", "p2l", "plane"):
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()
    raise ValueError(f"未知 ICP method: {method}")


def run_icp_level(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    init: np.ndarray,
    level: ICPLevel,
):
    src = _down_and_normals(source, level.voxel)
    tgt = _down_and_normals(target, level.voxel)
    return o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        float(level.max_correspondence),
        np.asarray(init, dtype=np.float64),
        _registration_estimation(level.method),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(level.iterations)
        ),
    )


def candidate_angles(
    angle_min: float,
    angle_max: float,
    angle_step: float,
    expected_sign: float,
    try_both_directions: bool,
) -> List[float]:
    """先尝试期望方向，再可选尝试反方向。"""
    if angle_step <= 0:
        raise ValueError("angle_step 必须 > 0")
    values = np.arange(
        float(angle_min),
        float(angle_max) + float(angle_step) * 0.5,
        float(angle_step),
    )
    signs = [1.0 if expected_sign >= 0 else -1.0]
    if try_both_directions:
        signs.append(-signs[0])
    out: List[float] = []
    for sign in signs:
        for value in values:
            angle = float(sign * value)
            if not any(abs(angle - old) < 1e-9 for old in out):
                out.append(angle)
    return out


def _candidate_score(fitness: float, rmse: float, max_corr: float) -> float:
    """优先重叠率，同时惩罚接近阈值的高残差伪匹配。"""
    penalty = min(float(rmse) / max(float(max_corr), 1e-12), 1.0)
    return float(fitness) - 0.25 * penalty


def extract_dominant_planes(
    pcd: "o3d.geometry.PointCloud",
    count: int = 2,
    distance_threshold: float = 0.0015,
    ransac_iters: int = 4000,
    min_points: int = 500,
) -> List[PlaneFeature]:
    """按点数依次提取主面；不会修改原点云。"""
    _require_o3d()
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(42)
    remaining = copy.deepcopy(pcd)
    planes: List[PlaneFeature] = []
    for _ in range(int(count)):
        if len(remaining.points) < int(min_points):
            break
        model, indices = remaining.segment_plane(
            distance_threshold=float(distance_threshold),
            ransac_n=3,
            num_iterations=int(ransac_iters),
        )
        if len(indices) < int(min_points):
            break
        inliers = remaining.select_by_index(indices)
        points = np.asarray(inliers.points)
        normal = normalize_axis(model[:3])
        planes.append(
            PlaneFeature(
                normal=normal,
                d=float(model[3]) / float(np.linalg.norm(model[:3])),
                centroid=points.mean(axis=0),
                inlier_count=len(indices),
            )
        )
        remaining = remaining.select_by_index(indices, invert=True)
    if len(planes) < int(count):
        raise RuntimeError(
            f"只找到 {len(planes)} 个有效平面，需要 {count} 个。"
            "请保留主面和相邻面条带，或调整 plane_distance_threshold_m。"
        )
    return planes


def _frame_from_two_normals(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    e1 = normalize_axis(first)
    residual = np.asarray(second, dtype=np.float64) - e1 * float(
        np.dot(e1, second)
    )
    if float(np.linalg.norm(residual)) < 0.15:
        raise RuntimeError("两个拟合平面接近平行，无法建立双平面坐标框架")
    e2 = normalize_axis(residual)
    e3 = normalize_axis(np.cross(e1, e2))
    return np.column_stack([e1, e2, e3])


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def build_plane_pair_initializations(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    source_planes: Sequence[PlaneFeature],
    target_planes: Sequence[PlaneFeature],
    source_anchors: np.ndarray | None,
    target_anchors: np.ndarray | None,
    selection_max_correspondence: float = 0.008,
    correspondence: str = "auto",
) -> List[Dict]:
    """枚举主面/邻面对应与法向方向，并按初始重叠率评分。"""
    mode = str(correspondence).strip().lower()
    if mode == "swap":
        assignments = [("swap", (1, 0))]
    elif mode == "direct":
        assignments = [("direct", (0, 1))]
    elif mode == "auto":
        assignments = [("swap", (1, 0)), ("direct", (0, 1))]
    else:
        raise ValueError("plane correspondence 必须为 auto/swap/direct")

    src_anchor_array = (
        None
        if source_anchors is None
        else np.asarray(source_anchors, dtype=np.float64).reshape(-1, 3)
    )
    tgt_anchor_array = (
        None
        if target_anchors is None
        else np.asarray(target_anchors, dtype=np.float64).reshape(-1, 3)
    )
    if (src_anchor_array is None) != (tgt_anchor_array is None):
        raise ValueError("source_anchors 和 target_anchors 必须同时提供")
    if src_anchor_array is not None and len(src_anchor_array) != len(tgt_anchor_array):
        raise ValueError("source_anchors/target_anchors 数量不一致")

    source_frame = _frame_from_two_normals(
        source_planes[0].normal, source_planes[1].normal
    )
    candidates: List[Dict] = []
    for assignment_name, target_indices in assignments:
        target_normals = [
            target_planes[target_indices[0]].normal,
            target_planes[target_indices[1]].normal,
        ]
        for sign0 in (1.0, -1.0):
            for sign1 in (1.0, -1.0):
                target_frame = _frame_from_two_normals(
                    sign0 * target_normals[0],
                    sign1 * target_normals[1],
                )
                rotation = target_frame @ source_frame.T
                if np.linalg.det(rotation) < 0.99:
                    continue

                if src_anchor_array is not None:
                    translations = tgt_anchor_array - (
                        rotation @ src_anchor_array.T
                    ).T
                    translation_source = "anchors"
                else:
                    translations = np.vstack([
                        target_planes[target_indices[i]].centroid
                        - rotation @ source_planes[i].centroid
                        for i in range(2)
                    ])
                    translation_source = "plane_centroids"

                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = rotation
                transform[:3, 3] = np.median(translations, axis=0)
                evaluation = o3d.pipelines.registration.evaluate_registration(
                    source,
                    target,
                    float(selection_max_correspondence),
                    transform,
                )
                score = _candidate_score(
                    evaluation.fitness,
                    evaluation.inlier_rmse,
                    selection_max_correspondence,
                )
                anchor_rmse = None
                if src_anchor_array is not None:
                    moved = (rotation @ src_anchor_array.T).T + transform[:3, 3]
                    anchor_rmse = float(
                        np.sqrt(np.mean(np.sum((moved - tgt_anchor_array) ** 2, axis=1)))
                    )
                candidates.append({
                    "assignment": assignment_name,
                    "target_plane_indices_for_source": list(target_indices),
                    "target_normal_signs": [sign0, sign1],
                    "translation_source": translation_source,
                    "rotation_angle_deg": _rotation_angle_deg(rotation),
                    "fitness": float(evaluation.fitness),
                    "rmse_m": float(evaluation.inlier_rmse),
                    "anchor_rmse_m": anchor_rmse,
                    "score": float(score),
                    "transformation": transform.tolist(),
                })
    return sorted(candidates, key=lambda row: row["score"], reverse=True)


def register_two_faces_plane_pair(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    levels: Sequence[ICPLevel],
    source_anchors: np.ndarray | None = None,
    target_anchors: np.ndarray | None = None,
    plane_distance_threshold: float = 0.0015,
    plane_ransac_iters: int = 4000,
    plane_min_points: int = 500,
    selection_max_correspondence: float = 0.008,
    correspondence: str = "auto",
    min_fitness: float = 0.5,
    max_rmse: float = 0.004,
    max_rotation_change_deg: float = 15.0,
    max_anchor_rmse: float = 0.010,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """双平面自动定向 + 锚点平移 + point-to-plane ICP。"""
    if not levels:
        raise ValueError("双平面初始化后至少需要一个 ICP level")
    source_planes = extract_dominant_planes(
        source,
        distance_threshold=plane_distance_threshold,
        ransac_iters=plane_ransac_iters,
        min_points=plane_min_points,
    )
    target_planes = extract_dominant_planes(
        target,
        distance_threshold=plane_distance_threshold,
        ransac_iters=plane_ransac_iters,
        min_points=plane_min_points,
    )
    candidates = build_plane_pair_initializations(
        source,
        target,
        source_planes,
        target_planes,
        source_anchors,
        target_anchors,
        selection_max_correspondence=selection_max_correspondence,
        correspondence=correspondence,
    )
    if not candidates:
        raise RuntimeError("双平面没有产生有效初值")
    eligible_candidates = candidates
    if source_anchors is not None and target_anchors is not None:
        eligible_candidates = [
            row for row in candidates
            if row["anchor_rmse_m"] is not None
            and float(row["anchor_rmse_m"]) <= float(max_anchor_rmse)
        ]
        if not eligible_candidates:
            best_anchor_rmse = min(
                float(row["anchor_rmse_m"])
                for row in candidates
                if row["anchor_rmse_m"] is not None
            )
            raise RuntimeError(
                "所有双平面候选都不满足共享棱锚点一致性："
                f"最小 anchor RMSE={best_anchor_rmse*1000:.2f} mm，"
                f"门槛={max_anchor_rmse*1000:.2f} mm。"
                "请确认两份点云完整，并按相同顺序重选同一物理棱的顶部/底部。"
            )
    best = eligible_candidates[0]
    current = np.asarray(best["transformation"], dtype=np.float64)
    initial_rotation = current[:3, :3].copy()

    if verbose:
        print(
            "[双平面初值] "
            f"mapping={best['assignment']} "
            f"angle={best['rotation_angle_deg']:.2f} deg "
            f"fitness={best['fitness']:.4f} "
            f"RMSE={best['rmse_m']*1000:.3f} mm"
        )

    level_reports: List[Dict] = []
    for i, level in enumerate(levels):
        result = run_icp_level(source, target, current, level)
        current = np.asarray(result.transformation, dtype=np.float64)
        level_reports.append({
            "level": i,
            "voxel_m": float(level.voxel),
            "max_correspondence_m": float(level.max_correspondence),
            "method": level.method,
            "fitness": float(result.fitness),
            "rmse_m": float(result.inlier_rmse),
        })
        if verbose:
            print(
                f"  refine L{i}: voxel={level.voxel*1000:.2f} mm "
                f"max_corr={level.max_correspondence*1000:.2f} mm "
                f"fitness={result.fitness:.4f} "
                f"RMSE={result.inlier_rmse*1000:.3f} mm"
            )

    final_level = levels[-1]
    evaluation = o3d.pipelines.registration.evaluate_registration(
        _down_and_normals(source, final_level.voxel),
        _down_and_normals(target, final_level.voxel),
        float(final_level.max_correspondence),
        current,
    )
    rotation_change = rotation_difference_deg(initial_rotation, current[:3, :3])
    final_anchor_rmse = None
    if source_anchors is not None and target_anchors is not None:
        src_anchor_array = np.asarray(source_anchors, dtype=np.float64).reshape(-1, 3)
        tgt_anchor_array = np.asarray(target_anchors, dtype=np.float64).reshape(-1, 3)
        moved_anchors = (
            current[:3, :3] @ src_anchor_array.T
        ).T + current[:3, 3]
        final_anchor_rmse = float(
            np.sqrt(np.mean(np.sum((moved_anchors - tgt_anchor_array) ** 2, axis=1)))
        )
    accepted = (
        float(evaluation.fitness) >= float(min_fitness)
        and float(evaluation.inlier_rmse) <= float(max_rmse)
        and rotation_change <= float(max_rotation_change_deg)
        and (
            final_anchor_rmse is None
            or final_anchor_rmse <= float(max_anchor_rmse)
        )
    )
    report = {
        "convention": "p_target = T_source_to_target @ p_source_h",
        "selected_initial_angle_deg": float(best["rotation_angle_deg"]),
        "candidates": candidates,
        "planes": {
            "source": [plane.to_dict() for plane in source_planes],
            "target": [plane.to_dict() for plane in target_planes],
        },
        "levels": level_reports,
        "final": {
            "fitness": float(evaluation.fitness),
            "rmse_m": float(evaluation.inlier_rmse),
            "accepted": bool(accepted),
            "min_fitness": float(min_fitness),
            "max_rmse_m": float(max_rmse),
            "rotation_change_from_selected_init_deg": float(rotation_change),
            "max_rotation_change_deg": float(max_rotation_change_deg),
            "selected_initial_anchor_rmse_m": best["anchor_rmse_m"],
            "final_anchor_rmse_m": final_anchor_rmse,
            "max_anchor_rmse_m": float(max_anchor_rmse),
            "transformation": current.tolist(),
            "rotation_determinant": float(np.linalg.det(current[:3, :3])),
            "translation_m": current[:3, 3].tolist(),
        },
        "initialization": {
            "mode": "plane_pair",
            "selected_assignment": best["assignment"],
            "used_anchors": bool(
                source_anchors is not None and target_anchors is not None
            ),
            "anchor_count": int(
                0
                if source_anchors is None
                else len(np.asarray(source_anchors).reshape(-1, 3))
            ),
        },
    }
    return current, report


def register_two_faces(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    rotation_axis: Sequence[float],
    angles_deg: Iterable[float],
    levels: Sequence[ICPLevel],
    min_fitness: float = 0.15,
    max_rmse: float = 0.004,
    max_rotation_change_deg: float = 25.0,
    source_anchors: np.ndarray | None = None,
    target_anchors: np.ndarray | None = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """返回 source->target 变换和完整报告。"""
    _require_o3d()
    if len(levels) < 1:
        raise ValueError("至少需要一个 ICP level")

    coarse = levels[0]
    candidates: List[Dict] = []
    best_result = None
    best_angle = None
    best_init = None
    best_score = -np.inf

    for angle in angles_deg:
        init = anchor_rotation_init(
            source,
            target,
            rotation_axis,
            float(angle),
            source_anchors=source_anchors,
            target_anchors=target_anchors,
        )
        result = run_icp_level(source, target, init, coarse)
        rotation_change = rotation_difference_deg(
            init[:3, :3], np.asarray(result.transformation)[:3, :3]
        )
        score = _candidate_score(
            result.fitness, result.inlier_rmse, coarse.max_correspondence
        )
        rotation_guard_ok = rotation_change <= float(max_rotation_change_deg)
        if not rotation_guard_ok:
            score = -1e6 - rotation_change
        row = {
            "angle_deg": float(angle),
            "fitness": float(result.fitness),
            "rmse_m": float(result.inlier_rmse),
            "score": float(score),
            "rotation_change_from_init_deg": float(rotation_change),
            "rotation_guard_ok": bool(rotation_guard_ok),
            "initial_transformation": init.tolist(),
            "transformation": np.asarray(result.transformation).tolist(),
        }
        candidates.append(row)
        if verbose:
            print(
                f"  candidate {angle:+.1f} deg: "
                f"fitness={result.fitness:.4f} "
                f"RMSE={result.inlier_rmse*1000:.3f} mm "
                f"dR={rotation_change:.2f} deg "
                f"guard={rotation_guard_ok} score={score:.4f}"
            )
        if rotation_guard_ok and score > best_score:
            best_score = score
            best_result = result
            best_angle = float(angle)
            best_init = init

    if best_result is None:
        raise RuntimeError(
            "所有粗配准都偏离约90°初值过多。正方体可能发生对称伪匹配；"
            "请点选共享棱锚点后重试，或确认旋转轴/方向。"
        )

    current = np.asarray(best_result.transformation, dtype=np.float64)
    level_reports = [{
        "level": 0,
        "voxel_m": float(coarse.voxel),
        "max_correspondence_m": float(coarse.max_correspondence),
        "method": coarse.method,
        "fitness": float(best_result.fitness),
        "rmse_m": float(best_result.inlier_rmse),
    }]
    if verbose:
        print(f"[粗配准最佳] angle={best_angle:+.1f} deg")

    for i, level in enumerate(levels[1:], start=1):
        result = run_icp_level(source, target, current, level)
        current = np.asarray(result.transformation, dtype=np.float64)
        level_reports.append({
            "level": i,
            "voxel_m": float(level.voxel),
            "max_correspondence_m": float(level.max_correspondence),
            "method": level.method,
            "fitness": float(result.fitness),
            "rmse_m": float(result.inlier_rmse),
        })
        if verbose:
            print(
                f"  refine L{i}: voxel={level.voxel*1000:.2f} mm "
                f"max_corr={level.max_correspondence*1000:.2f} mm "
                f"fitness={result.fitness:.4f} "
                f"RMSE={result.inlier_rmse*1000:.3f} mm"
            )

    final_level = levels[-1]
    src_eval = _down_and_normals(source, final_level.voxel)
    tgt_eval = _down_and_normals(target, final_level.voxel)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        src_eval,
        tgt_eval,
        float(final_level.max_correspondence),
        current,
    )
    assert best_init is not None
    final_rotation_change = rotation_difference_deg(
        best_init[:3, :3], current[:3, :3]
    )
    accepted = (
        float(evaluation.fitness) >= float(min_fitness)
        and float(evaluation.inlier_rmse) <= float(max_rmse)
        and final_rotation_change <= float(max_rotation_change_deg)
    )
    report = {
        "convention": "p_target = T_source_to_target @ p_source_h",
        "selected_initial_angle_deg": best_angle,
        "rotation_axis": normalize_axis(rotation_axis).tolist(),
        "candidates": candidates,
        "levels": level_reports,
        "final": {
            "fitness": float(evaluation.fitness),
            "rmse_m": float(evaluation.inlier_rmse),
            "accepted": bool(accepted),
            "min_fitness": float(min_fitness),
            "max_rmse_m": float(max_rmse),
            "rotation_change_from_selected_init_deg": float(final_rotation_change),
            "max_rotation_change_deg": float(max_rotation_change_deg),
            "transformation": current.tolist(),
            "rotation_determinant": float(np.linalg.det(current[:3, :3])),
            "translation_m": current[:3, 3].tolist(),
        },
        "initialization": {
            "used_anchors": bool(
                source_anchors is not None and target_anchors is not None
            ),
            "anchor_count": int(
                0 if source_anchors is None else len(np.asarray(source_anchors).reshape(-1, 3))
            ),
        },
    }
    return current, report


def transformed_copy(
    pcd: "o3d.geometry.PointCloud",
    transformation: np.ndarray,
) -> "o3d.geometry.PointCloud":
    out = copy.deepcopy(pcd)
    out.transform(np.asarray(transformation, dtype=np.float64))
    return out


def merge_clouds(
    target: "o3d.geometry.PointCloud",
    aligned_source: "o3d.geometry.PointCloud",
    voxel: float = 0.0005,
    sor_neighbors: int = 20,
    sor_std_ratio: float = 2.5,
) -> Tuple["o3d.geometry.PointCloud", "o3d.geometry.PointCloud"]:
    """合并完整两云；仅体素+轻度 SOR，不做最大簇。"""
    raw = copy.deepcopy(target) + copy.deepcopy(aligned_source)
    clean = raw.voxel_down_sample(float(voxel)) if voxel > 0 else copy.deepcopy(raw)
    if sor_neighbors > 0 and len(clean.points) > sor_neighbors:
        clean, _ = clean.remove_statistical_outlier(
            nb_neighbors=int(sor_neighbors),
            std_ratio=float(sor_std_ratio),
        )
    if len(clean.points) > 0:
        radius = max(float(voxel) * 4.0, 0.002)
        clean.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
        )
    return raw, clean


def colored_comparison(
    target: "o3d.geometry.PointCloud",
    aligned_source: "o3d.geometry.PointCloud",
) -> "o3d.geometry.PointCloud":
    """target=蓝色，aligned source=橙色。"""
    a = copy.deepcopy(target)
    b = copy.deepcopy(aligned_source)
    a.paint_uniform_color([0.1, 0.45, 1.0])
    b.paint_uniform_color([1.0, 0.45, 0.05])
    return a + b
