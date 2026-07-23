"""Four-face diagnostics, constrained registration, and cuboid geometry helpers.

All distances are metres.  The input clouds are assumed to already share the
ChArUco board coordinate frame; registration only estimates small SE(3)
corrections and never estimates scale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


FACES = ("face1", "face2", "face3", "face4")
PAIRS = (("face1", "face2"), ("face2", "face3"),
         ("face3", "face4"), ("face4", "face1"))
COLORS = {
    "face1": (0.95, 0.20, 0.20),
    "face2": (0.20, 0.80, 0.25),
    "face3": (0.20, 0.40, 0.95),
    "face4": (0.95, 0.75, 0.15),
}


def fit_sampled_rail_trajectory(
    distances_m: np.ndarray,
    rotations_board_to_camera: np.ndarray,
    centers_board: np.ndarray,
    min_samples: int = 3,
    max_center_residual_m: float = 0.004,
    max_rotation_deviation_deg: float = 1.5,
) -> dict[str, Any]:
    """Fit the same fixed-rotation/linear-center model as charuco_tracking."""
    distances = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    rotations = np.asarray(
        rotations_board_to_camera, dtype=np.float64).reshape(-1, 3, 3)
    centers = np.asarray(centers_board, dtype=np.float64).reshape(-1, 3)
    if not (len(distances) == len(rotations) == len(centers)):
        raise ValueError("trajectory arrays must have equal lengths")
    if len(distances) < min_samples:
        raise ValueError(
            f"有效样本仅 {len(distances)}，至少需要 {min_samples}")

    def mean_rotation(mask: np.ndarray) -> np.ndarray:
        matrix = rotations[mask].sum(axis=0)
        u, _, vt = np.linalg.svd(matrix)
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(u @ vt)
        return u @ correction @ vt

    def center_fit(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        reference = float(np.median(distances[mask]))
        design = np.c_[np.ones(mask.sum()), distances[mask] - reference]
        coefficients, _, _, _ = np.linalg.lstsq(
            design, centers[mask], rcond=None)
        predicted = np.c_[np.ones(len(distances)),
                          distances - reference] @ coefficients
        return coefficients, predicted, reference

    def rotation_errors(reference: np.ndarray) -> np.ndarray:
        relative = rotations @ reference.T
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) * 0.5,
                         -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    mask = np.ones(len(distances), dtype=bool)
    for _ in range(4):
        if int(mask.sum()) < min_samples:
            break
        _, predicted, _ = center_fit(mask)
        center_errors = np.linalg.norm(centers - predicted, axis=1)
        reference_rotation = mean_rotation(mask)
        angle_errors = rotation_errors(reference_rotation)
        selected = center_errors[mask]
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        robust_limit = max(
            max_center_residual_m,
            median + 3.5 * max(1.4826 * mad, 1e-6),
        )
        new_mask = ((center_errors <= robust_limit)
                    & (angle_errors <= max_rotation_deviation_deg))
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
    if int(mask.sum()) < min_samples:
        raise ValueError(
            f"异常值剔除后仅 {int(mask.sum())} 个样本，至少需要 {min_samples}")

    coefficients, predicted, reference = center_fit(mask)
    rotation = mean_rotation(mask)
    center_errors = np.linalg.norm(centers - predicted, axis=1)
    angle_errors = rotation_errors(rotation)
    return {
        "mean_rotation": rotation,
        "predicted_centers": predicted,
        "inlier_mask": mask,
        "reference_distance_m": reference,
        "center_at_reference_m": coefficients[0],
        "motion_per_nominal_distance": coefficients[1],
        "center_fit_rms_mm": float(
            np.sqrt(np.mean(center_errors[mask] ** 2)) * 1000.0),
        "max_center_residual_mm": float(center_errors[mask].max() * 1000.0),
        "max_rotation_deviation_deg": float(angle_errors[mask].max()),
    }


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) < 3:
        raise RuntimeError(f"点云为空或无法读取: {path}")
    return cloud


def write_json(path: Path, value: dict[str, Any]) -> None:
    def safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(child) for child in item]
        if isinstance(item, np.ndarray):
            return safe(item.tolist())
        if isinstance(item, (np.floating, float)):
            number = float(item)
            return number if np.isfinite(number) else None
        if isinstance(item, np.integer):
            return int(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2,
                               allow_nan=False),
                    encoding="utf-8")


def _normalized_plane(model: Iterable[float]) -> np.ndarray:
    plane = np.asarray(model, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-12:
        raise ValueError("invalid plane")
    return plane / norm


def fit_dominant_plane(
    points: np.ndarray,
    threshold: float = 0.0015,
    iterations: int = 1500,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit the dominant plane and refine it by SVD on RANSAC inliers."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    model, indices = cloud.segment_plane(
        distance_threshold=float(threshold),
        ransac_n=3,
        num_iterations=int(iterations),
    )
    mask = np.zeros(len(pts), dtype=bool)
    mask[np.asarray(indices, dtype=int)] = True
    if mask.sum() < 3:
        mask[:] = True
    center = pts[mask].mean(axis=0)
    _, _, vh = np.linalg.svd(pts[mask] - center, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    plane = np.r_[normal, -normal @ center]
    signed = pts @ plane[:3] + plane[3]
    mask = np.abs(signed) <= threshold
    rms = float(np.sqrt(np.mean(np.square(signed[mask]))))
    return plane, mask, rms


def fit_four_planes(
    clouds: dict[str, np.ndarray],
    threshold: float = 0.0015,
    iterations: int = 1500,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    centers = []
    for face in FACES:
        plane, mask, rms = fit_dominant_plane(
            clouds[face], threshold, iterations)
        center = clouds[face][mask].mean(axis=0)
        centers.append(center)
        result[face] = {
            "plane": plane,
            "mask": mask,
            "rms_m": rms,
            "center": center,
        }
    global_center = np.mean(centers, axis=0)
    for face in FACES:
        entry = result[face]
        plane = entry["plane"]
        # Outward normals give a consistent polygon half-space convention.
        if np.dot(plane[:2], entry["center"][:2] - global_center[:2]) < 0:
            entry["plane"] = -plane
    return result


def plane_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = np.clip(abs(float(np.dot(first[:3], second[:3]))), 0.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def xy_corners(
    planes: dict[str, np.ndarray],
    z_reference: float,
) -> dict[str, np.ndarray]:
    """Intersect adjacent side planes in XY at a representative Z."""
    corners: dict[str, np.ndarray] = {}
    for first, second in PAIRS:
        p, q = planes[first], planes[second]
        matrix = np.array([[p[0], p[1]], [q[0], q[1]]])
        rhs = -np.array([p[3] + p[2] * z_reference,
                         q[3] + q[2] * z_reference])
        if abs(np.linalg.det(matrix)) < 1e-7:
            raise RuntimeError(f"{first}/{second} 平面在 XY 中近似平行")
        corners[f"{first}_{second}"] = np.linalg.solve(matrix, rhs)
    return corners


def corner_distances(
    points: np.ndarray,
    corner_a: np.ndarray,
    corner_b: np.ndarray,
) -> dict[str, float]:
    """XY distance from one face's points to either theoretical edge."""
    xy = np.asarray(points)[:, :2]
    distance = np.minimum(np.linalg.norm(xy - corner_a, axis=1),
                          np.linalg.norm(xy - corner_b, axis=1))
    return {
        "nearest_mm": float(distance.min() * 1000.0),
        "p05_mm": float(np.percentile(distance, 5) * 1000.0),
        "p25_mm": float(np.percentile(distance, 25) * 1000.0),
        "median_mm": float(np.median(distance) * 1000.0),
    }


def proximity_coverage(
    first: np.ndarray,
    second: np.ndarray,
    thresholds_mm: Iterable[float] = (2, 5, 10, 15, 20, 25),
) -> dict[str, dict[str, float]]:
    first = np.asarray(first)
    second = np.asarray(second)
    d12 = cKDTree(second).query(first, k=1)[0]
    d21 = cKDTree(first).query(second, k=1)[0]
    result = {}
    for value in thresholds_mm:
        limit = float(value) * 1e-3
        result[str(value)] = {
            "first_to_second": float(np.mean(d12 <= limit)),
            "second_to_first": float(np.mean(d21 <= limit)),
            "symmetric_min": float(min(np.mean(d12 <= limit),
                                       np.mean(d21 <= limit))),
        }
    return result


def estimate_normals(points: np.ndarray, radius: float) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(radius, 1e-4), max_nn=40))
    normals = np.asarray(cloud.normals).copy()
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    return normals


def voxel_points(points: np.ndarray, voxel: float) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(cloud.voxel_down_sample(voxel).points).copy()


def apply_se3(points: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(vector[:3]).as_matrix()
    return np.asarray(points) @ rotation.T + vector[3:][None, :]


def se3_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    matrix[:3, 3] = vector[3:]
    return matrix


def build_pair_correspondences(
    first_points: np.ndarray,
    first_normals: np.ndarray,
    second_points: np.ndarray,
    second_normals: np.ndarray,
    max_distance: float,
    max_normal_angle_deg: float,
    neighbors: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build bidirectional, normal-consistent correspondences.

    Candidate search happens in a small spatial neighbourhood and rejects
    perpendicular normals, preventing a side's main plane from snapping to
    the adjacent side's main plane at the physical corner.
    """
    cosine = np.cos(np.radians(max_normal_angle_deg))

    def directed(src_p, src_n, dst_p, dst_n):
        count = min(neighbors, len(dst_p))
        distances, indices = cKDTree(dst_p).query(src_p, k=count,
                                                  distance_upper_bound=max_distance)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        chosen = np.full(len(src_p), -1, dtype=int)
        chosen_distance = np.full(len(src_p), np.inf)
        for row in range(len(src_p)):
            valid = indices[row] < len(dst_p)
            if not np.any(valid):
                continue
            candidates = indices[row, valid]
            compatible = np.abs(dst_n[candidates] @ src_n[row]) >= cosine
            if np.any(compatible):
                candidate_dist = distances[row, valid][compatible]
                best = int(np.argmin(candidate_dist))
                chosen[row] = int(candidates[compatible][best])
                chosen_distance[row] = float(candidate_dist[best])
        return chosen, chosen_distance

    forward, fd = directed(first_points, first_normals,
                           second_points, second_normals)
    backward, _ = directed(second_points, second_normals,
                           first_points, first_normals)
    src_indices = np.flatnonzero(forward >= 0)
    dst_indices = forward[src_indices]
    mutual = backward[dst_indices] == src_indices
    src_indices, dst_indices = src_indices[mutual], dst_indices[mutual]
    metrics = {
        "count": int(len(src_indices)),
        "coverage_first": float(len(np.unique(src_indices)) / len(first_points)),
        "coverage_second": float(len(np.unique(dst_indices)) / len(second_points)),
        "rmse_m": float(np.sqrt(np.mean(fd[src_indices] ** 2)))
        if len(src_indices) else float("inf"),
    }
    return src_indices, dst_indices, metrics


def globally_align(
    clouds: dict[str, np.ndarray],
    distance_levels: Iterable[float],
    voxel: float,
    normal_angle_deg: float,
    max_translation: float,
    max_rotation_deg: float,
    huber_scale: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Jointly optimize small corrections, rebuilding matches every level."""
    sampled = {face: voxel_points(clouds[face], voxel) for face in FACES}
    normals = {face: estimate_normals(sampled[face], voxel * 3.0)
               for face in FACES}
    parameters = np.zeros(18, dtype=np.float64)  # face1 is the fixed gauge.
    history: list[dict[str, Any]] = []
    rot_bound = np.radians(max_rotation_deg)
    lower = np.tile([-rot_bound] * 3 + [-max_translation] * 3, 3)
    upper = -lower

    def face_vector(values: np.ndarray, face: str) -> np.ndarray:
        if face == "face1":
            return np.zeros(6)
        index = FACES.index(face) - 1
        return values[index * 6:(index + 1) * 6]

    for distance in distance_levels:
        transformed = {
            face: apply_se3(sampled[face], face_vector(parameters, face))
            for face in FACES
        }
        rotated_normals = {}
        for face in FACES:
            rotation = Rotation.from_rotvec(
                face_vector(parameters, face)[:3]).as_matrix()
            rotated_normals[face] = normals[face] @ rotation.T
        matches = {}
        pair_metrics = {}
        for first, second in PAIRS:
            ia, ib, metrics = build_pair_correspondences(
                transformed[first], rotated_normals[first],
                transformed[second], rotated_normals[second],
                distance, normal_angle_deg)
            matches[(first, second)] = (ia, ib)
            pair_metrics[f"{first}_{second}"] = metrics

        def residual(values: np.ndarray) -> np.ndarray:
            chunks = []
            for first, second in PAIRS:
                ia, ib = matches[(first, second)]
                if len(ia):
                    first_vector = face_vector(values, first)
                    second_vector = face_vector(values, second)
                    pa = apply_se3(sampled[first][ia], first_vector)
                    pb = apply_se3(sampled[second][ib], second_vector)
                    delta = pa - pb
                    ra = Rotation.from_rotvec(first_vector[:3]).as_matrix()
                    rb = Rotation.from_rotvec(second_vector[:3]).as_matrix()
                    na = normals[first][ia] @ ra.T
                    nb = normals[second][ib] @ rb.T
                    signs = np.where(np.sum(na * nb, axis=1) < 0, -1.0, 1.0)
                    common_normal = na + signs[:, None] * nb
                    common_normal /= np.maximum(
                        np.linalg.norm(common_normal, axis=1)[:, None], 1e-12)
                    # Plane residual drives geometry; a weak point residual
                    # resolves along-plane motion without allowing scan-track
                    # sampling differences to dominate the solution.
                    point_plane = np.sum(delta * common_normal, axis=1)
                    chunks.append(point_plane / max(distance, 1e-9))
                    chunks.append(
                        (0.12 * delta / max(distance, 1e-9)).ravel())
            # The board frame is a strong initialization, so weakly prefer the
            # smallest correction when planar strips leave a tangential DOF.
            scales = np.tile(
                [rot_bound] * 3 + [max_translation] * 3, 3)
            chunks.append(0.5 * values / scales)
            return np.concatenate(chunks) if chunks else np.zeros(1)

        if sum(item["count"] for item in pair_metrics.values()) > 0:
            solution = least_squares(
                residual, parameters, bounds=(lower, upper),
                loss="huber", f_scale=huber_scale, max_nfev=100)
            parameters = solution.x
        history.append({
            "max_distance_mm": distance * 1000.0,
            "pairs": pair_metrics,
            "cost": float(np.sum(np.square(residual(parameters)))),
        })
    vectors = {face: face_vector(parameters, face).copy() for face in FACES}
    return vectors, history


def sample_closed_cuboid(
    corners: list[np.ndarray],
    z_min: float,
    z_max: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create side samples and a watertight six-face triangle mesh."""
    bottom = np.c_[np.asarray(corners), np.full(4, z_min)]
    top = np.c_[np.asarray(corners), np.full(4, z_max)]
    vertices = np.vstack([bottom, top])
    triangles = []
    samples = []
    for index in range(4):
        nxt = (index + 1) % 4
        triangles.extend([[index, nxt, 4 + nxt],
                          [index, 4 + nxt, 4 + index]])
        length = np.linalg.norm(corners[nxt] - corners[index])
        nu = max(2, int(np.ceil(length / spacing)) + 1)
        nv = max(2, int(np.ceil((z_max - z_min) / spacing)) + 1)
        u = np.linspace(0.0, 1.0, nu)
        z = np.linspace(z_min, z_max, nv)
        xy = ((1.0 - u[:, None]) * corners[index][None, :]
              + u[:, None] * corners[nxt][None, :])
        samples.append(np.column_stack([
            np.repeat(xy[:, 0], nv),
            np.repeat(xy[:, 1], nv),
            np.tile(z, nu),
        ]))
    triangles.extend([[0, 2, 1], [0, 3, 2],
                      [4, 5, 6], [4, 6, 7]])
    return np.vstack(samples), vertices, np.asarray(triangles, dtype=np.int32)
