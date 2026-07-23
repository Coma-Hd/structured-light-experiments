"""无标定板导轨：第一面 + 右侧第二面的两点云拼接入口。

示例：
    python scripts/6_merge_two_faces.py \
      --config ceshi/rail/two_faces/config.yaml

    python scripts/6_merge_two_faces.py \
      --target path/to/face1/cloud_clean.ply \
      --source path/to/face2/cloud_clean.ply \
      --out-dir ceshi/rail/two_faces/output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.two_face_registration import (  # noqa: E402
    ICPLevel,
    candidate_angles,
    cloud_stats,
    colored_comparison,
    load_cloud,
    merge_clouds,
    register_two_faces,
    register_two_faces_plane_pair,
    transformed_copy,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return data


def _levels(cfg: Dict[str, Any]):
    rows = cfg.get("icp_levels") or [
        {
            "voxel_m": 0.004,
            "max_correspondence_m": 0.015,
            "iterations": 80,
            "method": "point_to_point",
        },
        {
            "voxel_m": 0.002,
            "max_correspondence_m": 0.008,
            "iterations": 60,
            "method": "point_to_plane",
        },
        {
            "voxel_m": 0.001,
            "max_correspondence_m": 0.004,
            "iterations": 40,
            "method": "point_to_plane",
        },
    ]
    return [
        ICPLevel(
            voxel=float(row["voxel_m"]),
            max_correspondence=float(row["max_correspondence_m"]),
            iterations=int(row.get("iterations", 50)),
            method=str(row.get("method", "point_to_plane")),
        )
        for row in rows
    ]


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="两相邻侧面：双平面定向 + 锚点平移 + 多尺度ICP"
    )
    ap.add_argument(
        "--config",
        default="ceshi/rail/two_faces/config.yaml",
        help="两面拼接配置 YAML",
    )
    ap.add_argument("--target", default=None, help="face1 cloud_clean.ply（基准）")
    ap.add_argument("--source", default=None, help="face2 cloud_clean.ply（待对齐）")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--anchors",
        default=None,
        help="可选共享棱对应点 JSON；正方体对称伪匹配时强烈建议使用",
    )
    ap.add_argument(
        "--allow-low-quality",
        action="store_true",
        help="质量门槛未通过时仍返回成功；输出仍会保存",
    )
    args = ap.parse_args()

    cfg_path = _abs(args.config)
    cfg = _load_yaml(cfg_path)
    inputs = cfg.get("inputs") or {}
    target_path = _abs(args.target or inputs.get("face1", ""))
    source_path = _abs(args.source or inputs.get("face2", ""))
    out_dir = _abs(args.out_dir or cfg.get("output_dir", ""))
    if not target_path or not os.path.isfile(target_path):
        raise SystemExit(f"找不到 face1 target: {target_path}")
    if not source_path or not os.path.isfile(source_path):
        raise SystemExit(f"找不到 face2 source: {source_path}")
    if not out_dir:
        raise SystemExit("未设置 output_dir")
    os.makedirs(out_dir, exist_ok=True)
    aligned_dir = os.path.join(out_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    print("[两面拼接]")
    print(f"  target/face1: {target_path}")
    print(f"  source/face2: {source_path}")
    print(f"  output:       {out_dir}")

    target = load_cloud(target_path)
    source = load_cloud(source_path)
    target_stats = cloud_stats(target)
    source_stats = cloud_stats(source)
    print(
        f"  face1: {target_stats['points']} 点, "
        f"span_mm={[round(x*1000, 1) for x in target_stats['span_m']]}"
    )
    print(
        f"  face2: {source_stats['points']} 点, "
        f"span_mm={[round(x*1000, 1) for x in source_stats['span_m']]}"
    )

    levels = _levels(cfg)
    quality = cfg.get("quality") or {}
    init_cfg = cfg.get("initialization") or {}
    init_mode = str(init_cfg.get("mode", "plane_pair")).strip().lower()
    anchors_path_raw = args.anchors or cfg.get("anchors_file")
    source_anchors = target_anchors = None
    anchors_path = None
    if anchors_path_raw:
        anchors_path = _abs(str(anchors_path_raw))
        if os.path.isfile(anchors_path):
            with open(anchors_path, "r", encoding="utf-8") as f:
                anchors_doc = json.load(f)
            target_anchors = (
                anchors_doc.get("target_points")
                or anchors_doc.get("target_face1_points")
            )
            source_anchors = (
                anchors_doc.get("source_points")
                or anchors_doc.get("source_face2_points")
            )
            if not target_anchors or not source_anchors:
                raise SystemExit(
                    "锚点文件缺少 target_points/source_points "
                    "（或旧版 face1/face2 字段）: "
                    f"{anchors_path}"
                )
            if len(target_anchors) != len(source_anchors):
                raise SystemExit(f"两组锚点数量不一致: {anchors_path}")
            print(f"  anchors: {anchors_path} ({len(target_anchors)} pairs)")
        else:
            print(f"[警告] 配置了锚点但文件不存在: {anchors_path}")
    if source_anchors is None:
        print(
            "[警告] 未使用共享棱锚点，将以平面质心估计平移；"
            "正方体对称结构建议补选锚点。"
        )

    try:
        if init_mode == "plane_pair":
            plane_cfg = init_cfg.get("plane_pair") or {}
            refine_levels = (
                levels[1:]
                if bool(plane_cfg.get("skip_first_icp_level", True))
                and len(levels) > 1
                else levels
            )
            T, reg_report = register_two_faces_plane_pair(
                source=source,
                target=target,
                levels=refine_levels,
                source_anchors=source_anchors,
                target_anchors=target_anchors,
                plane_distance_threshold=float(
                    plane_cfg.get("distance_threshold_m", 0.0015)
                ),
                plane_ransac_iters=int(plane_cfg.get("ransac_iters", 4000)),
                plane_min_points=int(plane_cfg.get("min_points", 500)),
                selection_max_correspondence=float(
                    plane_cfg.get("selection_max_correspondence_m", 0.008)
                ),
                correspondence=str(plane_cfg.get("correspondence", "auto")),
                min_fitness=float(quality.get("min_fitness", 0.5)),
                max_rmse=float(quality.get("max_rmse_m", 0.004)),
                max_rotation_change_deg=float(
                    quality.get("max_rotation_change_deg", 15.0)
                ),
                max_anchor_rmse=float(
                    quality.get("max_anchor_rmse_m", 0.010)
                ),
                verbose=True,
            )
        elif init_mode == "angle_search":
            axis = cfg.get("rotation_axis", [0.0, 1.0, 0.0])
            turn = str(cfg.get("turn", "ccw")).strip().lower()
            expected_sign = 1.0 if turn == "ccw" else -1.0
            angle_cfg = cfg.get("angle_search") or {}
            angles = candidate_angles(
                angle_min=float(angle_cfg.get("min_deg", 70.0)),
                angle_max=float(angle_cfg.get("max_deg", 110.0)),
                angle_step=float(angle_cfg.get("step_deg", 10.0)),
                expected_sign=expected_sign,
                try_both_directions=bool(
                    angle_cfg.get("try_both_directions", True)
                ),
            )
            T, reg_report = register_two_faces(
                source=source,
                target=target,
                rotation_axis=axis,
                angles_deg=angles,
                levels=levels,
                min_fitness=float(quality.get("min_fitness", 0.5)),
                max_rmse=float(quality.get("max_rmse_m", 0.004)),
                max_rotation_change_deg=float(
                    quality.get("max_rotation_change_deg", 25.0)
                ),
                source_anchors=source_anchors,
                target_anchors=target_anchors,
                verbose=True,
            )
        else:
            raise ValueError(
                "initialization.mode 必须是 plane_pair 或 angle_search"
            )
    except RuntimeError as exc:
        raise SystemExit(f"粗配准失败: {exc}") from exc

    aligned_source = transformed_copy(source, T)
    merge_cfg = cfg.get("merge") or {}
    merged_raw, merged_clean = merge_clouds(
        target,
        aligned_source,
        voxel=float(merge_cfg.get("voxel_m", 0.0005)),
        sor_neighbors=int(merge_cfg.get("sor_neighbors", 20)),
        sor_std_ratio=float(merge_cfg.get("sor_std_ratio", 2.5)),
    )
    comparison = colored_comparison(target, aligned_source)

    import open3d as o3d

    paths = {
        "face1_reference": os.path.join(aligned_dir, "face1_reference.ply"),
        "face2_aligned": os.path.join(aligned_dir, "face2_aligned.ply"),
        "comparison_colored": os.path.join(out_dir, "comparison_colored.ply"),
        "merged_raw": os.path.join(out_dir, "cloud_merged_raw.ply"),
        "merged_clean": os.path.join(out_dir, "cloud_merged_clean.ply"),
        "transform": os.path.join(out_dir, "transform_face2_to_face1.json"),
        "report": os.path.join(out_dir, "registration_report.json"),
    }
    o3d.io.write_point_cloud(paths["face1_reference"], target)
    o3d.io.write_point_cloud(paths["face2_aligned"], aligned_source)
    o3d.io.write_point_cloud(paths["comparison_colored"], comparison)
    o3d.io.write_point_cloud(paths["merged_raw"], merged_raw)
    o3d.io.write_point_cloud(paths["merged_clean"], merged_clean)

    transform_doc = {
        "convention": "p_face1 = T_face2_to_face1 @ p_face2_h",
        "T_face2_to_face1": T.tolist(),
    }
    _write_json(paths["transform"], transform_doc)
    full_report = {
        "inputs": {
            "face1": target_path,
            "face2": source_path,
            "face1_stats": target_stats,
            "face2_stats": source_stats,
            "anchors_file": anchors_path,
        },
        "registration": reg_report,
        "outputs": paths,
        "merged": {
            "raw_points": int(len(merged_raw.points)),
            "clean_points": int(len(merged_clean.points)),
            "no_largest_cluster_filter": True,
        },
    }
    _write_json(paths["report"], full_report)

    final = reg_report["final"]
    print("")
    print(
        f"[最终] fitness={final['fitness']:.4f}, "
        f"RMSE={final['rmse_m']*1000:.3f} mm, "
        f"accepted={final['accepted']}"
    )
    print(f"已保存彩色接缝检查: {paths['comparison_colored']}")
    print(f"已保存合并点云:     {paths['merged_clean']}")
    print(f"已保存变换:         {paths['transform']}")
    print(f"已保存报告:         {paths['report']}")

    if not final["accepted"] and not args.allow_low_quality:
        raise SystemExit(
            "配准未通过质量门槛；输出已保存供排错。"
            "不要直接用于最终模型。必要时增加共享条带或使用 --allow-low-quality 仅作观察。"
        )


if __name__ == "__main__":
    main()
