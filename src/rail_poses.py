"""导轨行程位姿：从 positions.csv 按图像文件名对齐位移。"""
from __future__ import annotations

import csv
import os
from typing import Dict, Mapping, Optional


_DISTANCE_KEYS = (
    "distance_mm",
    "distance_m",
    "distance",
    "s_mm",
    "s_m",
    "s",
    "pos_mm",
    "pos",
)


def _unit_to_meters(value: float, unit: str) -> float:
    u = (unit or "mm").strip().lower()
    if u in ("mm", "millimeter", "millimetre"):
        return float(value) * 1e-3
    if u in ("m", "meter", "metre"):
        return float(value)
    if u in ("cm", "centimeter", "centimetre"):
        return float(value) * 1e-2
    raise ValueError(f"unsupported rail distance_unit: {unit}")


def detect_distance_column(fieldnames) -> Optional[str]:
    if not fieldnames:
        return None
    lower_map = {name.lower().strip(): name for name in fieldnames}
    for key in _DISTANCE_KEYS:
        if key in lower_map:
            return lower_map[key]
    return None


def load_rail_positions(
        csv_path: str,
        distance_unit: str = "mm",
        ) -> Dict[str, float]:
    """读取 positions.csv，返回 {basename: distance_meters}。

    支持列名：
      image|file|filename|name + distance_mm|distance_m|distance|s|...
    若只有一列数值且文件名来自行索引，不支持——必须有 image 列。
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"rail positions file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty positions csv: {csv_path}")

        fields = [c.strip() for c in reader.fieldnames]
        lower_map = {c.lower(): c for c in fields}

        image_col = None
        for key in ("image", "file", "filename", "name", "img"):
            if key in lower_map:
                image_col = lower_map[key]
                break
        if image_col is None:
            raise ValueError(
                f"positions csv missing image column (image/file/filename): {csv_path}")

        dist_col = detect_distance_column(fields)
        if dist_col is None:
            raise ValueError(
                f"positions csv missing distance column "
                f"(distance_mm/distance_m/distance/...): {csv_path}")

        # If column name encodes unit, prefer that over config unit.
        col_l = dist_col.lower()
        unit = distance_unit
        if col_l.endswith("_mm") or col_l in ("s_mm", "pos_mm"):
            unit = "mm"
        elif col_l.endswith("_m") or col_l in ("s_m",):
            unit = "m"

        out: Dict[str, float] = {}
        for row in reader:
            name = (row.get(image_col) or "").strip()
            if not name:
                continue
            raw = (row.get(dist_col) or "").strip()
            if raw == "":
                continue
            basename = os.path.basename(name.replace("\\", "/"))
            out[basename] = _unit_to_meters(float(raw), unit)

    if not out:
        raise ValueError(f"no valid rows in positions csv: {csv_path}")
    return out


def resolve_positions_path(image_dir: str, positions_file: str) -> str:
    """positions_file 可以是绝对路径，或相对扫描目录 / 工程相对路径。"""
    if os.path.isabs(positions_file) and os.path.isfile(positions_file):
        return positions_file
    cand = os.path.join(image_dir, positions_file)
    if os.path.isfile(cand):
        return cand
    if os.path.isfile(positions_file):
        return positions_file
    raise FileNotFoundError(
        f"positions file not found: tried '{cand}' and '{positions_file}'")


def lookup_distance(positions: Mapping[str, float], image_path: str) -> Optional[float]:
    """按 basename 查找行程（米）；找不到返回 None。"""
    return positions.get(os.path.basename(image_path))
