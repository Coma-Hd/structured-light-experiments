"""连续匀速导轨采集。

适用场景：
1. 步进电机由外部控制器启动，并已进入稳定匀速阶段。
2. 操作者在预览窗口按空格开始采集。
3. 程序按“空间步距 / 导轨速度”定时保存图像。
4. 每张图按实际采集时间自动写入 positions.csv。

按键：
    空格 / s : 开始采集；采集中再次按下则结束并退出
    q / ESC  : 结束并退出

示例：
    python scripts/continuous_rail_capture.py ^
      --out ceshi/rail/scan/two_faces_face1 ^
      --velocity-mm-s 1.0 --step-mm 0.5 ^
      --cam 0 --width 800 --height 600 --exposure -5 --gain 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.charuco import CharucoTarget  # noqa: E402
from src.charuco_tracking import corner_area_ratio  # noqa: E402
from src.config import CharucoConfig, load_config  # noqa: E402
from src.io_utils import imwrite_unicode  # noqa: E402
from src.laser_center import blue_laser_score  # noqa: E402


AUTO_EXPOSURE_OFF = 0.25


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="连续匀速导轨采集：按空间步距抽帧并自动生成 positions.csv"
    )
    parser.add_argument("--out", required=True, help="图像和 positions.csv 输出目录")
    parser.add_argument("--velocity-mm-s", type=float, default=1.0,
                        help="导轨名义匀速，单位 mm/s（默认 1.0）")
    parser.add_argument("--step-mm", type=float, default=0.5,
                        help="目标抽帧间距，单位 mm（默认 0.5）")
    parser.add_argument("--start-mm", type=float, default=0.0,
                        help="第一张图在 positions.csv 中的位置（默认 0）")
    parser.add_argument("--max-travel-mm", type=float, default=0.0,
                        help="自动停止的采集长度；0 表示手动停止（默认 0）")
    parser.add_argument("--cam", type=int, default=0, help="摄像头索引")
    parser.add_argument("--width", type=int, default=800, help="采集宽度")
    parser.add_argument("--height", type=int, default=600, help="采集高度")
    parser.add_argument("--exposure", type=float, default=-5.0, help="手动曝光值")
    parser.add_argument("--gain", type=float, default=1.0, help="相机增益")
    parser.add_argument("--brightness", type=float, default=None, help="相机亮度")
    parser.add_argument("--config", default=None,
                        help="扫描配置文件，用于蓝色激光响应预览")
    parser.add_argument("--prefix", default="img", help="图片文件名前缀")
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="开始前删除输出目录中旧的同前缀 PNG、positions.csv 和采集报告",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.velocity_mm_s <= 0:
        raise ValueError("--velocity-mm-s 必须大于 0")
    if args.step_mm <= 0:
        raise ValueError("--step-mm 必须大于 0")
    if args.max_travel_mm < 0:
        raise ValueError("--max-travel-mm 不能小于 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width 和 --height 必须大于 0")


def _prepare_output(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    positions_path = out_dir / "positions.csv"
    report_path = out_dir / "continuous_capture_report.json"
    old_images = sorted(out_dir.glob(f"{args.prefix}_*.png"))
    old_files = old_images + [
        path for path in (positions_path, report_path) if path.exists()
    ]

    if old_files and not args.clear_output:
        examples = ", ".join(path.name for path in old_files[:5])
        raise FileExistsError(
            f"输出目录已有旧采集文件：{examples}。"
            "请更换目录，或确认后加 --clear-output。"
        )
    if args.clear_output:
        for path in old_files:
            path.unlink()
    return out_dir, positions_path, report_path


def _open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.cam}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_OFF)
    cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    cap.set(cv2.CAP_PROP_GAIN, args.gain)
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[相机] 实际分辨率：{actual_width}x{actual_height}")
    if (actual_width, actual_height) != (args.width, args.height):
        cap.release()
        raise RuntimeError(
            f"相机没有采用请求的 {args.width}x{args.height}，"
            f"实际为 {actual_width}x{actual_height}。"
            "扫描分辨率必须与相机内参标定一致。"
        )
    return cap


def _open_positions_csv(path: Path) -> tuple[TextIO, csv.writer]:
    handle = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(handle)
    writer.writerow(["image", "distance_mm"])
    handle.flush()
    return handle, writer


def _draw_status(
    frame,
    *,
    started: bool,
    count: int,
    distance_mm: float,
    interval_s: float,
    exposure: float,
    gain: float,
    brightness: float,
    show_laser: bool,
    show_charuco: bool,
    charuco_count: int,
    charuco_area_ratio: float,
    min_charuco_corners: int,
    min_charuco_area_ratio: float,
):
    vis = frame.copy()
    if started:
        status = "REC"
        color = (0, 0, 255)
        cv2.circle(vis, (vis.shape[1] - 35, 35), 12, color, -1)
    else:
        status = "READY - start motor, then SPACE"
        color = (0, 255, 255)
    cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, color, 2)
    cv2.putText(
        vis,
        f"saved={count}  distance={distance_mm:.3f} mm  interval={interval_s:.3f} s",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        (
            f"exp={exposure:.1f}  gain={gain:.1f}  bright={brightness:.1f}  "
            f"laser={'ON' if show_laser else 'OFF'}  "
            f"board={'ON' if show_charuco else 'OFF'}"
        ),
        (10, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    board_ok = (
        charuco_count >= min_charuco_corners
        and charuco_area_ratio >= min_charuco_area_ratio
    )
    cv2.putText(
        vis,
        (
            f"ChArUco points={charuco_count}  "
            f"area={charuco_area_ratio*100:.2f}%  "
            f"{'PASS' if board_ok else 'NOT ENOUGH'}"
        ),
        (10, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if board_ok else (0, 0, 255),
        2,
    )
    cv2.putText(
        vis,
        "E/D: exposure  R/F: gain  B/V: brightness  L: laser  C: ChArUco",
        (10, vis.shape[0] - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        vis,
        "SPACE/S: start or finish   Q/ESC: finish",
        (10, vis.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )
    return vis


def _write_report(
    report_path: Path,
    args: argparse.Namespace,
    distances_mm: list[float],
    capture_times_s: list[float],
    stopped_by: str,
) -> None:
    spacings = [
        distances_mm[index] - distances_mm[index - 1]
        for index in range(1, len(distances_mm))
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "velocity_mm_s_assumed": float(args.velocity_mm_s),
        "requested_step_mm": float(args.step_mm),
        "requested_interval_s": float(args.step_mm / args.velocity_mm_s),
        "start_mm": float(args.start_mm),
        "camera_settings": {
            "exposure": float(args.exposure),
            "gain": float(args.gain),
            "brightness": (
                None if args.brightness is None else float(args.brightness)
            ),
        },
        "saved_images": len(distances_mm),
        "recorded_span_mm": (
            float(distances_mm[-1] - distances_mm[0]) if len(distances_mm) >= 2 else 0.0
        ),
        "elapsed_s": (
            float(capture_times_s[-1] - capture_times_s[0])
            if len(capture_times_s) >= 2
            else 0.0
        ),
        "actual_spacing_mm": {
            "min": float(min(spacings)) if spacings else None,
            "max": float(max(spacings)) if spacings else None,
            "mean": float(statistics.fmean(spacings)) if spacings else None,
        },
        "stopped_by": stopped_by,
        "position_formula": (
            "distance_mm = start_mm + "
            "(frame_perf_counter - first_saved_frame_perf_counter) * velocity_mm_s"
        ),
        "warning": (
            "位置基于匀速假设。ChArUco rail_fit 会从固定板位姿估计整体位移比例，"
            "但明显加减速、丢步或机械转动仍会增大轨迹残差。"
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    try:
        _validate_args(args)
        cfg = load_config(args.config)
        charuco_target = CharucoTarget(CharucoConfig.from_cfg(cfg))
        out_dir, positions_path, report_path = _prepare_output(args)
        cap = _open_camera(args)
    except (ValueError, FileExistsError, RuntimeError) as error:
        print(f"[错误] {error}")
        return 2

    interval_s = args.step_mm / args.velocity_mm_s
    print(f"[参数] 匀速={args.velocity_mm_s:.6f} mm/s")
    print(f"[参数] 目标步距={args.step_mm:.6f} mm")
    print(f"[参数] 保存周期={interval_s:.6f} s")
    print("[操作] 先让电机达到稳定匀速；激光到达起点前按空格开始。")
    print("[操作] 采集中再次按空格/s或按q结束；结束录制后再停止电机。")
    print(
        "[调节] E/D 曝光、R/F 增益、B/V 亮度、L 激光响应、"
        "C ChArUco可见性。"
    )

    csv_handle: TextIO | None = None
    csv_writer: csv.writer | None = None
    started = False
    finished = False
    first_capture_time: float | None = None
    next_due_time: float | None = None
    count = 0
    distances_mm: list[float] = []
    capture_times_s: list[float] = []
    stopped_by = "unknown"
    last_distance = args.start_mm
    exposure = float(args.exposure)
    gain = float(args.gain)
    brightness_value = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    brightness = (
        float(args.brightness)
        if args.brightness is not None
        else float(brightness_value if brightness_value not in (0.0, -1.0) else 128.0)
    )
    args.brightness = brightness
    show_laser = False
    show_charuco = True
    charuco_detection = None
    charuco_count = 0
    charuco_area = 0.0
    preview_frame_index = 0
    min_charuco_corners = int(
        (cfg.get("gating") or {}).get("min_charuco_corners", 6)
    )
    min_charuco_area = float(
        (cfg.get("charuco_tracking") or {}).get(
            "min_corner_area_ratio", 0.002
        )
    )
    score_mode = cfg.get("laser", {}).get("score_mode", "blue_minus_max")
    blue_gate_threshold = float(
        cfg.get("laser", {}).get("blue_gate_threshold", 20.0)
    )

    try:
        while not finished:
            ok, frame = cap.read()
            frame_time = time.perf_counter()
            if not ok:
                print("[错误] 相机读取失败")
                stopped_by = "camera_read_failed"
                break
            preview_frame_index += 1

            if started:
                if first_capture_time is None:
                    first_capture_time = frame_time
                    next_due_time = frame_time

                assert next_due_time is not None
                if frame_time >= next_due_time:
                    elapsed_s = frame_time - first_capture_time
                    distance_mm = args.start_mm + elapsed_s * args.velocity_mm_s
                    name = f"{args.prefix}_{count:06d}.png"
                    path = out_dir / name
                    if not imwrite_unicode(str(path), frame):
                        raise RuntimeError(f"图像保存失败：{path}")

                    assert csv_handle is not None and csv_writer is not None
                    csv_writer.writerow([name, f"{distance_mm:.6f}"])
                    csv_handle.flush()
                    count += 1
                    last_distance = distance_mm
                    distances_mm.append(distance_mm)
                    capture_times_s.append(frame_time)
                    print(
                        f"[保存] {name}  t={elapsed_s:.3f} s  "
                        f"distance={distance_mm:.3f} mm"
                    )

                    next_due_time += interval_s
                    skipped = 0
                    while next_due_time <= frame_time:
                        next_due_time += interval_s
                        skipped += 1
                    if skipped:
                        print(
                            f"[警告] 保存或系统延迟超过采样周期，"
                            f"跳过了 {skipped} 个目标位置。"
                        )

                    if (
                        args.max_travel_mm > 0
                        and distance_mm - args.start_mm >= args.max_travel_mm
                    ):
                        stopped_by = "max_travel"
                        finished = True

            preview_frame = frame
            if show_laser:
                score = blue_laser_score(
                    frame,
                    score_mode,
                    blue_gate_threshold=blue_gate_threshold,
                )
                heat = cv2.applyColorMap(score.astype("uint8"), cv2.COLORMAP_JET)
                preview_frame = cv2.addWeighted(frame, 0.5, heat, 0.5, 0)
            if show_charuco and preview_frame_index % 5 == 1:
                charuco_detection = charuco_target.detect(frame)
                if charuco_detection is None:
                    charuco_count = 0
                    charuco_area = 0.0
                else:
                    charuco_count = charuco_detection.count
                    charuco_area = corner_area_ratio(
                        charuco_detection.corners, frame.shape
                    )
            if show_charuco and charuco_detection is not None:
                for point in charuco_detection.corners:
                    x, y = np.rint(point).astype(int)
                    cv2.circle(preview_frame, (x, y), 3, (0, 255, 0), -1)

            vis = _draw_status(
                preview_frame,
                started=started,
                count=count,
                distance_mm=last_distance,
                interval_s=interval_s,
                exposure=exposure,
                gain=gain,
                brightness=brightness,
                show_laser=show_laser,
                show_charuco=show_charuco,
                charuco_count=charuco_count,
                charuco_area_ratio=charuco_area,
                min_charuco_corners=min_charuco_corners,
                min_charuco_area_ratio=min_charuco_area,
            )
            cv2.imshow("continuous rail capture", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                stopped_by = "keyboard"
                finished = True
            elif key in (ord("l"), ord("L")):
                if started:
                    print("[提示] 采集中不切换激光预览，避免影响定时保存。")
                else:
                    show_laser = not show_laser
                    print(f"[激光响应] {'开启' if show_laser else '关闭'}")
            elif key in (ord("c"), ord("C")):
                show_charuco = not show_charuco
                print(
                    f"[ChArUco预览] {'开启' if show_charuco else '关闭'}"
                )
            elif key in (ord("e"), ord("E"), ord("d"), ord("D")):
                if started:
                    print("[提示] 采集中禁止修改曝光，避免同一批图像亮度不一致。")
                else:
                    exposure += 1.0 if key in (ord("e"), ord("E")) else -1.0
                    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                    args.exposure = exposure
                    print(f"[曝光] {exposure:.1f}")
            elif key in (ord("r"), ord("R"), ord("f"), ord("F")):
                if started:
                    print("[提示] 采集中禁止修改增益，避免同一批图像亮度不一致。")
                else:
                    gain += 1.0 if key in (ord("r"), ord("R")) else -1.0
                    gain = max(0.0, gain)
                    cap.set(cv2.CAP_PROP_GAIN, gain)
                    args.gain = gain
                    print(f"[增益] {gain:.1f}")
            elif key in (ord("b"), ord("B"), ord("v"), ord("V")):
                if started:
                    print("[提示] 采集中禁止修改亮度，避免同一批图像亮度不一致。")
                else:
                    brightness += 5.0 if key in (ord("b"), ord("B")) else -5.0
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                    args.brightness = brightness
                    print(f"[亮度] {brightness:.1f}")
            elif key in (ord(" "), ord("s"), ord("S")):
                if not started:
                    if show_laser:
                        show_laser = False
                        print("[激光响应] 开始采集，已自动关闭预览叠加；保存的始终是原始帧。")
                    csv_handle, csv_writer = _open_positions_csv(positions_path)
                    started = True
                    print("[REC] 开始采集。请保持电机匀速运行。")
                else:
                    stopped_by = "keyboard"
                    finished = True
    except KeyboardInterrupt:
        stopped_by = "ctrl_c"
        print("\n[停止] 收到 Ctrl+C")
    except RuntimeError as error:
        stopped_by = "runtime_error"
        print(f"[错误] {error}")
    finally:
        if csv_handle is not None:
            csv_handle.close()
        cap.release()
        cv2.destroyAllWindows()
        if distances_mm:
            _write_report(
                report_path,
                args,
                distances_mm,
                capture_times_s,
                stopped_by,
            )

    if not distances_mm:
        print("[结果] 没有保存图像。")
        return 1

    print("")
    print(f"[完成] 图像数量：{len(distances_mm)}")
    print(f"[完成] 采集跨度：{distances_mm[-1] - distances_mm[0]:.3f} mm")
    print(f"[完成] 位置表：{positions_path}")
    print(f"[完成] 报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
