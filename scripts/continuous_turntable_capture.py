"""控制 HC32 转台连续采集图像，并生成编码器实测的 angles.csv。"""
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import imwrite_unicode  # noqa: E402
from src.turntable_motor import (  # noqa: E402
    TurntableError,
    TurntableMotor,
    output_dps_to_motor_rpm,
)


AUTO_EXPOSURE_OFF = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="连续匀速转台采集：按角度步距抽帧并生成 angles.csv"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--motor-port",
        required=True,
        help="HC32 USART3 在 Windows 中枚举出的串口，例如 COM5",
    )
    parser.add_argument(
        "--angular-velocity-deg-s",
        type=float,
        required=True,
        help="目标输出轴角速度；正数为 ccw，负数为 cw，单位 deg/s",
    )
    parser.add_argument("--step-deg", type=float, default=0.2)
    parser.add_argument("--start-deg", type=float, default=0.0)
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=360.0,
        help="达到该累计绝对角度后自动停止；0 表示手动停止",
    )
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--exposure", type=float, default=-4.0)
    parser.add_argument("--gain", type=float, default=10.0)
    parser.add_argument("--brightness", type=float, default=None)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="设置曝光和增益后丢弃的预热帧数（默认 5）",
    )
    parser.add_argument("--prefix", default="img")
    parser.add_argument(
        "--stable-timeout-s",
        type=float,
        default=10.0,
        help="等待编码器检测到持续运动的最长时间",
    )
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    output_dps_to_motor_rpm(args.angular_velocity_deg_s)
    if args.step_deg <= 0:
        raise ValueError("--step-deg 必须大于 0")
    if args.max_rotation_deg < 0:
        raise ValueError("--max-rotation-deg 不能小于 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width 和 --height 必须大于 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames 不能小于 0")
    if args.stable_timeout_s <= 0:
        raise ValueError("--stable-timeout-s 必须大于 0")


def prepare_output(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    angles_path = out_dir / "angles.csv"
    report_path = out_dir / "continuous_capture_report.json"
    old_files = sorted(out_dir.glob(f"{args.prefix}_*.png"))
    old_files += [
        path for path in (angles_path, report_path) if path.exists()
    ]
    if old_files and not args.clear_output:
        examples = ", ".join(path.name for path in old_files[:5])
        raise FileExistsError(
            f"输出目录已有旧采集文件：{examples}。请更换目录或加 --clear-output。"
        )
    if args.clear_output:
        for path in old_files:
            path.unlink()
    return out_dir, angles_path, report_path


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
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
    actual_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if actual_size != (args.width, args.height):
        cap.release()
        raise RuntimeError(
            f"相机实际分辨率 {actual_size[0]}x{actual_size[1]}，"
            f"不是请求的 {args.width}x{args.height}"
        )
    for index in range(args.warmup_frames):
        ok, _frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(
                f"相机预热失败：无法读取第 {index + 1} 帧"
            )
    if args.warmup_frames:
        print(
            f"[相机] 已丢弃前 {args.warmup_frames} 帧，"
            "等待曝光和增益设置生效"
        )
    return cap


def open_angles_csv(path: Path) -> tuple[TextIO, csv.writer]:
    handle = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(handle)
    writer.writerow(["image", "angle_deg", "elapsed_s"])
    handle.flush()
    return handle, writer


def draw_status(
    frame,
    *,
    count: int,
    angle_deg: float,
    angular_velocity: float,
):
    view = frame.copy()
    color = (0, 0, 255)
    cv2.putText(
        view, "REC - encoder synchronized", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2
    )
    cv2.putText(
        view,
        (
            f"saved={count}  angle={angle_deg:.3f} deg  "
            f"target={angular_velocity:.3f} deg/s"
        ),
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        view,
        "angle source=VCE2755 actual_mdeg",
        (10, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        view,
        "SPACE/S/Q/ESC: stop scan and motor",
        (10, view.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )
    return view


def write_report(
    path: Path,
    args: argparse.Namespace,
    angles_deg: list[float],
    capture_times_s: list[float],
    interpolation_spans_s: list[float],
    telemetry_stats: dict[str, float | int | None],
    encoder_direction_multiplier: int,
    stopped_by: str,
) -> None:
    increments = [
        angles_deg[index] - angles_deg[index - 1]
        for index in range(1, len(angles_deg))
    ]
    measured_speeds = [
        increments[index - 1]
        / (capture_times_s[index] - capture_times_s[index - 1])
        for index in range(1, len(capture_times_s))
        if capture_times_s[index] > capture_times_s[index - 1]
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "motion": "encoder_synchronized_turntable",
        "angle_source": "VCE2755 actual_mdeg linear interpolation",
        "motor_port": args.motor_port,
        "target_angular_velocity_deg_s": float(args.angular_velocity_deg_s),
        "encoder_direction_multiplier": encoder_direction_multiplier,
        "requested_step_deg": float(args.step_deg),
        "start_deg": float(args.start_deg),
        "saved_images": len(angles_deg),
        "recorded_span_deg": (
            float(angles_deg[-1] - angles_deg[0]) if len(angles_deg) >= 2 else 0.0
        ),
        "elapsed_s": (
            float(capture_times_s[-1] - capture_times_s[0])
            if len(capture_times_s) >= 2 else 0.0
        ),
        "actual_increment_deg": {
            "min": float(min(increments)) if increments else None,
            "max": float(max(increments)) if increments else None,
            "mean": float(statistics.fmean(increments)) if increments else None,
        },
        "measured_speed_deg_s": {
            "min": float(min(measured_speeds)) if measured_speeds else None,
            "max": float(max(measured_speeds)) if measured_speeds else None,
            "mean": (
                float(statistics.fmean(measured_speeds))
                if measured_speeds
                else None
            ),
        },
        "encoder_telemetry": telemetry_stats,
        "interpolation_bracket_span_s": {
            "min": (
                float(min(interpolation_spans_s))
                if interpolation_spans_s
                else None
            ),
            "max": (
                float(max(interpolation_spans_s))
                if interpolation_spans_s
                else None
            ),
            "mean": (
                float(statistics.fmean(interpolation_spans_s))
                if interpolation_spans_s
                else None
            ),
        },
        "camera_settings": {
            "exposure": float(args.exposure),
            "gain": float(args.gain),
            "brightness": args.brightness,
            "warmup_frames": int(args.warmup_frames),
        },
        "stopped_by": stopped_by,
        "angle_formula": (
            "angle_deg = start_deg + "
            "(interpolated_actual_mdeg - first_frame_actual_mdeg) / 1000"
        ),
        "warning": "编码器回复时间用于同步；串口时延和相机曝光时刻仍会形成同步误差。",
    }
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    cap: cv2.VideoCapture | None = None
    motor: TurntableMotor | None = None
    try:
        validate_args(args)
        out_dir, angles_path, report_path = prepare_output(args)
        cap = open_camera(args)
        motor = TurntableMotor(args.motor_port)
        print(f"[连接] 正在连接转台 {args.motor_port}（115200 8N1）")
        motor.connect()
        motor.start(args.angular_velocity_deg_s)
        print(
            f"[启动] 目标输出轴速度 "
            f"{args.angular_velocity_deg_s:.3f} deg/s，等待编码器检测运动"
        )
        measured_start_speed = motor.wait_until_moving(
            args.angular_velocity_deg_s,
            timeout_s=args.stable_timeout_s,
        )
        encoder_direction_multiplier = (
            1
            if measured_start_speed * args.angular_velocity_deg_s > 0
            else -1
        )
        print(
            f"[运动] 编码器实测速度 {measured_start_speed:.3f} deg/s；"
            "允许开环速度波动"
        )
        if encoder_direction_multiplier < 0:
            print("[方向] 编码器原始符号与控制方向相反，已自动校正")
    except (
        ValueError,
        FileExistsError,
        RuntimeError,
        TurntableError,
    ) as error:
        print(f"[错误] {error}")
        if motor is not None:
            motor.close()
        if cap is not None:
            cap.release()
        return 2

    print(f"[参数] 目标角速度={args.angular_velocity_deg_s:.6f} deg/s")
    print(f"[参数] 目标角步距={args.step_deg:.6f} deg")
    print("[角度] 每帧使用 VCE2755 actual_mdeg 时间插值")
    print("[操作] 程序已自动启动转台；按 SPACE/S/Q/ESC 可提前停止。")

    csv_handle: TextIO | None = None
    csv_writer: csv.writer | None = None
    finished = False
    first_capture_time: float | None = None
    first_angle_mdeg: float | None = None
    count = 0
    angles_deg: list[float] = []
    capture_times_s: list[float] = []
    interpolation_spans_s: list[float] = []
    last_angle = float(args.start_deg)
    stopped_by = "unknown"
    failed = False

    try:
        csv_handle, csv_writer = open_angles_csv(angles_path)
        while not finished:
            ok, frame = cap.read()
            frame_time = time.perf_counter()
            if not ok:
                stopped_by = "camera_read_failed"
                failed = True
                break

            angle_mdeg, bracket_span_s = motor.angle_mdeg_at_with_span(
                frame_time
            )
            if first_capture_time is None or first_angle_mdeg is None:
                first_capture_time = frame_time
                first_angle_mdeg = angle_mdeg
            elapsed_s = frame_time - first_capture_time
            relative_angle_deg = (
                encoder_direction_multiplier
                * (angle_mdeg - first_angle_mdeg)
                / 1000.0
            )
            angle_deg = args.start_deg + relative_angle_deg

            direction_check_threshold = max(0.05, args.step_deg * 0.5)
            if (
                abs(relative_angle_deg) >= direction_check_threshold
                and relative_angle_deg * args.angular_velocity_deg_s < 0
            ):
                raise TurntableError(
                    "编码器角度方向与请求方向不一致；"
                    "请检查 ccw/cw 与转轴正方向定义"
                )

            should_save = count == 0 or abs(angle_deg - last_angle) >= args.step_deg
            reached_limit = (
                args.max_rotation_deg > 0
                and abs(relative_angle_deg) >= args.max_rotation_deg
            )
            if should_save or reached_limit:
                name = f"{args.prefix}_{count:06d}.png"
                image_path = out_dir / name
                if not imwrite_unicode(str(image_path), frame):
                    raise RuntimeError(f"图像保存失败：{image_path}")
                assert csv_handle is not None and csv_writer is not None
                csv_writer.writerow([
                    name, f"{angle_deg:.9f}", f"{elapsed_s:.9f}"
                ])
                csv_handle.flush()
                angles_deg.append(angle_deg)
                capture_times_s.append(frame_time)
                interpolation_spans_s.append(bracket_span_s)
                count += 1
                increment = abs(angle_deg - last_angle)
                last_angle = angle_deg
                print(
                    f"[保存] {name}  t={elapsed_s:.3f}s  "
                    f"angle={angle_deg:.4f}deg"
                )
                if count > 1 and increment >= args.step_deg * 1.8:
                    print(
                        f"[警告] 相机帧率不足，实际角步距达到 "
                        f"{increment:.4f} deg"
                    )
            if reached_limit:
                stopped_by = "max_rotation"
                finished = True

            cv2.imshow(
                "continuous turntable capture",
                draw_status(
                    frame,
                    count=count,
                    angle_deg=angle_deg,
                    angular_velocity=args.angular_velocity_deg_s,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (
                ord("q"),
                ord("Q"),
                ord("s"),
                ord("S"),
                ord(" "),
                27,
            ):
                stopped_by = "keyboard"
                finished = True
    except KeyboardInterrupt:
        stopped_by = "ctrl_c"
    except (RuntimeError, TurntableError, ValueError) as error:
        stopped_by = "runtime_error"
        failed = True
        print(f"[错误] {error}")
    finally:
        if csv_handle is not None:
            csv_handle.close()
        telemetry_stats = motor.telemetry_stats
        motor.stop()
        motor.close()
        cap.release()
        cv2.destroyAllWindows()
        if angles_deg:
            write_report(
                report_path,
                args,
                angles_deg,
                capture_times_s,
                interpolation_spans_s,
                telemetry_stats,
                encoder_direction_multiplier,
                stopped_by,
            )

    if not angles_deg:
        print("[结果] 没有保存图像。")
        return 2 if failed else 1
    print(f"[完成] 图像数量：{len(angles_deg)}")
    print(f"[完成] 角度跨度：{angles_deg[-1] - angles_deg[0]:.3f} deg")
    print(f"[完成] 角度表：{angles_path}")
    print(f"[完成] 报告：{report_path}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
