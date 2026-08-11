"""HC32 turntable motor control and encoder telemetry.

The controller owns the serial port, renews the firmware's open-loop lease,
polls the continuous output-axis encoder angle, and timestamps every reply
with ``time.perf_counter()`` for camera/encoder synchronization.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import re
import threading
import time
from typing import Deque, Iterable

import serial


BAUD_RATE = 115200
REDUCER_RATIO = 100
MOTOR_FULL_STEPS_PER_REV = 200
MOTOR_RUN_DUTY_PERMIL = 700
MOTOR_SPEED_MIN_RPM = 5.0
MOTOR_SPEED_MAX_RPM = 60.0
FIRMWARE_TICK_US = 1000

HELLO_COMMAND = b"hello\r\n"
HELLO_RESPONSE = "Hello HC32F448 position control"
AXIS_STATUS_COMMAND = b"axis status\r\n"
MOTOR_STOP_COMMAND = b"motor off\r\n"
GLOBAL_STOP_COMMAND = b"stop\r\n"
SAFE_STOP_COMMANDS = MOTOR_STOP_COMMAND + GLOBAL_STOP_COMMAND

# 20 Hz status traffic caused occasional command-byte loss through the
# DAPLink UART bridge (for example 18000 arriving as 1800). 10 Hz still
# brackets camera frames well while leaving firmware command-processing margin.
STATUS_INTERVAL_S = 0.1
LEASE_REFRESH_INTERVAL_S = 0.25
TELEMETRY_TIMEOUT_S = 0.75
MAX_CONSECUTIVE_RUN_REJECTIONS = 3

_AXIS_STATUS = re.compile(
    r"^axis target_mdeg=-?\d+ "
    r"actual_mdeg=(?P<actual_mdeg>-?\d+) "
    r"error_mdeg=-?\d+ "
    r"fault_active=(?P<fault_active>[01]) "
    r"target_valid=[01] "
    r"pid_out=-?\d+$"
)


class TurntableError(RuntimeError):
    """Raised when the turntable cannot be controlled safely."""


@dataclass(frozen=True)
class AxisSample:
    received_at: float
    actual_mdeg: int
    fault_active: bool = False


def output_dps_to_motor_rpm(output_dps: float) -> float:
    """Convert reducer-output degrees/second to motor-shaft RPM."""
    if not math.isfinite(output_dps) or abs(output_dps) < 1e-12:
        raise ValueError("输出轴角速度必须是非零有限值")
    motor_rpm = abs(output_dps) * REDUCER_RATIO / 6.0
    if not MOTOR_SPEED_MIN_RPM <= motor_rpm <= MOTOR_SPEED_MAX_RPM:
        minimum = MOTOR_SPEED_MIN_RPM * 6.0 / REDUCER_RATIO
        maximum = MOTOR_SPEED_MAX_RPM * 6.0 / REDUCER_RATIO
        raise ValueError(
            f"输出轴角速度必须在 {minimum:.3f}..{maximum:.3f} deg/s 范围内"
        )
    return motor_rpm


def motor_speed_to_step_us(motor_rpm: float) -> int:
    """Convert motor RPM to a firmware-tick-aligned commutation interval."""
    if (
        not math.isfinite(motor_rpm)
        or motor_rpm < MOTOR_SPEED_MIN_RPM
        or motor_rpm > MOTOR_SPEED_MAX_RPM
    ):
        raise ValueError(
            f"电机速度必须在 {MOTOR_SPEED_MIN_RPM:g}.."
            f"{MOTOR_SPEED_MAX_RPM:g} rpm 范围内"
        )
    exact_us = 60_000_000.0 / MOTOR_FULL_STEPS_PER_REV / motor_rpm
    return int(math.ceil(exact_us / FIRMWARE_TICK_US) * FIRMWARE_TICK_US)


def build_motor_run_command(output_dps: float) -> bytes:
    """Build an open-loop command; positive output speed maps to CCW."""
    motor_rpm = output_dps_to_motor_rpm(output_dps)
    direction = "ccw" if output_dps > 0 else "cw"
    step_us = motor_speed_to_step_us(motor_rpm)
    return (
        f"motor run {direction} {MOTOR_RUN_DUTY_PERMIL} {step_us}\r\n"
    ).encode("ascii")


def parse_axis_status(line: str, received_at: float) -> AxisSample | None:
    match = _AXIS_STATUS.match(line.strip())
    if match is None:
        return None
    return AxisSample(
        received_at=received_at,
        actual_mdeg=int(match.group("actual_mdeg")),
        fault_active=match.group("fault_active") == "1",
    )


def interpolate_mdeg(
    samples: Iterable[AxisSample],
    timestamp: float,
) -> float:
    """Linearly interpolate continuous encoder angle at a monotonic timestamp."""
    ordered = list(samples)
    if len(ordered) < 2:
        raise ValueError("角度插值至少需要两个编码器样本")
    for left, right in zip(ordered, ordered[1:]):
        if left.received_at <= timestamp <= right.received_at:
            span = right.received_at - left.received_at
            if span <= 0:
                raise ValueError("编码器样本时间戳必须严格递增")
            fraction = (timestamp - left.received_at) / span
            return left.actual_mdeg + fraction * (
                right.actual_mdeg - left.actual_mdeg
            )
    raise ValueError("目标时间不在编码器样本覆盖范围内")


def measured_speed_dps(
    samples: Iterable[AxisSample],
    *,
    window_s: float = 0.5,
) -> float:
    """Calculate signed output speed over the newest telemetry window."""
    ordered = list(samples)
    if len(ordered) < 2:
        raise ValueError("速度计算至少需要两个编码器样本")
    newest = ordered[-1]
    cutoff = newest.received_at - window_s
    oldest = next(
        (sample for sample in ordered if sample.received_at >= cutoff),
        ordered[0],
    )
    elapsed = newest.received_at - oldest.received_at
    if elapsed <= 0:
        raise ValueError("编码器样本时间戳必须严格递增")
    return (newest.actual_mdeg - oldest.actual_mdeg) / 1000.0 / elapsed


class TurntableMotor:
    """Own one HC32 serial port and expose timestamped encoder telemetry."""

    def __init__(self, port: str) -> None:
        self.port = port
        self._serial: serial.Serial | None = None
        self._worker: threading.Thread | None = None
        self._stop_worker = threading.Event()
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._samples: Deque[AxisSample] = deque(maxlen=4096)
        self._run_command: bytes | None = None
        self._error: str | None = None
        self._stop_requested = False
        self._sample_count = 0
        self._sample_interval_sum = 0.0
        self._sample_interval_min: float | None = None
        self._sample_interval_max: float | None = None
        self._consecutive_run_rejections = 0

    @property
    def samples(self) -> list[AxisSample]:
        with self._condition:
            return list(self._samples)

    @property
    def telemetry_stats(self) -> dict[str, float | int | None]:
        with self._condition:
            intervals = max(0, self._sample_count - 1)
            mean_interval = (
                self._sample_interval_sum / intervals if intervals else None
            )
            return {
                "sample_count": self._sample_count,
                "mean_interval_s": mean_interval,
                "min_interval_s": self._sample_interval_min,
                "max_interval_s": self._sample_interval_max,
                "mean_rate_hz": (
                    1.0 / mean_interval
                    if mean_interval is not None and mean_interval > 0
                    else None
                ),
            }

    def connect(self, timeout_s: float = 2.0) -> None:
        if self._serial is not None:
            raise TurntableError("转台串口已经连接")
        try:
            connection = serial.Serial(
                port=self.port,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
                write_timeout=1.0,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            connection.reset_input_buffer()
            connection.reset_output_buffer()
            connection.write(HELLO_COMMAND)
            connection.flush()
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                line = connection.readline().decode("utf-8", errors="replace")
                if HELLO_RESPONSE in line:
                    break
            else:
                connection.close()
                raise TurntableError(
                    f"{self.port} 未返回 HC32 握手应答"
                )
        except (OSError, serial.SerialException) as exc:
            raise TurntableError(f"无法打开转台串口 {self.port}: {exc}") from exc

        self._serial = connection
        self._stop_worker.clear()
        self._error = None
        self._stop_requested = False
        self._worker = threading.Thread(
            target=self._io_loop,
            name="turntable-serial",
            daemon=True,
        )
        self._worker.start()
        self.wait_for_samples(1, timeout_s=TELEMETRY_TIMEOUT_S)

    def start(self, output_dps: float) -> None:
        command = build_motor_run_command(output_dps)
        self._raise_if_unavailable()
        with self._condition:
            self._samples.clear()
            self._sample_count = 0
            self._sample_interval_sum = 0.0
            self._sample_interval_min = None
            self._sample_interval_max = None
            self._consecutive_run_rejections = 0
            self._run_command = command
            self._stop_requested = False
        self._write(AXIS_STATUS_COMMAND + command)

    def wait_until_stable(
        self,
        target_dps: float,
        *,
        timeout_s: float = 10.0,
        stable_duration_s: float = 0.75,
        relative_tolerance: float = 0.20,
        absolute_tolerance_dps: float = 0.05,
    ) -> float:
        """Wait until measured signed speed remains near the requested speed."""
        deadline = time.perf_counter() + timeout_s
        stable_since: float | None = None
        latest_speed = 0.0
        while time.perf_counter() < deadline:
            self.wait_for_samples(2, timeout_s=TELEMETRY_TIMEOUT_S)
            samples = self.samples
            try:
                latest_speed = measured_speed_dps(samples)
            except ValueError:
                continue
            tolerance = max(
                absolute_tolerance_dps,
                abs(target_dps) * relative_tolerance,
            )
            speed_ok = abs(latest_speed - target_dps) <= tolerance
            direction_ok = latest_speed * target_dps > 0
            now = time.perf_counter()
            if speed_ok and direction_ok:
                stable_since = stable_since or now
                if now - stable_since >= stable_duration_s:
                    return latest_speed
            else:
                stable_since = None
            with self._condition:
                self._condition.wait(timeout=STATUS_INTERVAL_S)
            self._raise_if_unavailable()
        raise TurntableError(
            f"转台未在 {timeout_s:g}s 内稳定到 {target_dps:.3f} deg/s；"
            f"最近实测 {latest_speed:.3f} deg/s"
        )

    def wait_until_moving(
        self,
        target_dps: float,
        *,
        timeout_s: float = 10.0,
        moving_duration_s: float = 0.5,
        minimum_speed_dps: float | None = None,
    ) -> float:
        """Wait for sustained encoder movement without requiring stable speed.

        Open-loop speed ripple is acceptable because capture uses encoder
        position directly. The returned speed retains the encoder's raw sign,
        which may be opposite to the requested CCW/CW convention.
        """
        if minimum_speed_dps is None:
            minimum_speed_dps = max(0.05, abs(target_dps) * 0.1)
        deadline = time.perf_counter() + timeout_s
        moving_since: float | None = None
        latest_speed = 0.0
        while time.perf_counter() < deadline:
            self.wait_for_samples(2, timeout_s=TELEMETRY_TIMEOUT_S)
            try:
                latest_speed = measured_speed_dps(self.samples)
            except ValueError:
                continue
            now = time.perf_counter()
            if abs(latest_speed) >= minimum_speed_dps:
                moving_since = moving_since or now
                if now - moving_since >= moving_duration_s:
                    return latest_speed
            else:
                moving_since = None
            with self._condition:
                self._condition.wait(timeout=STATUS_INTERVAL_S)
            self._raise_if_unavailable()
        raise TurntableError(
            f"转台在 {timeout_s:g}s 内未检测到持续运动；"
            f"最近实测 {latest_speed:.3f} deg/s"
        )

    def angle_mdeg_at(self, timestamp: float, timeout_s: float = 0.5) -> float:
        """Return interpolated encoder angle after waiting for a right bracket."""
        value, _span = self.angle_mdeg_at_with_span(timestamp, timeout_s)
        return value

    def angle_mdeg_at_with_span(
        self,
        timestamp: float,
        timeout_s: float = 0.5,
    ) -> tuple[float, float]:
        """Return interpolated angle and the bracketing sample time span."""
        deadline = time.perf_counter() + timeout_s
        while True:
            self._raise_if_unavailable()
            samples = self.samples
            if (
                len(samples) >= 2
                and samples[0].received_at <= timestamp
                and samples[-1].received_at >= timestamp
            ):
                for left, right in zip(samples, samples[1:]):
                    if left.received_at <= timestamp <= right.received_at:
                        return (
                            interpolate_mdeg((left, right), timestamp),
                            right.received_at - left.received_at,
                        )
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TurntableError(
                    "编码器样本未能覆盖相机帧时间，无法可靠插值"
                )
            with self._condition:
                self._condition.wait(timeout=min(remaining, STATUS_INTERVAL_S))

    def latest_speed_dps(self, window_s: float = 0.5) -> float:
        self._raise_if_unavailable()
        return measured_speed_dps(self.samples, window_s=window_s)

    def wait_for_samples(self, count: int, timeout_s: float) -> None:
        deadline = time.perf_counter() + timeout_s
        with self._condition:
            while len(self._samples) < count:
                self._raise_if_unavailable()
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TurntableError("等待编码器角度反馈超时")
                self._condition.wait(timeout=remaining)

    def stop(self) -> None:
        if self._serial is None:
            return
        with self._condition:
            self._run_command = None
            self._stop_requested = True
        try:
            self._write(MOTOR_STOP_COMMAND + AXIS_STATUS_COMMAND)
        except TurntableError:
            self._safe_write(SAFE_STOP_COMMANDS)

    def close(self) -> None:
        connection = self._serial
        if connection is None:
            return
        with self._condition:
            self._run_command = None
            self._stop_requested = True
        self._safe_write(SAFE_STOP_COMMANDS)
        self._stop_worker.set()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        try:
            connection.close()
        except (OSError, serial.SerialException):
            pass
        self._serial = None
        self._worker = None

    def __enter__(self) -> "TurntableMotor":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _write(self, payload: bytes) -> None:
        self._raise_if_unavailable()
        connection = self._serial
        assert connection is not None
        try:
            with self._write_lock:
                connection.write(payload)
                connection.flush()
        except (OSError, serial.SerialException) as exc:
            self._set_error(str(exc))
            raise TurntableError(f"转台串口写入失败: {exc}") from exc

    def _safe_write(self, payload: bytes) -> None:
        connection = self._serial
        if connection is None or not connection.is_open:
            return
        try:
            with self._write_lock:
                connection.write(payload)
                connection.flush()
        except (OSError, serial.SerialException):
            pass

    def _raise_if_unavailable(self) -> None:
        if self._error is not None:
            raise TurntableError(self._error)
        if self._serial is None or not self._serial.is_open:
            raise TurntableError("转台串口未连接")

    def _set_error(self, message: str) -> None:
        with self._condition:
            if self._error is None:
                self._error = message
            self._condition.notify_all()

    def _handle_line(self, line: str, received_at: float) -> None:
        sample = parse_axis_status(line, received_at)
        if sample is not None:
            if sample.fault_active:
                self._set_error("转台报告活动故障，已停止采集")
                return
            with self._condition:
                if not self._samples or (
                    sample.received_at > self._samples[-1].received_at
                ):
                    if self._samples:
                        interval = (
                            sample.received_at - self._samples[-1].received_at
                        )
                        self._sample_interval_sum += interval
                        self._sample_interval_min = (
                            interval
                            if self._sample_interval_min is None
                            else min(self._sample_interval_min, interval)
                        )
                        self._sample_interval_max = (
                            interval
                            if self._sample_interval_max is None
                            else max(self._sample_interval_max, interval)
                        )
                    self._samples.append(sample)
                    self._sample_count += 1
                self._condition.notify_all()
            return
        stripped = line.strip()
        if stripped.startswith(
            ("Motor run STARTED", "Motor run ACTIVE", "Motor run UPDATED")
        ):
            self._consecutive_run_rejections = 0
        elif stripped.startswith("Motor run REJECTED"):
            self._consecutive_run_rejections += 1
            if (
                self._consecutive_run_rejections
                >= MAX_CONSECUTIVE_RUN_REJECTIONS
            ):
                self._set_error(
                    "固件连续拒绝转台运行命令 "
                    f"{self._consecutive_run_rejections} 次: {stripped}"
                )
        elif (
            stripped.startswith("Motor run STOPPED")
            and self._run_command is not None
            and not self._stop_requested
        ):
            self._set_error(f"转台意外停止: {stripped}")

    def _io_loop(self) -> None:
        pending = bytearray()
        next_status = time.perf_counter()
        next_lease = next_status
        try:
            while not self._stop_worker.is_set():
                now = time.perf_counter()
                with self._condition:
                    run_command = self._run_command
                payload = bytearray()
                if run_command is not None and now >= next_lease:
                    payload.extend(run_command)
                    next_lease = now + LEASE_REFRESH_INTERVAL_S
                if now >= next_status:
                    payload.extend(AXIS_STATUS_COMMAND)
                    next_status = now + STATUS_INTERVAL_S
                if payload:
                    self._write(bytes(payload))

                connection = self._serial
                if connection is None:
                    break
                waiting = connection.in_waiting
                chunk = connection.read(waiting if waiting > 0 else 1)
                if not chunk:
                    continue
                pending.extend(chunk)
                while b"\n" in pending:
                    raw, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    line = raw.rstrip(b"\r").decode(
                        "utf-8", errors="replace"
                    )
                    if line:
                        self._handle_line(line, time.perf_counter())
        except (OSError, TurntableError, serial.SerialException) as exc:
            if not self._stop_worker.is_set():
                self._set_error(f"转台串口线程失败: {exc}")
        finally:
            if self._error is not None:
                self._safe_write(SAFE_STOP_COMMANDS)
