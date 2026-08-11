import threading
import time
import unittest
from unittest.mock import patch

from src.turntable_motor import (
    AxisSample,
    TurntableError,
    TurntableMotor,
    build_motor_run_command,
    interpolate_mdeg,
    measured_speed_dps,
    motor_speed_to_step_us,
    output_dps_to_motor_rpm,
    parse_axis_status,
)


class FakeSerial:
    instances = []

    def __init__(self, **_kwargs):
        self.is_open = True
        self.timeout = 0.05
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._running = False
        self._direction = 1
        self._angle_mdeg = 0
        self.writes = []
        self.__class__.instances.append(self)

    @property
    def in_waiting(self):
        with self._lock:
            return len(self._buffer)

    def reset_input_buffer(self):
        with self._lock:
            self._buffer.clear()

    def reset_output_buffer(self):
        pass

    def write(self, payload):
        self.writes.append(bytes(payload))
        for command in payload.decode("ascii").splitlines():
            if command == "hello":
                self._enqueue("Hello HC32F448 position control\r\n")
            elif command.startswith("motor run "):
                self._direction = 1 if " ccw " in command else -1
                self._running = True
                self._enqueue(
                    "Motor run ACTIVE mode=open_loop direction=ccw "
                    "duty=700 step_us=18000 lease_ms=1000\r\n"
                )
            elif command == "axis status":
                if self._running:
                    self._angle_mdeg += 50 * self._direction
                self._enqueue(
                    "axis target_mdeg=0 "
                    f"actual_mdeg={self._angle_mdeg} "
                    "error_mdeg=0 fault_active=0 target_valid=0 pid_out=0\r\n"
                )
            elif command == "motor off":
                self._running = False
                self._enqueue("Motor run STOPPED reason=command\r\n")
            elif command == "stop":
                self._running = False
                self._enqueue("Stop: hold\r\n")
        return len(payload)

    def flush(self):
        pass

    def readline(self):
        deadline = time.perf_counter() + self.timeout
        data = bytearray()
        while time.perf_counter() < deadline:
            chunk = self.read(1)
            if chunk:
                data.extend(chunk)
                if chunk == b"\n":
                    break
        return bytes(data)

    def read(self, size=1):
        deadline = time.perf_counter() + self.timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._buffer:
                    count = min(size, len(self._buffer))
                    data = self._buffer[:count]
                    del self._buffer[:count]
                    return bytes(data)
            time.sleep(0.001)
        return b""

    def close(self):
        self.is_open = False

    def _enqueue(self, text):
        with self._lock:
            self._buffer.extend(text.encode("ascii"))


class TurntableProtocolTest(unittest.TestCase):
    def test_converts_output_speed_and_builds_direction(self):
        self.assertAlmostEqual(output_dps_to_motor_rpm(1.2), 20.0)
        self.assertEqual(motor_speed_to_step_us(20.0), 15_000)
        self.assertEqual(
            build_motor_run_command(1.2),
            b"motor run ccw 700 15000\r\n",
        )
        self.assertEqual(
            build_motor_run_command(-1.2),
            b"motor run cw 700 15000\r\n",
        )
        with self.assertRaises(ValueError):
            output_dps_to_motor_rpm(0.2)
        with self.assertRaises(ValueError):
            output_dps_to_motor_rpm(4.0)

    def test_parses_continuous_encoder_angle_and_fault(self):
        sample = parse_axis_status(
            "axis target_mdeg=0 actual_mdeg=361234 "
            "error_mdeg=-1 fault_active=1 target_valid=0 pid_out=0",
            12.5,
        )
        self.assertEqual(sample, AxisSample(12.5, 361234, True))
        self.assertIsNone(parse_axis_status("unrelated", 1.0))

    def test_interpolates_frame_angle_across_continuous_samples(self):
        samples = [
            AxisSample(10.0, 359_900),
            AxisSample(10.2, 360_300),
        ]
        self.assertAlmostEqual(interpolate_mdeg(samples, 10.1), 360_100)
        self.assertAlmostEqual(measured_speed_dps(samples), 2.0)
        with self.assertRaises(ValueError):
            interpolate_mdeg(samples, 9.9)


class TurntableStateMachineTest(unittest.TestCase):
    def setUp(self):
        FakeSerial.instances.clear()

    @patch("src.turntable_motor.serial.Serial", FakeSerial)
    def test_connect_start_sample_interpolate_and_safe_close(self):
        motor = TurntableMotor("COM_TEST")
        motor.connect()
        motor.start(1.0)
        speed = motor.wait_until_moving(
            1.0,
            timeout_s=2.0,
            moving_duration_s=0.1,
        )
        self.assertGreater(speed, 0)

        timestamp = time.perf_counter()
        angle, span = motor.angle_mdeg_at_with_span(timestamp, timeout_s=0.5)
        self.assertGreater(angle, 0)
        self.assertGreater(span, 0)
        self.assertGreater(motor.telemetry_stats["sample_count"], 1)

        motor.close()
        written = b"".join(FakeSerial.instances[0].writes)
        self.assertIn(b"hello\r\n", written)
        self.assertIn(b"motor run ccw", written)
        self.assertIn(b"motor off\r\nstop\r\n", written)

    def test_wait_for_feedback_times_out(self):
        motor = TurntableMotor("COM_TEST")
        motor._serial = type("OpenSerial", (), {"is_open": True})()
        with self.assertRaises(TurntableError):
            motor.wait_for_samples(1, timeout_s=0.01)

    def test_requires_repeated_run_rejections_before_failing(self):
        motor = TurntableMotor("COM_TEST")
        motor._serial = type("OpenSerial", (), {"is_open": True})()
        rejected = (
            "Motor run REJECTED reason=range duty=700 step_us=1800"
        )
        motor._handle_line(rejected, time.perf_counter())
        motor._handle_line(rejected, time.perf_counter())
        motor._raise_if_unavailable()
        motor._handle_line(
            "Motor run ACTIVE mode=open_loop direction=ccw "
            "duty=700 step_us=18000 lease_ms=1000",
            time.perf_counter(),
        )
        motor._handle_line(rejected, time.perf_counter())
        motor._raise_if_unavailable()
        motor._handle_line(rejected, time.perf_counter())
        motor._handle_line(rejected, time.perf_counter())
        with self.assertRaises(TurntableError):
            motor._raise_if_unavailable()


if __name__ == "__main__":
    unittest.main()
