# tpw/power.py
"""GPU energy measurement.

Prefers NVML's buffered power-sample stream, which the driver populates
at roughly 50 Hz and which remains available on consumer GPUs where both
the instantaneous power query and the cumulative energy counter return
NVML_ERROR_NOT_SUPPORTED. Falls back to the energy counter, then to
polled instantaneous power, then to a no-op so the pipeline runs
unchanged on machines without a usable GPU.
"""
import threading
import time

from tpw.contracts import PowerTrace

try:
    import pynvml
except ImportError:
    pynvml = None

MODE_SAMPLES = "nvml_power_samples"
MODE_COUNTER = "nvml_energy_counter"
MODE_POLLED = "nvml_polled_power"
MODE_NONE = "unavailable"


def _handle():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def _drain(handle, since_us: int) -> tuple[list[tuple[int, float]], int]:
    """Pull buffered power samples newer than since_us.

    Returns (samples, newest_timestamp) with timestamps in microseconds
    and power in watts. NVML reports power samples as unsigned
    milliwatts; the union field depends on the value type it declares.
    """
    val_type, raw = pynvml.nvmlDeviceGetSamples(
        handle, pynvml.NVML_TOTAL_POWER_SAMPLES, since_us
    )
    out = []
    newest = since_us
    for sample in raw:
        if val_type == pynvml.NVML_VALUE_TYPE_DOUBLE:
            milliwatts = sample.sampleValue.dVal
        elif val_type == pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
            milliwatts = sample.sampleValue.ullVal
        else:
            milliwatts = sample.sampleValue.uiVal
        out.append((sample.timeStamp, milliwatts / 1000.0))
        newest = max(newest, sample.timeStamp)
    return out, newest


def detect_mode() -> tuple[str, str]:
    """Return (mode, human-readable detail) for the best available method."""
    if pynvml is None:
        return MODE_NONE, "nvidia-ml-py not installed"
    try:
        handle = _handle()
        name = pynvml.nvmlDeviceGetName(handle)
    except Exception as exc:
        return MODE_NONE, f"nvml init failed: {exc}"

    try:
        samples, _ = _drain(handle, 0)
        if samples:
            return MODE_SAMPLES, f"{name}: {len(samples)} buffered power samples"
    except Exception:
        pass
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return MODE_COUNTER, f"{name}: cumulative energy counter"
    except Exception:
        pass
    try:
        pynvml.nvmlDeviceGetPowerUsage(handle)
        return MODE_POLLED, f"{name}: instantaneous power only"
    except Exception as exc:
        return MODE_NONE, f"{name}: no power telemetry ({type(exc).__name__})"


def diagnose() -> str:
    mode, detail = detect_mode()
    return f"{mode} — {detail}"


def _temperature(handle) -> float:
    try:
        return float(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )
    except Exception:
        return float("nan")


def _integrate(points: list[tuple[float, float]]) -> float:
    """Trapezoid rule over (seconds, watts) pairs."""
    return sum(
        (points[i][1] + points[i + 1][1]) / 2 * (points[i + 1][0] - points[i][0])
        for i in range(len(points) - 1)
    )


class PowerSampler:
    """Context manager measuring GPU energy over the enclosed block."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.mode, self.detail = detect_mode()
        self.enabled = self.mode != MODE_NONE
        self._handle = _handle() if self.enabled else None
        self._points: list[tuple[float, float]] = []
        self._since_us = 0
        self._start_mj: int | None = None
        self._end_mj: int | None = None
        self._temps: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._wall = 0.0

    def __enter__(self) -> "PowerSampler":
        if not self.enabled:
            return self
        self._temps.append(_temperature(self._handle))
        if self.mode == MODE_SAMPLES:
            _, self._since_us = _drain(self._handle, 0)  # discard stale buffer
        elif self.mode == MODE_COUNTER:
            self._start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        self._wall = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self.enabled:
            return
        self._wall = time.monotonic() - self._wall
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self.mode == MODE_COUNTER:
            self._end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        elif self.mode == MODE_SAMPLES:
            self._collect()
        self._temps.append(_temperature(self._handle))

    def _collect(self) -> None:
        """Drain the driver's buffer; timestamps convert to seconds."""
        samples, self._since_us = _drain(self._handle, self._since_us)
        self._points.extend((ts / 1e6, watts) for ts, watts in samples)

    def _loop(self) -> None:
        """Poll often enough that the driver's finite buffer does not
        overflow — it holds a few seconds of samples."""
        while not self._stop.is_set():
            try:
                if self.mode == MODE_SAMPLES:
                    self._collect()
                elif self.mode == MODE_POLLED:
                    watts = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    self._points.append((time.monotonic(), watts))
                else:
                    watts = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    self._points.append((time.monotonic(), watts))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def trace(self) -> PowerTrace | None:
        if not self.enabled:
            return None

        if self.mode == MODE_COUNTER and self._start_mj is not None:
            energy_j = (self._end_mj - self._start_mj) / 1000.0
        elif len(self._points) >= 2:
            energy_j = _integrate(sorted(self._points))
        else:
            return None

        watts = [w for _, w in self._points]
        return PowerTrace(
            energy_j=energy_j,
            mean_power_w=energy_j / self._wall if self._wall else float("nan"),
            peak_power_w=max(watts) if watts else float("nan"),
            temp_start_c=self._temps[0],
            temp_end_c=self._temps[-1],
            n_samples=len(self._points),
        )