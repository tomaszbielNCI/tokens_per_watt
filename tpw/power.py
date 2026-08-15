# tpw/power.py
import subprocess
import threading
import time

from tpw.contracts import PowerTrace

_FIELDS = "power.draw,temperature.gpu"
_CMD = ["nvidia-smi", f"--query-gpu={_FIELDS}", "--format=csv,noheader,nounits"]


def gpu_available() -> bool:
    """True if nvidia-smi reports a numeric power reading."""
    try:
        _read()
    except Exception:
        return False
    return True


def _read() -> tuple[float, float]:
    out = subprocess.check_output(_CMD, text=True, timeout=5)
    power, temp = out.strip().splitlines()[0].split(",")
    return float(power), float(temp)


class PowerSampler:
    """Context manager sampling GPU power; integrates energy by trapezoid rule.

    Falls back to a no-op when no GPU is present, so the same code runs
    on a CPU-only laptop during development.
    """

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.enabled = gpu_available()
        self._samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PowerSampler":
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                power, temp = _read()
                self._samples.append((time.monotonic(), power, temp))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def trace(self) -> PowerTrace | None:
        if len(self._samples) < 2:
            return None
        ts = [s[0] for s in self._samples]
        ws = [s[1] for s in self._samples]
        energy = sum(
            (ws[i] + ws[i + 1]) / 2 * (ts[i + 1] - ts[i])
            for i in range(len(ts) - 1)
        )
        return PowerTrace(
            energy_j=energy,
            mean_power_w=sum(ws) / len(ws),
            peak_power_w=max(ws),
            temp_start_c=self._samples[0][2],
            temp_end_c=self._samples[-1][2],
            n_samples=len(self._samples),
        )
