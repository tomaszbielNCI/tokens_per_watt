import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpw.power import PowerSampler


def main(seconds: int = 300) -> None:
    with PowerSampler() as sampler:
        time.sleep(seconds)
    trace = sampler.trace()
    if trace is None:
        print("no power reading")
        return
    line = (f"idle_seconds\t{seconds}\n"
            f"idle_energy_j\t{trace.energy_j:.1f}\n"
            f"idle_mean_w\t{trace.mean_power_w:.2f}\n"
            f"idle_peak_w\t{trace.peak_power_w:.2f}\n")
    print(line)
    (ROOT / "results" / "idle.txt").write_text(line, encoding="utf-8")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))