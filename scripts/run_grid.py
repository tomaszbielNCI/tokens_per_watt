import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpw.runner import build_grid, run_grid
from tpw.tasks import load_gsm8k


def main(config_path: str | None = None) -> None:
    cfg_path = Path(config_path) if config_path else ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    tasks = load_gsm8k(cfg["n_per_difficulty"])
    cells = build_grid(cfg["models"], cfg["modes"], tasks, cfg["repeats"])
    run_grid(cells, tasks, ROOT / cfg["output"], **cfg["generation"])


if __name__ == "__main__":
    main(*sys.argv[1:])