import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpw.runner import build_grid, run_grid
from tpw.tasks import load_gsm8k


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    tasks = load_gsm8k(cfg["n_per_difficulty"])
    cells = build_grid(cfg["models"], cfg["modes"], tasks, cfg["repeats"])
    run_grid(cells, tasks, cfg["output"], **cfg["generation"])


if __name__ == "__main__":
    main(*sys.argv[1:])