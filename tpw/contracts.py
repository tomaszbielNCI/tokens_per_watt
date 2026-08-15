# tpw/contracts.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cell:
    """One point in the experimental grid."""
    model: str
    mode: str          # "cot" | "direct"
    difficulty: str    # "easy" | "medium" | "hard"
    task_id: str
    repeat: int


@dataclass
class PowerTrace:
    energy_j: float
    mean_power_w: float
    peak_power_w: float
    temp_start_c: float
    temp_end_c: float
    n_samples: int


@dataclass
class RunResult:
    cell: Cell
    prompt_tokens: int
    output_tokens: int
    wall_s: float
    truncated: bool
    raw_output: str
    power: PowerTrace | None = None
    parsed: str | None = None
    correct: bool | None = None
    meta: dict = field(default_factory=dict)