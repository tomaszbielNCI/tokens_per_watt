# tpw/tasks.py
import re
from dataclasses import dataclass

_STEP_MARKER = re.compile(r"<<.+?>>")


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    answer: str
    difficulty: str
    source: str


def _gsm8k_difficulty(solution: str) -> str:
    """Bucket by reasoning depth, proxied by the number of calculator
    annotations in the reference solution."""
    steps = len(_STEP_MARKER.findall(solution))
    if steps <= 2:
        return "easy"
    return "medium" if steps <= 4 else "hard"


def _gsm8k_answer(solution: str) -> str:
    return solution.split("####")[-1].strip().replace(",", "")


def load_gsm8k(n_per_difficulty: int = 60, seed: int = 0) -> list[Task]:
    """Stratified sample of the GSM8K test split."""
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    buckets: dict[str, list[Task]] = {"easy": [], "medium": [], "hard": []}

    for i, row in enumerate(rows):
        level = _gsm8k_difficulty(row["answer"])
        if len(buckets[level]) >= n_per_difficulty:
            continue
        buckets[level].append(
            Task(
                task_id=f"gsm8k-{i}",
                question=row["question"],
                answer=_gsm8k_answer(row["answer"]),
                difficulty=level,
                source="gsm8k",
            )
        )
        if all(len(b) >= n_per_difficulty for b in buckets.values()):
            break

    return [task for level in ("easy", "medium", "hard") for task in buckets[level]]