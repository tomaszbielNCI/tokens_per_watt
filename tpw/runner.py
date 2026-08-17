# tpw/runner.py
import csv
import itertools
import random
from pathlib import Path

from tpw.contracts import Cell, RunResult
from tpw.grading import (
    PROMPT_COT,
    PROMPT_DIRECT,
    followed_format,
    is_correct,
    parse_answer,
    split_thinking,
)
from tpw.ollama import generate
from tpw.power import PowerSampler

_PROMPTS = {"direct": PROMPT_DIRECT, "cot": PROMPT_COT}

_FIELDS = [
    "model", "mode", "difficulty", "task_id", "repeat",
    "prompt_tokens", "output_tokens", "thinking_chars", "wall_s", "truncated",
    "energy_j", "mean_power_w", "peak_power_w", "temp_start_c", "temp_end_c",
    "parsed", "expected", "correct", "format_ok", "raw_output",
]


def build_grid(models, modes, tasks, repeats, seed=0) -> list[Cell]:
    """Full cross product in randomised order, so thermal drift is not
    confounded with model identity."""
    cells = [
        Cell(model=m, mode=mo, difficulty=t.difficulty, task_id=t.task_id, repeat=r)
        for m, mo, t, r in itertools.product(models, modes, tasks, range(repeats))
    ]
    random.Random(seed).shuffle(cells)
    return cells


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {
            (r["model"], r["mode"], r["task_id"], r["repeat"])
            for r in csv.DictReader(fh)
        }


def _row(result: RunResult, expected: str) -> dict:
    p = result.power
    return {
        "model": result.cell.model,
        "mode": result.cell.mode,
        "difficulty": result.cell.difficulty,
        "task_id": result.cell.task_id,
        "repeat": result.cell.repeat,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "thinking_chars": result.meta.get("thinking_chars", 0),
        "wall_s": round(result.wall_s, 4),
        "truncated": result.truncated,
        "energy_j": round(p.energy_j, 3) if p else "",
        "mean_power_w": round(p.mean_power_w, 2) if p else "",
        "peak_power_w": round(p.peak_power_w, 2) if p else "",
        "temp_start_c": p.temp_start_c if p else "",
        "temp_end_c": p.temp_end_c if p else "",
        "parsed": result.parsed,
        "expected": expected,
        "correct": result.correct,
        "format_ok": result.meta.get("format_ok"),
        "raw_output": result.raw_output,
    }


def run_one(cell: Cell, question: str, expected: str, **kwargs) -> RunResult:
    prompt = _PROMPTS[cell.mode].format(question=question)
    with PowerSampler() as sampler:
        payload = generate(cell.model, prompt, seed=cell.repeat, **kwargs)
    raw = payload["response"]
    thinking, _ = split_thinking(raw)
    parsed = parse_answer(raw)
    return RunResult(
        cell=cell,
        prompt_tokens=payload.get("prompt_eval_count", 0),
        output_tokens=payload.get("eval_count", 0),
        wall_s=payload["wall_s"],
        truncated=payload.get("done_reason") == "length",
        raw_output=raw,
        power=sampler.trace(),
        parsed=parsed,
        correct=is_correct(parsed, expected),
        meta={"thinking_chars": len(thinking), "format_ok": followed_format(raw)},
    )


def run_grid(cells, tasks, out_path, **kwargs) -> None:
    """Execute the grid, appending each row immediately and skipping
    anything already present, so an interrupted run can resume."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {t.task_id: t for t in tasks}
    done = _done(out_path)
    fresh = not out_path.exists()

    with out_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        if fresh:
            writer.writeheader()
        for i, cell in enumerate(cells, 1):
            key = (cell.model, cell.mode, cell.task_id, str(cell.repeat))
            if key in done:
                continue
            task = by_id[cell.task_id]
            result = run_one(cell, task.question, task.answer, **kwargs)
            writer.writerow(_row(result, task.answer))
            fh.flush()
            print(f"{i}/{len(cells)} {cell.model} {cell.mode} {cell.difficulty} "
                  f"correct={result.correct}")