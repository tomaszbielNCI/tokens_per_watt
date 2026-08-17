import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpw.metrics import load, per_cell, rank_table, balanced

pd.set_option("display.width", 200)


def read_idle() -> float:
    """Idle draw in watts, from scripts/idle_baseline.py output."""
    path = ROOT / "results" / "idle.txt"
    if not path.exists():
        print("no idle.txt — reporting raw energy only\n")
        return 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("idle_mean_w"):
            return float(line.split("\t")[1])
    return 0.0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the accuracy uncertainty to report."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main(csv_path: str = "results/grid.csv") -> None:
    df = load(ROOT / csv_path)
    idle_w = read_idle()
    print(f"rows: {len(df)}   idle: {idle_w:.1f} W\n")

    # Balance the panel to equal task_ids across models
    rows_before = df.groupby("model").size()
    df = balanced(df)
    rows_after = df.groupby("model").size()
    rows_dropped = rows_before.sum() - rows_after.sum()
    print(f"BALANCED PANEL: dropped {rows_dropped} rows to equalize task_ids across models")
    print("Per-model row counts:")
    print(pd.DataFrame({"before": rows_before, "after": rows_after}).to_string())
    print()

    print("=" * 100)
    print("PLACEMENT AND TRUNCATION CHECK — any truncation or CPU split invalidates those rows")
    print("=" * 100)
    print(df.groupby("model").agg(
        n=("correct", "size"),
        truncated=("truncated", "mean"),
        format_ok=("format_ok", "mean"),
    ).round(3).to_string())

    print(f"\n{'=' * 100}")
    print("TRUNCATION DETAIL — count and rate of truncated=True by model and mode")
    print("=" * 100)
    trunc_detail = df.groupby(["model", "mode"]).agg(
        n=("correct", "size"),
        truncated_count=("truncated", "sum"),
    )
    trunc_detail["truncated_rate"] = (trunc_detail["truncated_count"] / trunc_detail["n"]).round(3)
    print(trunc_detail.to_string())

    for label, idle in (("RAW", 0.0), ("IDLE-SUBTRACTED", idle_w)):
        cells = per_cell(df, idle_w=idle)
        print(f"\n{'=' * 100}\nPER CELL — {label}\n{'=' * 100}")
        cols = ["model", "mode", "difficulty", "n", "accuracy", "tokens",
                "energy_j", "j_per_token", "j_per_correct",
                "tokens_per_correct", "tps", "tps_per_mw", "usd_per_task"]
        print(cells[cols].round(3).to_string(index=False))

        print(f"\nRANKINGS — {label}  (1 = best under that metric)")
        print(rank_table(cells).round(1).to_string())

    print(f"\n{'=' * 100}\nACCURACY WITH WILSON INTERVALS\n{'=' * 100}")
    grouped = df.groupby(["model", "mode", "difficulty"])["correct"]
    rows = []
    for key, series in grouped:
        lo, hi = wilson(int(series.sum()), len(series))
        rows.append({"model": key[0], "mode": key[1], "difficulty": key[2],
                     "n": len(series), "accuracy": round(series.mean(), 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n{'=' * 100}\nVERBALISATION OVERHEAD — mean output tokens by mode\n{'=' * 100}")
    pivot = df.pivot_table(index="model", columns="mode",
                           values="output_tokens", aggfunc="mean").round(1)
    pivot["ratio_cot_direct"] = (pivot["cot"] / pivot["direct"]).round(2)
    print(pivot.to_string())

    print(f"\n{'=' * 100}\nMoE VS DENSE AT EQUAL TOTAL SIZE\n{'=' * 100}")
    pair = per_cell(df, idle_w=idle_w)
    pair = pair[pair["model"].isin(["qwen2.5:3b", "granite3-moe:3b"])]
    print(pair[["model", "mode", "difficulty", "accuracy", "tokens",
                "energy_j", "j_per_token", "j_per_correct"]].round(3).to_string(index=False))

    out = ROOT / "results" / "per_cell.csv"
    per_cell(df, idle_w=idle_w).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])