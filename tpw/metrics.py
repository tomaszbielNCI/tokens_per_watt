# tpw/metrics.py
import pandas as pd

MWATT = 1e6


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(bool)
    return df


def per_cell(df: pd.DataFrame, idle_w: float = 0.0) -> pd.DataFrame:
    """Aggregate the four metrics that the report compares. Passing
    idle_w subtracts the measured idle draw; report both variants."""
    d = df.copy()
    if idle_w:
        d["energy_j"] = d["energy_j"] - idle_w * d["wall_s"]

    grouped = d.groupby(["model", "mode", "difficulty"])
    out = grouped.agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        format_ok=("format_ok", "mean"),
        truncated=("truncated", "mean"),
        tokens=("output_tokens", "sum"),
        energy_j=("energy_j", "sum"),
        wall_s=("wall_s", "sum"),
        n_correct=("correct", "sum"),
    ).reset_index()

    out["j_per_token"] = out["energy_j"] / out["tokens"]
    out["j_per_correct"] = out["energy_j"] / out["n_correct"].replace(0, pd.NA)
    out["tokens_per_correct"] = out["tokens"] / out["n_correct"].replace(0, pd.NA)
    out["tps"] = out["tokens"] / out["wall_s"]
    out["tps_per_mw"] = out["tps"] / (out["energy_j"] / out["wall_s"]) * MWATT
    out["usd_per_task"] = out["wall_s"] / 3600 * 3.0 / out["n"]
    return out


def rank_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Rank models under each metric. Divergence between columns is the
    finding; agreement would be the null result."""
    metrics = ["j_per_token", "j_per_correct", "usd_per_task", "tps_per_mw"]
    ranks = cells.groupby("model")[metrics].mean()
    ascending = {"tps_per_mw": False}
    return ranks.rank(ascending=[ascending.get(m, True) for m in metrics])