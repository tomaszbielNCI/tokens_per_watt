# tpw/metrics.py
import pandas as pd

MWATT = 1e6


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(bool)
    return df


def balanced(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to task_ids attempted by every model, so cell counts
    are equal. An earlier partial run used a smaller task sample and
    the resume key does not include sample size, leaving
    qwen2.5:0.5b with more observations than the other models."""
    task_ids_by_model = df.groupby("model")["task_id"].apply(set)
    common_task_ids = set.intersection(*task_ids_by_model)
    return df[df["task_id"].isin(common_task_ids)].copy()


def per_cell(df: pd.DataFrame, idle_w: float = 0.0) -> pd.DataFrame:
    """Aggregate the four metrics that the report compares. Passing
    idle_w subtracts the measured idle draw; report both variants.
    tps_per_mw is the reciprocal of j_per_token (tokens/energy in different
    units) and is kept for plotting against NVIDIA's published axes."""
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
    metrics = ["j_per_token", "j_per_correct", "tokens_per_correct", "usd_per_task"]
    ranks = cells.groupby("model")[metrics].mean()
    return ranks.rank(ascending=True)