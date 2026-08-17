import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(csv_path: str = "results/test.csv", n: int = 10) -> None:
    df = pd.read_csv(ROOT / csv_path)

    print("format compliance and truncation by model\n")
    print(df.groupby(["model", "mode"]).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        format_ok=("format_ok", "mean"),
        truncated=("truncated", "mean"),
    ).round(3).to_string())

    print("\n\nwrong answers — inspect for parser failures vs real errors\n")
    wrong = df[~df["correct"].astype(bool)]
    for _, row in wrong.head(n).iterrows():
        print(f"--- {row['model']} / {row['mode']} / {row['difficulty']}")
        print(f"expected={row['expected']}  parsed={row['parsed']}")
        print(str(row["raw_output"])[-400:].replace("\n", " "))
        print()


if __name__ == "__main__":
    main(*sys.argv[1:])
