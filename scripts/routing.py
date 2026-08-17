# scripts/routing.py
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpw.metrics import balanced, load, per_cell


def main(threshold: float = 0.80, idle_w: float = 52.1) -> None:
    """Compare a difficulty-conditioned routing policy against the best
    fixed model, subject to an accuracy floor. Both policies are
    evaluated on the same runs, so the saving is a reallocation of
    existing measurements rather than an extrapolation."""
    cells = per_cell(balanced(load(ROOT / "results/grid.csv")), idle_w=idle_w)
    cot = cells[cells["mode"] == "cot"].copy()

    eligible = cot[cot["accuracy"] >= threshold]
    routed = eligible.loc[eligible.groupby("difficulty")["j_per_correct"].idxmin()]

    print(f"accuracy floor: {threshold}\n")
    print("ROUTED POLICY")
    print(routed[["difficulty", "model", "accuracy", "j_per_correct"]]
          .round(3).to_string(index=False))
    routed_cost = routed["j_per_correct"].sum()

    print("\nFIXED-MODEL POLICIES (models meeting the floor on every difficulty)")
    per_model = cot.pivot_table(index="model", columns="difficulty",
                                values="accuracy", aggfunc="min")
    qualified = per_model[(per_model >= threshold).all(axis=1)].index
    for model in qualified:
        cost = cot[cot["model"] == model]["j_per_correct"].sum()
        print(f"  {model:20s} {cost:9.1f} J")

    if len(qualified):
        best_fixed = min(
            cot[cot["model"] == m]["j_per_correct"].sum() for m in qualified
        )
        saving = 1 - routed_cost / best_fixed
        print(f"\nrouted {routed_cost:.1f} J vs best fixed {best_fixed:.1f} J "
              f"— saving {saving:.1%}")
    else:
        print(f"\nno single model meets the floor everywhere; "
              f"routing costs {routed_cost:.1f} J")


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:]))
