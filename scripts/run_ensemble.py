"""Build the Sec. 7 ensemble AIS results from per-index equity curves."""
from __future__ import annotations
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


import argparse

import pandas as pd

from src import config, data, ensemble, metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["arima", "lstm", "hybrid"])
    p.add_argument("--variant", default="base")
    args = p.parse_args()

    out_dir = config.RESULTS_DIR / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Buy&Hold does not depend on the strategy mode. Building it inside the mode
    # loop produced two byte-identical rows mislabeled "Long Only"/"Long Short"
    # and two identical CSVs; paper Table 10 has a single benchmark row.
    bh = ensemble.ensemble_equity(
        {idx: ensemble.load_buy_hold(idx, args.models, args.variant)
         for idx in config.INDICES})
    bh.to_frame("equity").to_csv(out_dir / "equity_buy_hold.csv")
    rows.append({"strategy": "Buy&Hold", "model": "buy_hold",
                 **metrics.compute_all(bh.pct_change().fillna(0.0), bh)})

    for mode in ("long_only", "long_short"):
        per_model = {}
        for model in args.models:
            try:
                eqs = {idx: ensemble.load_equity_curve(idx, model, mode, args.variant)
                       for idx in config.INDICES}
            except FileNotFoundError as exc:
                print(f"skipping {model} ({mode}): {exc}")
                continue
            per_model[model] = ensemble.ensemble_equity(eqs)

        for model, eq in per_model.items():
            eq.to_frame("equity").to_csv(out_dir / f"equity_{model}_{mode}.csv")
            rets = eq.pct_change().fillna(0.0)
            rows.append({"strategy": {"long_only": "Long Only",
                                      "long_short": "Long Short"}[mode],
                         "model": model, **metrics.compute_all(rets, eq)})

    df = pd.DataFrame(rows).round(4)
    df.to_csv(out_dir / "metrics.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
