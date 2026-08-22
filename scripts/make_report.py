"""Aggregate results: paper-style tables (2-4, 5-6, 7-9, 10) and equity plots."""
from __future__ import annotations
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config, data, ensemble, stats_tests

MODEL_LABELS = {"arima": "ARIMA", "lstm": "LSTM", "hybrid": "LSTM-ARIMA"}
PAPER_IR2 = {  # paper Tables 2-4 targets for quick comparison
    ("^GSPC", "long_only"): {"buy_hold": 5.09, "arima": 0.53, "lstm": 1.94, "hybrid": 5.79},
    ("^GSPC", "long_short"): {"buy_hold": 5.09, "arima": 7.13, "lstm": 3.87, "hybrid": 7.18},
    ("^FTSE", "long_only"): {"buy_hold": 0.66, "arima": -1.91, "lstm": 1.44, "hybrid": 7.19},
    ("^FTSE", "long_short"): {"buy_hold": 0.66, "arima": 0.07, "lstm": 0.67, "hybrid": 16.65},
    ("^FCHI", "long_only"): {"buy_hold": 0.98, "arima": -1.93, "lstm": 1.43, "hybrid": 3.04},
    ("^FCHI", "long_short"): {"buy_hold": 0.98, "arima": -0.21, "lstm": 0.97, "hybrid": 14.29},
}


def load_metrics(index: str, model: str, variant: str):
    path = config.RESULTS_DIR / index / f"{model}_{variant}" / "metrics.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


METRIC_COLS = ("ARC(%)", "ASD(%)", "MD(%)", "MLD", "IR*(%)", "IR**(%)")
MODEL_KEYS = {"LSTM-ARIMA": "hybrid", "ARIMA": "arima", "LSTM": "lstm"}


def base_tables(args) -> pd.DataFrame:
    """Tables 2-4 style: each run's metrics.csv holds Long Only / Long Short /
    Buy&Hold rows. B&H is identical across models, so it is emitted once per
    index rather than once per model."""
    rows = []
    for index in config.INDICES:
        buy_hold = None
        for model in args.models:
            df = load_metrics(index, model, args.variant)
            if df is None:
                continue
            for _, r in df.iterrows():
                mode = {"Long Only": "long_only", "Long Short": "long_short",
                        "Buy&Hold": "buy_hold"}[r["strategy"]]
                row = {"index": config.INDICES[index]["name"],
                       "strategy": r["strategy"], "model": r["model"],
                       **{c: r[c] for c in METRIC_COLS}}
                if mode == "buy_hold":
                    # PAPER_IR2 is keyed by strategy mode only; the benchmark
                    # value lives under either mode's "buy_hold" entry.
                    row["paper_IR**(%)"] = PAPER_IR2.get(
                        (index, "long_only"), {}).get("buy_hold")
                    # dedupe: keep the first, do not re-emit per model
                    buy_hold = buy_hold or row
                    continue
                row["paper_IR**(%)"] = PAPER_IR2.get((index, mode), {}).get(
                    MODEL_KEYS.get(r["model"], r["model"]))
                rows.append(row)
        if buy_hold is not None:
            rows.append(buy_hold)
    return pd.DataFrame(rows)


def significance_tables(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paper Sec. 5.2 tests. Returns are aligned on EARNING days: predictions
    are decision-day indexed, returns_{mode}.csv earning-day indexed, so the
    benchmark must be taken on the returns' own index (a decision-day reindex
    would lag the benchmark by one day)."""
    t_rows, reg_rows = [], []
    for index in config.INDICES:
        feats = data.build_features(index)
        mkt_ret_all = data.market_returns(feats["close"])
        for model in args.models:
            run = config.RESULTS_DIR / index / f"{model}_{args.variant}"
            if not (run / "metrics.csv").exists():
                continue
            for mode in ("long_only", "long_short"):
                path = run / f"returns_{mode}.csv"
                if not path.exists():
                    continue
                rets = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
                mkt = mkt_ret_all.reindex(rets.index)
                tt = stats_tests.paired_ttest(rets, mkt)
                t_rows.append({"index": config.INDICES[index]["name"],
                               "strategy": mode, "model": MODEL_LABELS[model],
                               **tt})
                rg = stats_tests.regression_alpha(rets, mkt)
                reg_rows.append({"index": config.INDICES[index]["name"],
                                 "strategy": mode, "model": MODEL_LABELS[model],
                                 **rg})
    return pd.DataFrame(t_rows), pd.DataFrame(reg_rows)


def plot_equity_curves(args) -> None:
    for index in config.INDICES:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        found = False
        for ax, mode in zip(axes, ("long_only", "long_short")):
            for model in args.models:
                run = config.RESULTS_DIR / index / f"{model}_{args.variant}"
                path = run / f"equity_{mode}.csv"
                if path.exists():
                    eq = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
                    ax.plot(eq.index, eq.values, label=MODEL_LABELS[model],
                            linewidth=1.2)
                    found = True
            try:
                bh = ensemble.load_buy_hold(index, args.models, args.variant)
                ax.plot(bh.index, bh.values, label="Buy&Hold",
                        color="black", linewidth=1.2)
                found = True
            except FileNotFoundError:
                pass
            ax.set_title(f"{config.INDICES[index]['name']} - "
                         f"{'Long-Only' if mode == 'long_only' else 'Long-Short'}")
            ax.legend(fontsize=8)
            ax.set_ylabel("Equity")
        if found:
            fig.tight_layout()
            out = config.RESULTS_DIR / index / f"equity_curves_{args.variant}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["arima", "lstm", "hybrid"])
    p.add_argument("--variant", default="base")
    args = p.parse_args()

    out_dir = config.RESULTS_DIR
    base = base_tables(args)
    if not base.empty:
        base.round(4).to_csv(out_dir / "table_base_case.csv", index=False)
        print("\n=== Base case (Tables 2-4 style) ===")
        print(base.round(4).to_string(index=False))

    tt, rg = significance_tables(args)
    if not tt.empty:
        tt.to_csv(out_dir / "table5_paired_ttest.csv", index=False)
        rg.to_csv(out_dir / "table6_regression.csv", index=False)
        print("\n=== Paired t-test (Table 5 style) ===")
        print(tt.round(4).to_string(index=False))
        print("\n=== Regression alpha (Table 6 style) ===")
        print(rg.round(4).to_string(index=False))

    plot_equity_curves(args)


if __name__ == "__main__":
    main()
