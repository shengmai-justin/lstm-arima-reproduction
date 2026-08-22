"""Aggregate results: paper-style tables (2-4, 5-6, 7-9, 10) and equity plots."""
from __future__ import annotations
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


import argparse
import pathlib

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


# Paper Tables 7-9, IR**(%) per (index, model, strategy) -> variant.
# The S&P 500 ARIMA base cells are the paper's own Table 2 values, not Table 7's,
# because Table 7's S&P block duplicates the LSTM-ARIMA row (see README).
PAPER_SENSITIVITY = {
    ("^GSPC", "arima", "long_only"):  {"base": 0.53, "arima_orders_0_3": 0.91, "arima_bic": 0.73},
    ("^GSPC", "arima", "long_short"): {"base": 7.13, "arima_orders_0_3": 7.62, "arima_bic": 7.86},
    ("^FTSE", "arima", "long_only"):  {"base": -1.91, "arima_orders_0_3": -1.26, "arima_bic": -1.23},
    ("^FTSE", "arima", "long_short"): {"base": 0.07, "arima_orders_0_3": 0.63, "arima_bic": 0.58},
    ("^FCHI", "arima", "long_only"):  {"base": -1.93, "arima_orders_0_3": -2.86, "arima_bic": -2.35},
    ("^FCHI", "arima", "long_short"): {"base": -0.21, "arima_orders_0_3": -1.06, "arima_bic": -0.53},

    ("^GSPC", "lstm", "long_only"):  {"base": 1.94, "dropout_0.05": 0.13, "dropout_0.10": 2.13,
                                      "batch_16": 11.46, "batch_64": 1.39},
    ("^GSPC", "lstm", "long_short"): {"base": 3.87, "dropout_0.05": 2.13, "dropout_0.10": 15.57,
                                      "batch_16": 0.04, "batch_64": 0.49},
    ("^FTSE", "lstm", "long_only"):  {"base": 1.44, "dropout_0.05": 2.26, "dropout_0.10": 2.12,
                                      "batch_16": 0.78, "batch_64": -0.04},
    ("^FTSE", "lstm", "long_short"): {"base": 0.67, "dropout_0.05": 2.15, "dropout_0.10": 4.39,
                                      "batch_16": 5.40, "batch_64": 0.46},
    ("^FCHI", "lstm", "long_only"):  {"base": 1.43, "dropout_0.05": 5.38, "dropout_0.10": 1.20,
                                      "batch_16": 2.25, "batch_64": 1.99},
    ("^FCHI", "lstm", "long_short"): {"base": 0.97, "dropout_0.05": -0.13, "dropout_0.10": 0.22,
                                      "batch_16": -0.19, "batch_64": 1.43},

    ("^GSPC", "hybrid", "long_only"):  {"base": 5.79, "dropout_0.05": 12.99, "dropout_0.10": 4.35,
                                        "batch_16": 9.11, "batch_64": 0.18},
    ("^GSPC", "hybrid", "long_short"): {"base": 7.18, "dropout_0.05": 0.49, "dropout_0.10": -0.44,
                                        "batch_16": -2.79, "batch_64": 0.90},
    ("^FTSE", "hybrid", "long_only"):  {"base": 7.19, "dropout_0.05": 1.55, "dropout_0.10": 2.35,
                                        "batch_16": 1.68, "batch_64": 0.03},
    ("^FTSE", "hybrid", "long_short"): {"base": 16.65, "dropout_0.05": 3.01, "dropout_0.10": 1.40,
                                        "batch_16": -0.04, "batch_64": 1.96},
    ("^FCHI", "hybrid", "long_only"):  {"base": 3.04, "dropout_0.05": 1.28, "dropout_0.10": 0.01,
                                        "batch_16": -0.04, "batch_64": -0.13},
    ("^FCHI", "hybrid", "long_short"): {"base": 14.29, "dropout_0.05": 10.60, "dropout_0.10": 3.84,
                                        "batch_16": 0.56, "batch_64": -0.06},
}

ARIMA_VARIANTS = ["base", "arima_orders_0_3", "arima_bic"]
NEURAL_VARIANTS = ["base", "dropout_0.05", "dropout_0.10", "batch_16", "batch_64"]


def load_metrics(index: str, model: str, variant: str):
    path = config.RESULTS_DIR / index / f"{model}_{variant}" / "metrics.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


METRIC_COLS = ("ARC(%)", "ASD(%)", "MD(%)", "MLD", "IR*(%)", "IR**(%)")
# run_experiment writes the model label as args.model.upper(), so the label in
# metrics.csv is HYBRID rather than the paper's LSTM-ARIMA. Accept both.
MODEL_KEYS = {"HYBRID": "hybrid", "LSTM-ARIMA": "hybrid",
              "ARIMA": "arima", "LSTM": "lstm"}


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


def sensitivity_tables(args) -> pd.DataFrame:
    """Tables 7-9 style: every variant of every model, against the paper."""
    rows = []
    for index in config.INDICES:
        for model in args.models:
            variants = ARIMA_VARIANTS if model == "arima" else NEURAL_VARIANTS
            for variant in variants:
                df = load_metrics(index, model, variant)
                if df is None:
                    continue
                for _, r in df.iterrows():
                    if r["strategy"] == "Buy&Hold":
                        continue
                    mode = {"Long Only": "long_only",
                            "Long Short": "long_short"}[r["strategy"]]
                    paper = PAPER_SENSITIVITY.get(
                        (index, model, mode), {}).get(variant)
                    rows.append({
                        "index": config.INDICES[index]["name"],
                        "model": MODEL_LABELS[model],
                        "strategy": r["strategy"], "variant": variant,
                        **{c: r[c] for c in METRIC_COLS},
                        "paper_IR**(%)": paper,
                        "diff": (None if paper is None
                                 else round(r["IR**(%)"] - paper, 2)),
                    })
    return pd.DataFrame(rows)


def saturation_report(args) -> pd.DataFrame:
    """How often the tanh output ceiling, not the network, set the position.

    The target scaler is fit on each walk's training window, so an inverse-
    transformed forecast can never exceed that window's maximum close. Any OOS
    day already above it is mechanically short under the Sec. 4.8 rule.
    """
    from src import wfo

    rows = []
    for index in config.INDICES:
        closes = data.build_features(index)["close"].to_numpy()
        above, total, pinned_walks, n_walks = 0, 0, 0, 0
        for _, is_start, oos_start, oos_end in wfo._oos_slices(len(closes)):
            train_max = closes[is_start:is_start + config.TRAIN_DAYS].max()
            oos = closes[oos_start:oos_end]
            n = int((oos > train_max).sum())
            above += n
            total += len(oos)
            n_walks += 1
            pinned_walks += int(n == len(oos))
        row = {"index": config.INDICES[index]["name"],
               "OOS days above train max": f"{above}/{total}",
               "share": round(above / total, 4),
               "fully pinned walks": f"{pinned_walks}/{n_walks}"}

        # how often the realised signal never changed, per model
        for model in args.models:
            if model == "arima":
                continue
            path = (config.RESULTS_DIR / index / f"{model}_{args.variant}"
                    / "returns_long_only.csv")
            if not path.exists():
                continue
            r = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            # long_only earns 0 exactly when flat; a permanently-short model
            # under the level target shows up as an all-zero long_only series
            row[f"{MODEL_LABELS[model]} days flat"] = round((r == 0).mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


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
    p.add_argument("--results-dir", default=None,
                   help="read from a directory other than results/")
    args = p.parse_args()
    if args.results_dir:
        config.RESULTS_DIR = pathlib.Path(args.results_dir).resolve()

    out_dir = config.RESULTS_DIR
    base = base_tables(args)
    if not base.empty:
        base.round(4).to_csv(out_dir / "table_base_case.csv", index=False)
        print("\n=== Base case (Tables 2-4 style) ===")
        print(base.round(4).to_string(index=False))

    sens = sensitivity_tables(args)
    if not sens.empty:
        sens.round(4).to_csv(out_dir / "table_sensitivity.csv", index=False)
        print("\n=== Sensitivity (Tables 7-9 style) ===")
        print(sens.round(3).to_string(index=False))

    sat = saturation_report(args)
    if not sat.empty:
        sat.to_csv(out_dir / "table_saturation.csv", index=False)
        print("\n=== tanh output-ceiling saturation ===")
        print(sat.to_string(index=False))

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
