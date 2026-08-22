"""Run one (model x index x variant) experiment (building block for server runs).

Examples:
    python scripts/run_experiment.py --model arima --index ^GSPC
    python scripts/run_experiment.py --model hybrid --index ^FTSE --variant dropout_0.05
    python scripts/run_experiment.py --model lstm --index ^FCHI --max-walks 2 --trials 3 \
        --epochs 5          # light local smoke run

Outputs under results/<index>/<model>_<variant>/:
    predictions.csv, equity_long_only.csv, equity_long_short.csv,
    returns_long_only.csv, returns_long_short.csv, metrics.csv, walks.csv
"""
from __future__ import annotations
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


import argparse
import json

import pandas as pd

from src import backtest, config, data, metrics, wfo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["arima", "lstm", "hybrid"])
    p.add_argument("--index", required=True, choices=list(config.INDICES))
    p.add_argument("--variant", default="base", choices=list(config.SENSITIVITY_VARIANTS))
    p.add_argument("--trials", type=int, default=config.N_TRIALS,
                   help="random-search trials per walk")
    p.add_argument("--epochs", type=int, default=config.LSTM_MAX_EPOCHS,
                   help="max training epochs (LSTM only)")
    p.add_argument("--max-walks", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (LSTM only)")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel workers: ARIMA order search and (for CPU runs) "
                        "whole walks; keep 1 for single-GPU LSTM runs")
    p.add_argument("--amp", action="store_true",
                   help="mixed-precision training (CUDA only)")
    p.add_argument("--arima-forecast", dest="forecast_mode",
                   default=config.ARIMA_FORECAST_MODE,
                   choices=["rolling", "static"],
                   help="how the standalone ARIMA forecasts the OOS block "
                        "(gap G10 in the README); ignored by lstm/hybrid")
    p.add_argument("--target", dest="target_mode", default=config.LSTM_TARGET,
                   choices=["level", "return"],
                   help="what the LSTM/hybrid regresses on; 'return' avoids the "
                        "tanh price ceiling (a deliberate deviation from the "
                        "paper). Ignored by arima")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def evaluate_predictions(pred_df, features) -> dict:
    """Equity curves, net returns and metrics for both strategies + B&H.

    pred_df is indexed by decision day t with F(t) in `pred` and close P(t) in
    `close`; signals_from_forecast shifts to earning day t+1.
    """
    close = features["close"]
    mkt_all = backtest.buy_and_hold_returns(close)
    out = {"metrics": {}, "equity": {}, "returns": {}}
    for mode in ("long_only", "long_short"):
        sig = backtest.signals_from_forecast(pred_df["pred"], close, mode=mode)
        rets = backtest.strategy_returns(sig, mkt_all.reindex(sig.index))
        eq = backtest.equity_curve(rets)
        out["metrics"][mode] = metrics.compute_all(rets, eq)
        out["equity"][mode] = eq
        out["returns"][mode] = rets

    oos_ret = mkt_all[mkt_all.index.isin(
        backtest.signals_from_forecast(pred_df["pred"], close).index)]
    bh_eq = backtest.equity_curve(oos_ret)
    out["metrics"]["buy_hold"] = metrics.compute_all(oos_ret, bh_eq)
    out["equity"]["buy_hold"] = bh_eq
    return out


def main() -> None:
    args = parse_args()

    features = data.build_features(args.index)
    # max_epochs travels as an argument (not a config mutation) so that it is
    # honoured inside joblib workers when --jobs > 1.
    out = wfo.run_wfo(features, args.model, variant=args.variant,
                      n_trials=args.trials, max_walks=args.max_walks,
                      device=args.device, seed=args.seed, n_jobs=args.jobs,
                      amp=args.amp, verbose=args.verbose,
                      max_epochs=args.epochs, forecast_mode=args.forecast_mode,
                      target_mode=args.target_mode)

    ev = evaluate_predictions(out.predictions, features)

    # Non-default modes get their own directory so variants can be compared
    # side by side: ARIMA's two readings of gap G10, and the LSTM target.
    if args.model == "arima":
        suffix = ("" if args.forecast_mode == config.ARIMA_FORECAST_MODE
                  else f"_{args.forecast_mode}")
    else:
        suffix = ("" if args.target_mode == config.LSTM_TARGET
                  else f"_{args.target_mode}")
    run_dir = config.RESULTS_DIR / args.index / f"{args.model}_{args.variant}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out.predictions.to_csv(run_dir / "predictions.csv")

    rows = []
    for mode in ("long_only", "long_short", "buy_hold"):
        label = {"long_only": "Long Only", "long_short": "Long Short",
                 "buy_hold": "Buy&Hold"}[mode]
        model_label = (config.INDICES[args.index]["name"] if mode == "buy_hold"
                       else args.model.upper())
        rows.append({"strategy": label, "model": model_label,
                     **ev["metrics"][mode]})
        ev["equity"][mode].to_frame("equity").to_csv(run_dir / f"equity_{mode}.csv")
        if mode != "buy_hold":
            ev["returns"][mode].to_frame("returns").to_csv(run_dir / f"returns_{mode}.csv")

    metrics_df = (run_dir / "metrics.csv")
    pd_write_metrics(rows, metrics_df)

    walks_df = pd_from_dicts(
        [w.__dict__ | {"chosen": json.dumps(w.chosen, default=str),
                       "top5": json.dumps(w.top5, default=str)}
         for w in out.walks])
    walks_df.to_csv(run_dir / "walks.csv", index=False)

    print(f"\n=== {args.model} {args.index} ({args.variant}) ===")
    print(pd.read_csv(metrics_df, index_col=0).to_string(index=False))


def pd_write_metrics(rows, path):
    import pandas as pd
    pd.DataFrame(rows).round(4).to_csv(path)


def pd_from_dicts(rows):
    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
