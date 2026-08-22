"""Sweep the Sec. 6 sensitivity variants for a given index/model (server-side).

ARIMA variants: arima_orders_0_3, arima_bic (CPU-fast).
LSTM/hybrid variants: dropout_0.05, dropout_0.10, batch_16, batch_64.

Paper Sec. 6 alters ONLY the order range and information criterion for ARIMA,
and ONLY dropout and batch size for LSTM / LSTM-ARIMA, so the two variant lists
are deliberately disjoint apart from the shared base case.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import subprocess  # noqa: E402

from src import config  # noqa: E402

ARIMA_VARIANTS = ["base", "arima_orders_0_3", "arima_bic"]
NEURAL_VARIANTS = ["base", "dropout_0.05", "dropout_0.10", "batch_16", "batch_64"]

RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["arima", "lstm", "hybrid"])
    p.add_argument("--index", required=True, choices=list(config.INDICES))
    p.add_argument("--device", default=None)
    p.add_argument("--trials", type=int, default=config.N_TRIALS)
    p.add_argument("--epochs", type=int, default=config.LSTM_MAX_EPOCHS)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--arima-forecast", dest="forecast_mode",
                   default=config.ARIMA_FORECAST_MODE,
                   choices=["rolling", "static"])
    p.add_argument("--keep-going", action="store_true",
                   help="continue the sweep if one variant fails")
    args = p.parse_args()

    variants = ARIMA_VARIANTS if args.model == "arima" else NEURAL_VARIANTS
    failures = []
    for variant in variants:
        # Absolute path: a relative one breaks the moment this is launched from
        # anywhere but the repo root, which is the normal case on a job runner.
        cmd = [sys.executable, str(RUN_EXPERIMENT),
               "--model", args.model, "--index", args.index,
               "--variant", variant, "--trials", str(args.trials),
               "--epochs", str(args.epochs), "--jobs", str(args.jobs),
               "--seed", str(args.seed),
               "--arima-forecast", args.forecast_mode]
        if args.device:
            cmd += ["--device", args.device]
        if args.amp:
            cmd += ["--amp"]
        print(f"\n>>> running: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=str(ROOT), check=not args.keep_going)
        if result.returncode != 0:
            failures.append(variant)
            print(f"!!! variant {variant} failed (exit {result.returncode})",
                  flush=True)

    if failures:
        print(f"\nFAILED variants: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
