"""Data download, cleansing and feature engineering (paper Sec. 3 and 4.9.2).

Features per index:
- close: raw closing price
- volatility: VIX for ^GSPC, else 21-day annualized realized volatility (rolling,
  past-only; see gap G4 in the README)
- volume: raw trading volume from Yahoo (see the README's Data section)
- arima_residual: added later by the hybrid pipeline (src/wfo.py)
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from . import config


def _realized_volatility(close: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualized realized volatility of daily simple returns.

    annRV = sqrt(mean(R_t^2 over last `window` days)) * sqrt(252)
    """
    rets = close.pct_change()
    return np.sqrt(rets.pow(2).rolling(window).mean()) * np.sqrt(config.TRADING_DAYS)


@functools.lru_cache(maxsize=None)
def _fetch_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_raw(ticker: str) -> pd.DataFrame:
    """Download (cached in data/) and clean a single ticker's OHLCV data."""
    config.DATA_DIR.mkdir(exist_ok=True)
    cache = config.DATA_DIR / f"{ticker.replace('^', '')}.csv"

    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        df = _fetch_yf(ticker, config.DATA_START, config.DATA_END)
        df.to_csv(cache)

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    df["Volume"] = df["Volume"].fillna(0.0)
    return df.rename(columns={"Close": "close", "Volume": "volume"})


def build_features(ticker: str) -> pd.DataFrame:
    """Feature frame: close, volatility (VIX or realized), volume; indexed by date."""
    df = load_raw(ticker)[["close", "volume"]].copy()

    if config.INDICES[ticker]["vol_source"] == "vix":
        vix = load_raw(config.VOL_TICKER)[["close"]].rename(columns={"close": "volatility"})
        df = df.join(vix, how="left")
        df["volatility"] = df["volatility"].ffill().bfill()
    else:
        df["volatility"] = _realized_volatility(df["close"])
        # Seed early NaNs with the first computable value so the initial
        # walk-forward windows remain usable.
        df["volatility"] = df["volatility"].bfill()

    # ^GSPC reports zero volume for much of its history (gap G5): zero it out
    # consistently rather than mixing 0s with real values.
    if ticker == "^GSPC" and (df["volume"] == 0).mean() > 0.5:
        df["volume"] = 0.0

    return df[["close", "volatility", "volume"]]


def market_returns(close: pd.Series) -> pd.Series:
    """Daily simple returns of the close series."""
    return close.pct_change().fillna(0.0)
