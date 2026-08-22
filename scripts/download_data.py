"""Download and cache index data + VIX (paper Sec. 3)."""
from __future__ import annotations
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


from src import config, data


def main() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    for ticker in [*config.INDICES, config.VOL_TICKER]:
        raw = data.load_raw(ticker)
        feats = data.build_features(ticker) if ticker in config.INDICES else None
        last = raw.index[-1].date()
        print(f"{ticker:8s} rows={len(raw):5d} {raw.index[0].date()} -> {last}"
              + (f" | features: {list(feats.columns)}" if feats is not None else ""))


if __name__ == "__main__":
    main()
