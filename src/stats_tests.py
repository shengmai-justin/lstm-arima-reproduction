"""Statistical significance tests (paper Sec. 5.2).

1. Paired t-test on daily returns (strategy - benchmark), H1: mean > 0, alpha = 10%.
2. OLS regression R_strategy = alpha + beta * R_benchmark + eps, right-tailed
   test on alpha (H1: alpha > 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_ttest(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    diff = (strategy_returns - benchmark_returns).dropna()
    t, p_two = stats.ttest_1samp(diff, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2  # H1: mean > 0
    return {"mean_diff": float(diff.mean()), "t_stat": float(t),
            "p_value_one_sided": float(p_one), "significant_10pct": bool(p_one < 0.10)}


def regression_alpha(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    # dropna to match paired_ttest: without it a single NaN makes every output
    # NaN and `significant_10pct` silently False (nan < 0.10 is False), so the
    # two Sec. 5.2 tests would also run on different samples.
    joined = pd.concat([strategy_returns.rename("y"),
                        benchmark_returns.rename("x")], axis=1,
                       join="inner").dropna()
    if len(joined) < 3:
        raise ValueError(f"need >=3 paired observations, got {len(joined)}")
    y = joined["y"].to_numpy()
    x = joined["x"].to_numpy()
    n, dof = len(joined), len(joined) - 2

    slope, intercept, _, p_beta_two, se_beta = stats.linregress(x, y)

    resid = y - (intercept + slope * x)
    s2 = float((resid ** 2).sum() / dof)
    x_bar = float(x.mean())
    sxx = float(((x - x_bar) ** 2).sum())
    se_alpha = float(np.sqrt(s2 * (1.0 / n + x_bar ** 2 / sxx)))

    t_alpha = intercept / se_alpha
    p_alpha = float(stats.t.sf(t_alpha, dof))  # right tail: H1: alpha > 0
    t_beta = slope / se_beta
    p_beta = p_beta_two / 2 if t_beta > 0 else 1 - p_beta_two / 2

    return {"alpha": float(intercept), "se_alpha": se_alpha,
            "t_alpha": float(t_alpha), "p_alpha_one_sided": float(p_alpha),
            "beta": float(slope), "se_beta": float(se_beta),
            "t_beta": float(t_beta), "p_beta": float(p_beta),
            "significant_10pct": bool(p_alpha < 0.10)}
