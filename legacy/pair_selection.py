"""
Pair discovery -- LAYER 1 of a two-layer validation scheme.

Uses statsmodels (adfuller / coint / multipletests) for the actual hypothesis
tests -- the standard, correctly-calibrated tool, same dependency your
original script already used. NOTE: this specific file could not be executed
in the sandbox used to build this (no network access to install statsmodels
there), so it has not been run end-to-end here. Its logic mirrors
validate_fdr_concept.py exactly -- a from-scratch Monte Carlo ADF/EG
implementation that WAS fully tested in that sandbox and is included alongside
this pipeline -- with statsmodels swapped in for the real p-value calculation,
because that validation exercise showed a from-scratch Monte Carlo p-value
can't resolve the tail multiple-testing correction needs at any real universe
size (see the docstring and printed output of validate_fdr_concept.py for why).
Smoke-test this file on a small symbol subset before trusting it.

Key fixes vs. the original script:
  - log-prices throughout (raw price-level regression is dominated by scale
    when instruments span orders of magnitude, which crypto does -- BTC vs a
    sub-cent altcoin -- and makes the hedge ratio close to meaningless)
  - ADF prescreen on each leg before testing the pair (Engle-Granger is only
    a meaningful test if both legs are individually I(1); skipping this means
    you can "detect cointegration" between two series where the test doesn't
    mean what you think it means)
  - Benjamini-Hochberg FDR correction across the ENTIRE set of pairs tested
    in one pass, not a flat p<0.05 per pair (the original script's single
    biggest issue -- see validate_fdr_concept.py for the empirical proof)
  - FDR alpha here defaults to a deliberately LENIENT 0.10, not 0.05. This
    is not a mistake -- validate_fdr_concept.py demonstrated that a strict
    correction across a large universe can erase real signal along with the
    noise (the true pairs' own p-values weren't small enough to survive a
    razor-thin per-rank threshold). This layer's job is to cut an
    unmanageable number of candidates down to a shortlist; backtest.py's
    walk-forward P&L simulation with real costs is the actual arbiter of
    whether a pair is tradeable, and it applies its OWN multiple-testing
    correction to the realized Sharpe ratios (see backtest.py).
"""

from itertools import combinations
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.multitest import multipletests

from config import Config


def log_prices(ohlcv: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    return {sym: np.log(df["close"]) for sym, df in ohlcv.items() if "close" in df.columns}


def _adf_nonstationary(series: pd.Series, pvalue_min: float) -> bool:
    """True if we FAIL to reject the unit-root null -- i.e. series looks I(1),
    which is the prerequisite for Engle-Granger to mean what you think it means."""
    s = series.dropna()
    if len(s) < 30:
        return False
    try:
        result = adfuller(s, autolag="BIC", result_object=True)
        pvalue = result.pvalue
    except Exception:
        return False
    return pvalue > pvalue_min


def half_life_ou(spread: pd.Series) -> tuple:
    """AR(1) mean-reversion speed on the spread: ds_t = c + theta*s_{t-1} + eps.
    Returns (half_life_in_bars, t_stat_on_theta). A theta that isn't
    statistically distinguishable from zero shouldn't be trusted as a real
    half-life point estimate, even if the arithmetic produces a number."""
    s = spread.dropna()
    s_lag = s.shift(1)
    ds = s.diff()
    df = pd.DataFrame({"ds": ds, "s_lag": s_lag}).dropna()
    if len(df) < 20:
        return np.inf, 0.0
    X = np.column_stack([np.ones(len(df)), df["s_lag"].values])
    y = df["ds"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 2
    if dof <= 0:
        return np.inf, 0.0
    sigma2 = (resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.inf, 0.0
    se = np.sqrt(sigma2 * xtx_inv[1, 1])
    theta = -beta[1]
    t_stat = beta[1] / se if se > 0 else 0.0
    half_life = np.log(2) / theta if theta > 0 else np.inf
    return half_life, t_stat


def select_candidate_pairs(log_price_panel: Dict[str, pd.Series], config: Config) -> pd.DataFrame:
    """
    Runs on a FORMATION window only (caller is responsible for slicing —
    see backtest.walk_forward, which passes only data strictly before the
    trading window to avoid lookahead).
    """
    # --- ADF prescreen ---
    nonstationary = {
        sym: s for sym, s in log_price_panel.items()
        if len(s.dropna()) >= config.min_data_points
        and _adf_nonstationary(s, config.adf_nonstationarity_pvalue_min)
    }
    symbols = list(nonstationary.keys())
    if len(symbols) < 2:
        return pd.DataFrame()

    # --- Engle-Granger on every candidate pair ---
    records = []
    for a, b in combinations(symbols, 2):
        common = nonstationary[a].index.intersection(nonstationary[b].index)
        if len(common) < config.min_data_points:
            continue
        ya, xb = nonstationary[a].loc[common], nonstationary[b].loc[common]
        if np.isclose(ya.var(), 0) or np.isclose(xb.var(), 0):
            continue
        try:
            score, pvalue, _ = coint(ya, xb, autolag="BIC")
        except Exception:
            continue
        records.append({"asset1": a, "asset2": b, "pvalue": pvalue, "eg_score": score, "n_obs": len(common)})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # --- LAYER 1 FDR correction across ALL pairs tested this formation window ---
    reject, adj_p, *_ = multipletests(df["pvalue"].values, alpha=config.coint_fdr_alpha, method="fdr_bh")
    df["pvalue_fdr_adj"] = adj_p
    df["fdr_significant"] = reject
    df = df[df["fdr_significant"]].copy()
    if df.empty:
        return df

    # --- hedge ratio + half-life on the formation window, for each surviving pair ---
    betas, half_lives, hl_tstats = [], [], []
    for _, row in df.iterrows():
        a, b = row["asset1"], row["asset2"]
        common = nonstationary[a].index.intersection(nonstationary[b].index)
        ya, xb = nonstationary[a].loc[common], nonstationary[b].loc[common]
        cov = np.cov(ya, xb)[0, 1]
        var = np.var(xb, ddof=1)
        beta = cov / var if var > 0 else np.nan
        spread = ya - beta * xb
        hl, t_stat = half_life_ou(spread)
        betas.append(beta)
        half_lives.append(hl)
        hl_tstats.append(t_stat)

    df["beta"] = betas
    df["half_life_bars"] = half_lives
    df["half_life_tstat"] = hl_tstats

    df = df[
        (df["half_life_bars"] >= config.min_half_life_bars)
        & (df["half_life_bars"] <= config.max_half_life_bars)
        & (df["half_life_tstat"].abs() >= 1.96)   # require the mean-reversion itself to be significant, not just the level cointegration
    ]

    return df.sort_values("pvalue_fdr_adj").reset_index(drop=True)
