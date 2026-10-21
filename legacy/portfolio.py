"""
Position sizing and portfolio-level netting -- absent entirely from the
original script, which sized each pair in isolation and never checked
whether the same underlying asset was being stacked across multiple pairs
simultaneously (very possible: BTC as a leg in several pairs at once means
your "diversified" pairs book can be a concentrated BTC bet without anyone
noticing until it moves).
"""

from typing import Dict

import numpy as np
import pandas as pd


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                    n_trades: int = 0, min_trades: int = 30) -> float:
    """f* = p - q/R, R = avg_win/avg_loss (payoff ratio).

    Returns 0 if n_trades < min_trades: with few trades (e.g. a 100% hit
    rate over 12 samples), the win-rate/payoff-ratio estimates themselves
    are too noisy to size against -- an apparently "perfect" small sample is
    a reason for MORE caution, not a green light to size up. This is a
    deliberate floor, not a smoothing trick around the avg_loss=0 edge case
    that a 100%-hit-rate small sample produces."""
    if n_trades < min_trades:
        return 0.0
    if avg_loss <= 0 or avg_win <= 0 or np.isnan(win_rate):
        return 0.0
    R = avg_win / avg_loss
    q = 1 - win_rate
    f = win_rate - q / R
    return max(0.0, f)


def vol_target_multiplier(pnl_series: pd.Series, target_vol_annual: float, bars_per_year: float) -> float:
    """Scalar to apply to a unit (+-1) position so the pair's OOS spread P&L
    vol matches target_vol_annual. Uses REALIZED backtest vol, not an
    assumption -- if the pair was choppier than expected, this sizes it down
    automatically."""
    realized_vol = pnl_series.std() * np.sqrt(bars_per_year)
    if realized_vol <= 0 or np.isnan(realized_vol):
        return 0.0
    return target_vol_annual / realized_vol


def size_portfolio(backtest_results: pd.DataFrame, net_pnl_by_pair: Dict[tuple, pd.Series],
                    config, bars_per_year: float, capital: float = 1.0) -> pd.DataFrame:
    """
    Only sizes pairs that already passed BOTH layers (FDR-significant
    discovery AND FDR-significant backtest Sharpe) -- sizing a pair that
    didn't clear the significance bar just means allocating real capital to
    something indistinguishable from noise.
    """
    if backtest_results.empty:
        return pd.DataFrame()

    eligible = backtest_results[backtest_results["backtest_fdr_significant"]].copy()
    if eligible.empty:
        return eligible

    sizes = []
    for _, row in eligible.iterrows():
        key = (row["asset1"], row["asset2"])
        pnl = net_pnl_by_pair.get(key)
        vol_mult = vol_target_multiplier(pnl, config.target_pair_vol_annual, bars_per_year) if pnl is not None else 0.0
        f_kelly = kelly_fraction(row.get("hit_rate", np.nan), row.get("avg_win", 0), row.get("avg_loss", 0),
                                  n_trades=row.get("n_trades", 0))
        size = capital * vol_mult * f_kelly * config.kelly_fraction
        sizes.append(size)

    eligible["kelly_fraction_raw"] = [
        kelly_fraction(r.get("hit_rate", np.nan), r.get("avg_win", 0), r.get("avg_loss", 0), n_trades=r.get("n_trades", 0))
        for _, r in eligible.iterrows()
    ]
    eligible["vol_target_multiplier"] = [
        vol_target_multiplier(net_pnl_by_pair.get((r["asset1"], r["asset2"])),
                               config.target_pair_vol_annual, bars_per_year)
        if net_pnl_by_pair.get((r["asset1"], r["asset2"])) is not None else 0.0
        for _, r in eligible.iterrows()
    ]
    eligible["position_size"] = sizes
    return eligible.sort_values("position_size", ascending=False).reset_index(drop=True)


def net_asset_exposure(sized_pairs: pd.DataFrame, beta_col: str = "beta") -> pd.Series:
    """Nets exposure per underlying asset across the whole selected book.
    Pair convention: position_size > 0 means long asset1 / short beta*asset2."""
    exposure: Dict[str, float] = {}
    for _, row in sized_pairs.iterrows():
        size = row["position_size"]
        beta = row.get(beta_col, 1.0)
        exposure[row["asset1"]] = exposure.get(row["asset1"], 0.0) + size
        exposure[row["asset2"]] = exposure.get(row["asset2"], 0.0) - size * beta
    return pd.Series(exposure).sort_values(key=abs, ascending=False)


def check_concentration(sized_pairs: pd.DataFrame, config, beta_col: str = "beta") -> pd.DataFrame:
    """Flags (does not silently fix) any asset whose net notional across the
    book exceeds max_asset_concentration of total gross exposure. Original
    script had no equivalent check at all -- this is the difference between
    'four diversified pairs' and 'one leveraged BTC bet wearing four hats.'"""
    exposure = net_asset_exposure(sized_pairs, beta_col=beta_col)
    gross = sized_pairs["position_size"].abs().sum() + (sized_pairs["position_size"].abs() * sized_pairs[beta_col].abs()).sum()
    if gross == 0:
        return pd.DataFrame()
    conc = (exposure.abs() / gross).rename("concentration")
    flags = conc[conc > config.max_asset_concentration]
    return flags.to_frame()
