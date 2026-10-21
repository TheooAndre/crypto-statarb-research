"""
LAYER 2 of the two-layer validation scheme: this is the actual arbiter of
whether a pair is tradeable, not the discovery p-value from pair_selection.py.

Pure numpy/pandas -- no statsmodels dependency, so unlike pair_selection.py
this WAS fully executed and validated in the build sandbox (see the
__main__ block and the chat response for the synthetic validation results).

Design choices worth knowing about, not just the defaults:
  - Hedge ratio is refit periodically (config.beta_refit_every_bars), not
    every bar. A beta that changes every 4h candle looks great in a backtest
    and is not something you can actually execute without eating constant
    rebalancing costs -- this is deliberately the more conservative,
    practically-executable choice.
  - 1-bar execution lag: a signal computed on bar t's close is applied
    starting bar t+1. Backtesting on the same bar you observed the signal
    is a common, easy-to-miss source of phantom edge.
  - Transaction cost is charged on BOTH legs, weighted by the hedge ratio
    (a beta far from 1 means very unequal notional on each leg -- charging
    a flat "2x fee" ignores that).
  - Funding P&L is a first-class input, not an afterthought -- for perp
    pairs this can dominate the mean-reversion P&L, especially on the alt
    leg. You need to supply a per-bar net funding-rate-differential series;
    this module doesn't fabricate one.
  - evaluate_pair_significance applies a block-bootstrap p-value + FDR
    correction to the pooled REALIZED out-of-sample Sharpe across pairs.
    This is a simplified stand-in for Bailey & Lopez de Prado's deflated
    Sharpe ratio (2014) -- if you're allocating real capital, that's the
    citation to go read next; this gets you most of the same protection
    (multiple-testing-aware backtest significance) without the full
    machinery.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Dependency-free BH so this module works even without statsmodels."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    cutoff = sorted_p[np.max(np.where(below)[0])]
    reject = np.zeros(m, dtype=bool)
    reject[p <= cutoff] = True
    return reject


def periodic_beta(y: pd.Series, x: pd.Series, window: int, refit_every: int) -> pd.Series:
    """Hedge ratio via closed-form rolling OLS slope (Cov/Var -- exact, not
    an approximation, for single-regressor OLS), refit only every
    `refit_every` bars using strictly-trailing data (beta at index i uses
    only [i-window, i), never index i itself) -- causal by construction."""
    n = len(y)
    beta = pd.Series(np.nan, index=y.index)
    yv, xv = y.values, x.values
    for i in range(window, n, refit_every):
        y_win = yv[i - window:i]
        x_win = xv[i - window:i]
        var = np.var(x_win, ddof=1)
        if var > 0:
            cov = np.cov(y_win, x_win)[0, 1]
            beta.iloc[i] = cov / var
    return beta.ffill()


def generate_positions(z: pd.Series, entry_z: float, exit_z: float,
                        stop_z: float, max_holding: int) -> pd.Series:
    """Sequential state machine -- this is inherently path-dependent
    (hysteresis between entry/exit thresholds, max holding period), so a
    loop is the correct tool here, not a forced vectorization."""
    z_vals = z.values
    pos = np.zeros(len(z_vals))
    current, held = 0, 0
    for i, zt in enumerate(z_vals):
        if np.isnan(zt):
            pos[i] = current
            continue
        if current == 0:
            if zt > entry_z:
                current, held = -1, 0
            elif zt < -entry_z:
                current, held = 1, 0
        else:
            held += 1
            exit_now = (
                (current == 1 and zt >= -exit_z)
                or (current == -1 and zt <= exit_z)
                or (abs(zt) >= stop_z)
                or (held >= max_holding)
            )
            if exit_now:
                current, held = 0, 0
        pos[i] = current
    return pd.Series(pos, index=z.index)


def backtest_pair_oos(y: pd.Series, x: pd.Series, beta: pd.Series,
                       config, bars_per_year: float,
                       funding_diff: Optional[pd.Series] = None,
                       eval_start=None) -> dict:
    """Runs the full signal + cost + funding P&L simulation.

    If eval_start is given, y/x/beta may include pre-eval_start history --
    the rolling z-score window uses ALL of it (proper warm-up, no truncation
    artifact), but the position state machine is only allowed to see signal
    from eval_start onward (masked to NaN before that), so no trade can open
    before eval_start and no pre-eval_start P&L leaks into the result. This
    matters: slicing to OOS-only data before computing a rolling stat throws
    away real warm-up history and starves the backtest of usable bars right
    when you need them."""
    common = y.index.intersection(x.index).intersection(beta.dropna().index)
    if len(common) < 30:
        return None
    y, x, beta = y.loc[common], x.loc[common], beta.loc[common]
    spread = y - beta * x

    win = config.zscore_window_bars
    roll_mean = spread.rolling(win, min_periods=max(10, win // 2)).mean()
    roll_std = spread.rolling(win, min_periods=max(10, win // 2)).std()
    z = (spread - roll_mean) / roll_std.replace(0, np.nan)
    if eval_start is not None:
        z = z.copy()
        z.loc[z.index < eval_start] = np.nan  # no entries can form before this point

    positions = generate_positions(z, config.entry_z, config.exit_z, config.stop_z, config.max_holding_bars)
    exec_pos = positions.shift(1).fillna(0)  # 1-bar execution lag
    if eval_start is not None:
        # report only the eval window -- pre-eval_start is guaranteed flat
        # (masked above) so this is a clean slice, not a truncation of live state
        keep = spread.index >= eval_start
        y, x, beta, spread, exec_pos = y[keep], x[keep], beta[keep], spread[keep], exec_pos[keep]

    spread_diff = spread.diff().fillna(0)
    gross_pnl = exec_pos * spread_diff

    pos_change = exec_pos.diff().abs()
    pos_change.iloc[0] = abs(exec_pos.iloc[0])
    cost = pos_change * (1 + beta.abs()) * (config.taker_fee_bps + config.slippage_bps) / 1e4

    funding_pnl = pd.Series(0.0, index=spread.index)
    if funding_diff is not None:
        fd = funding_diff.reindex(spread.index).fillna(0)
        funding_pnl = -exec_pos * fd

    net_pnl = gross_pnl - cost + funding_pnl
    equity = net_pnl.cumsum()

    ret_std = net_pnl.std()
    sharpe = (net_pnl.mean() / ret_std) * np.sqrt(bars_per_year) if ret_std > 0 else 0.0
    running_max = equity.cummax()
    max_dd = float((equity - running_max).min()) if len(equity) else 0.0

    trade_pnls, in_trade, accum = [], False, 0.0
    for t, is_open in (exec_pos != 0).items():
        if is_open and not in_trade:
            in_trade, accum = True, 0.0
        if in_trade:
            accum += net_pnl.loc[t]
        if not is_open and in_trade:
            in_trade = False
            trade_pnls.append(accum)
    if in_trade:
        trade_pnls.append(accum)
    hit_rate = float(np.mean([p > 0 for p in trade_pnls])) if trade_pnls else np.nan

    return {
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "total_pnl": float(equity.iloc[-1]) if len(equity) else 0.0,
        "n_trades": len(trade_pnls),
        "hit_rate": hit_rate,
        "trade_pnls": trade_pnls,
        "turnover": float(pos_change.sum()),
        "equity_curve": equity,
        "net_pnl": net_pnl,
        "gross_pnl_sum": float(gross_pnl.sum()),
        "cost_sum": float(cost.sum()),
        "funding_pnl_sum": float(funding_pnl.sum()),
    }


def block_bootstrap_sharpe_pvalue(net_pnl: pd.Series, bars_per_year: float,
                                   n_boot: int = 1000, block_size: int = 10,
                                   seed: int = 0) -> tuple:
    """H0: true mean P&L per bar is zero. Demean the observed series (imposes
    the null), then circular block-bootstrap it (blocks, not iid resampling,
    because trade P&L is autocorrelated within a holding period) to build
    the null distribution of the Sharpe ratio a series with THIS variance/
    autocorrelation structure would show by chance alone. p = P(boot_sharpe
    >= observed_sharpe)."""
    x = net_pnl.dropna().values
    n = len(x)
    if n < 30 or x.std() == 0:
        return 1.0, 0.0
    observed_sharpe = (x.mean() / x.std()) * np.sqrt(bars_per_year)
    demeaned = x - x.mean()
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    boot_sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        sample = np.concatenate([demeaned[s:s + block_size] for s in starts])[:n]
        s_std = sample.std()
        boot_sharpes[b] = (sample.mean() / s_std) * np.sqrt(bars_per_year) if s_std > 0 else 0.0
    p = (np.sum(boot_sharpes >= observed_sharpe) + 1) / (n_boot + 1)
    return float(p), float(observed_sharpe)


def walk_forward(log_price_panel: Dict[str, pd.Series], candidate_pairs: pd.DataFrame,
                  config, bars_per_year: float,
                  funding_panel: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
    """Genuine multi-fold walk-forward: rolls a (formation_days, trading_days)
    window forward by step_days across the full available history, refitting
    beta causally within each fold and evaluating OOS P&L only in that fold's
    trading window. All folds' OOS P&L for a pair are pooled into one series
    before computing Sharpe and the Layer-2 significance test -- a single
    30-day OOS slice rarely contains enough independent trades to say
    anything statistically meaningful; pooling folds across the full
    lookback is what gives the significance test real power."""
    if candidate_pairs.empty:
        return pd.DataFrame(), {}

    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(config.timeframe, 6)
    formation_bars = config.formation_days * bars_per_day
    trading_bars = config.trading_days * bars_per_day
    step_bars = max(1, config.step_days * bars_per_day)

    results = []
    for _, row in candidate_pairs.iterrows():
        a, b = row["asset1"], row["asset2"]
        if a not in log_price_panel or b not in log_price_panel:
            continue
        y_full, x_full = log_price_panel[a], log_price_panel[b]
        common = y_full.index.intersection(x_full.index)
        y_full, x_full = y_full.loc[common], x_full.loc[common]
        n = len(y_full)

        fold_pnls, fold_metrics = [], []
        fold_start = 0
        while fold_start + formation_bars + trading_bars <= n:
            trade_end = fold_start + formation_bars + trading_bars
            y_fold = y_full.iloc[fold_start:trade_end]
            x_fold = x_full.iloc[fold_start:trade_end]
            beta_fold = periodic_beta(y_fold, x_fold, window=formation_bars,
                                       refit_every=config.beta_refit_every_bars)
            eval_start_ts = y_fold.index[formation_bars]

            funding_diff = None
            if funding_panel is not None and a in funding_panel and b in funding_panel:
                fa = funding_panel[a].reindex(y_fold.index).fillna(0)
                fb = funding_panel[b].reindex(x_fold.index).fillna(0)
                funding_diff = fa - float(row.get("beta", 1.0)) * fb

            res = backtest_pair_oos(y_fold, x_fold, beta_fold, config, bars_per_year,
                                     funding_diff=funding_diff, eval_start=eval_start_ts)
            if res is not None:
                fold_pnls.append(res["net_pnl"])
                fold_metrics.append(res)
            fold_start += step_bars

        if not fold_pnls:
            continue

        combined_pnl = pd.concat(fold_pnls).sort_index()
        combined_equity = combined_pnl.cumsum()
        ret_std = combined_pnl.std()
        sharpe = (combined_pnl.mean() / ret_std) * np.sqrt(bars_per_year) if ret_std > 0 else 0.0
        running_max = combined_equity.cummax()
        max_dd = float((combined_equity - running_max).min()) if len(combined_equity) else 0.0

        # pool raw trade P&Ls across folds (not per-fold hit-rate averages --
        # per-fold trade counts are too small to average reliably)
        pooled_trades = [p for m in fold_metrics for p in m["trade_pnls"]]
        wins = [p for p in pooled_trades if p > 0]
        losses = [-p for p in pooled_trades if p <= 0]
        hit_rate = float(len(wins) / len(pooled_trades)) if pooled_trades else np.nan
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        p_value, _ = block_bootstrap_sharpe_pvalue(combined_pnl, bars_per_year,
                                                     n_boot=config.bootstrap_iterations,
                                                     seed=config.random_seed)

        results.append({
            "asset1": a, "asset2": b,
            "n_folds": len(fold_metrics),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "total_pnl": float(combined_equity.iloc[-1]) if len(combined_equity) else 0.0,
            "n_trades": len(pooled_trades),
            "hit_rate": hit_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "turnover": float(sum(m["turnover"] for m in fold_metrics)),
            "backtest_pvalue": p_value,
            "discovery_pvalue_fdr_adj": row.get("pvalue_fdr_adj", np.nan),
            "_net_pnl": combined_pnl,
        })

    if not results:
        return pd.DataFrame(), {}

    # --- LAYER 2 multiple-testing correction on realized, pooled OOS Sharpe ---
    reject = _benjamini_hochberg(np.array([r["backtest_pvalue"] for r in results]), config.backtest_fdr_alpha)
    for r, rej in zip(results, reject):
        r["backtest_fdr_significant"] = bool(rej)

    net_pnl_by_pair = {(r["asset1"], r["asset2"]): r["_net_pnl"] for r in results}
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_net_pnl"} for r in results])
    return df.sort_values("backtest_pvalue").reset_index(drop=True), net_pnl_by_pair
