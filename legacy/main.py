"""
CLI entry point. NOTE: the `scan` command depends on data_layer.py (ccxt)
and pair_selection.py (statsmodels), neither of which could be executed in
the sandbox used to build this -- see those files' docstrings. Everything
downstream of "you already have a log-price panel in hand" (selection ->
backtest -> sizing) was fully tested against synthetic data; the live data
path was not. Run `python main.py demo` first -- it exercises the tested
path end-to-end on synthetic data with no network/ccxt/statsmodels
dependency, so you have a known-good reference run before pointing this at
a real exchange.

Usage:
    python main.py scan --lookback-days 120 --timeframe 4h
    python main.py demo
"""

import argparse
import asyncio
import sys

import numpy as np
import pandas as pd

from config import Config


def _bars_per_year(timeframe: str) -> float:
    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 6)
    return 365 * bars_per_day


async def _build_funding_panel(config: Config, candidates, log_price_panel) -> dict:
    """Fetches funding history only for symbols actually appearing in a
    candidate pair (not the whole universe -- no point paying the API
    calls for symbols that didn't survive Layer 1), resampled onto each
    symbol's own bar index via DataLayer.get_funding_series."""
    from data_layer import DataLayer

    symbols = sorted(set(candidates["asset1"]) | set(candidates["asset2"]))
    layer = DataLayer(config)
    panel = {}
    try:
        for sym in symbols:
            if sym not in log_price_panel:
                continue
            bar_index = log_price_panel[sym].index
            try:
                panel[sym] = await layer.get_funding_series(sym, bar_index)
            except Exception as e:
                print(f"  funding fetch failed for {sym}, proceeding without it for this symbol: {e}")
    finally:
        await layer.close()
    return panel


def run_scan(config: Config):
    """Full live pipeline: fetch -> select -> backtest -> size -> report."""
    import asyncio
    from data_layer import build_universe_and_data
    from pair_selection import log_prices, select_candidate_pairs
    from backtest import walk_forward
    from portfolio import size_portfolio, check_concentration
    from data_layer import get_profile

    profile = get_profile(config.exchange_id)  # raises clearly if exchange_id has no profile registered
    print(f"Exchange: {config.exchange_id}  |  quote currencies: {profile.quote_currencies}  |  "
          f"timeframe: {config.timeframe}  |  lookback: {config.lookback_days}d")
    if profile.notes:
        print(f"  note: {profile.notes}")
    print("Fetching universe...")
    ohlcv = asyncio.run(build_universe_and_data(config))
    if not ohlcv:
        print("No data fetched -- aborting. Check network/exchange connectivity and filters.")
        sys.exit(1)
    print(f"Fetched {len(ohlcv)} symbols passing liquidity filters.")

    log_price_panel = log_prices(ohlcv)

    # Layer 1: formation-window-only pair discovery (use the earliest
    # formation_days worth of bars so this doesn't peek at the trading window)
    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(config.timeframe, 6)
    formation_bars = config.formation_days * bars_per_day
    formation_panel = {sym: s.iloc[:formation_bars] for sym, s in log_price_panel.items()}

    print(f"Running cointegration screen (ADF prescreen + Engle-Granger + FDR alpha={config.coint_fdr_alpha})...")
    candidates = select_candidate_pairs(formation_panel, config)
    if candidates.empty:
        print("No pairs survived Layer 1 (FDR-corrected cointegration screen). "
              "Consider a larger universe, longer formation window, or a higher coint_fdr_alpha.")
        sys.exit(0)
    print(f"{len(candidates)} candidate pairs survived Layer 1.")

    print("Running walk-forward backtest (Layer 2: realized OOS Sharpe + FDR)...")
    bars_per_year = _bars_per_year(config.timeframe)

    print("Fetching funding rate history for candidate pair legs...")
    funding_panel = asyncio.run(_build_funding_panel(config, candidates, log_price_panel))

    results, net_pnl_by_pair = walk_forward(log_price_panel, candidates, config, bars_per_year,
                                             funding_panel=funding_panel)
    if results.empty:
        print("No pairs produced a usable OOS backtest (insufficient history for a full fold?).")
        sys.exit(0)

    n_sig = int(results["backtest_fdr_significant"].sum())
    print(f"{n_sig} / {len(results)} pairs are significant after Layer-2 FDR correction "
          f"(alpha={config.backtest_fdr_alpha}).")
    print(results[["asset1", "asset2", "n_folds", "sharpe", "n_trades", "hit_rate",
                    "backtest_pvalue", "backtest_fdr_significant"]].to_string(index=False))

    merged = results.merge(candidates[["asset1", "asset2", "beta"]], on=["asset1", "asset2"], how="left")
    sized = size_portfolio(merged, net_pnl_by_pair, config, bars_per_year)
    if sized.empty:
        print("\nNo pairs sized -- either none passed Layer 2, or none had enough pooled trades "
              "for a reliable Kelly estimate yet.")
        return

    print("\nPortfolio sizing:")
    print(sized[["asset1", "asset2", "sharpe", "hit_rate", "n_trades",
                  "kelly_fraction_raw", "vol_target_multiplier", "position_size"]].to_string(index=False))

    flags = check_concentration(sized, config)
    if not flags.empty:
        print("\nCONCENTRATION WARNING -- net exposure exceeds max_asset_concentration:")
        print(flags.to_string())
    else:
        print("\nNo asset concentration breaches.")


def run_demo(config: Config):
    """End-to-end run on synthetic data -- no network, no ccxt, no
    statsmodels required. This is the path that was actually executed and
    validated while building this pipeline."""
    from backtest import walk_forward
    from portfolio import size_portfolio, check_concentration

    rng = np.random.default_rng(config.random_seed)
    bars_per_day = 6
    n_bars = 150 * bars_per_day

    def make_pair(half_life, seed):
        r = np.random.default_rng(seed)
        phi = 0.5 ** (1.0 / half_life)
        trend = np.cumsum(r.normal(0, 1, n_bars))
        spread = np.zeros(n_bars)
        for t in range(1, n_bars):
            spread[t] = phi * spread[t - 1] + r.normal(0, 0.5)
        idx = pd.date_range("2026-01-01", periods=n_bars, freq="4h")
        return (pd.Series(trend + 0.5 * spread, index=idx),
                pd.Series(trend - 0.5 * spread, index=idx))

    def make_noise(seed):
        r = np.random.default_rng(seed)
        idx = pd.date_range("2026-01-01", periods=n_bars, freq="4h")
        return (pd.Series(np.cumsum(r.normal(0, 1, n_bars)), index=idx),
                pd.Series(np.cumsum(r.normal(0, 1, n_bars)), index=idx))

    panel = {}
    panel["DEMO_TRUE_A"], panel["DEMO_TRUE_B"] = make_pair(8, seed=30)
    for i in range(6):
        panel[f"DEMO_NOISE{i}_A"], panel[f"DEMO_NOISE{i}_B"] = make_noise(seed=100 + i)

    candidates = pd.DataFrame(
        [{"asset1": "DEMO_TRUE_A", "asset2": "DEMO_TRUE_B", "beta": 1.0, "pvalue_fdr_adj": 0.001}]
        + [{"asset1": f"DEMO_NOISE{i}_A", "asset2": f"DEMO_NOISE{i}_B", "beta": 1.0, "pvalue_fdr_adj": 0.08}
           for i in range(6)]
    )

    print("Running walk-forward backtest on synthetic universe (1 real pair + 6 noise pairs)...")
    results, net_pnl_by_pair = walk_forward(panel, candidates, config, _bars_per_year(config.timeframe))
    print(results[["asset1", "asset2", "n_folds", "sharpe", "n_trades", "hit_rate",
                    "backtest_pvalue", "backtest_fdr_significant"]].to_string(index=False))

    merged = results.merge(candidates[["asset1", "asset2", "beta"]], on=["asset1", "asset2"], how="left")
    sized = size_portfolio(merged, net_pnl_by_pair, config, _bars_per_year(config.timeframe))
    print("\nPortfolio sizing (only Layer-2-significant pairs are eligible):")
    if sized.empty:
        print("  (none sized)")
    else:
        print(sized[["asset1", "asset2", "sharpe", "n_trades", "kelly_fraction_raw",
                      "vol_target_multiplier", "position_size"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Crypto stat-arb research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="run the full live pipeline against an exchange")
    p_scan.add_argument("--lookback-days", type=int, default=None)
    p_scan.add_argument("--timeframe", type=str, default=None)
    p_scan.add_argument("--exchange", type=str, default=None,
                         choices=["binance", "bybit", "hyperliquid"],
                         help="argparse enforces this against the exchanges this pipeline currently "
                              "has a profile for -- add a new one to data_layer.EXCHANGE_PROFILES "
                              "before adding it here, don't just widen this list")

    sub.add_parser("demo", help="run the tested path end-to-end on synthetic data (no network needed)")

    args = parser.parse_args()
    config = Config()
    if args.command == "scan":
        if args.lookback_days:
            config.lookback_days = args.lookback_days
        if args.timeframe:
            config.timeframe = args.timeframe
        if args.exchange:
            config.exchange_id = args.exchange
        run_scan(config)
    elif args.command == "demo":
        run_demo(config)


if __name__ == "__main__":
    main()
