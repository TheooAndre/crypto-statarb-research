from dataclasses import dataclass


@dataclass
class Config:
    # --- data ---
    timeframe: str = "4h"
    lookback_days: int = 120
    formation_days: int = 45          # window used to select pairs + fit hedge ratio
    trading_days: int = 21            # out-of-sample window simulated per walk-forward fold
    step_days: int = 14               # roll-forward step between folds
    exchange_id: str = "binance"      # 'binance' | 'bybit' | 'hyperliquid' -- see data_layer.EXCHANGE_PROFILES.
                                       # Market type (linear/inverse, quote currency, etc.) is resolved from
                                       # the exchange's ExchangeProfile, not set here -- those details are
                                       # exchange-specific in ways a single flat field can't safely represent
                                       # (see data_layer.py's module docstring for why).

    # --- universe filters ---
    min_open_interest_usd: float = 5_000_000
    min_24h_volume_usd: float = 25_000_000

    # --- pair selection (LAYER 1: narrow the field, not the final gate) ---
    adf_nonstationarity_pvalue_min: float = 0.10   # each leg must fail to reject unit root above this
    coint_fdr_alpha: float = 0.10                  # deliberately lenient -- see note in pair_selection.py
    min_half_life_bars: float = 2
    max_half_life_bars: float = 90
    beta_refit_every_bars: int = 42                # ~weekly at 4h bars -- avoids per-bar rebalancing churn

    # --- signal ---
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_holding_bars: int = 180
    zscore_window_bars: int = 90

    # --- costs (LAYER 2 model inputs -- the actual arbiter is the backtest, not the p-value) ---
    taker_fee_bps: float = 4.0      # per leg, per fill
    slippage_bps: float = 2.0       # additional assumed slippage per fill
    funding_interval_hours: float = 8.0

    # --- backtest significance (LAYER 2: multiple-testing correction on realized Sharpe, not just discovery p) ---
    backtest_fdr_alpha: float = 0.10
    bootstrap_iterations: int = 1000

    # --- portfolio / sizing ---
    target_pair_vol_annual: float = 0.15
    kelly_fraction: float = 0.3
    max_asset_concentration: float = 0.25   # cap net notional in any one underlying asset

    # --- execution / misc ---
    max_workers: int = 8
    min_data_points: int = 150
    random_seed: int = 7
