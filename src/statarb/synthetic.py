"""Synthetic fixtures for MECHANICAL testing, never return projections."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import ResearchData


def synthetic_market(days: int = 720, seed: int = 7) -> ResearchData:
    rng = np.random.default_rng(seed)
    n = days * 6
    idx = pd.date_range('2023-01-01', periods=n, freq='4h', tz='UTC')
    common = np.cumsum(rng.normal(0, .010, n)) + 5
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = .94 * spread[i-1] + rng.normal(0, .018)
    series = {'SIM_A': common + spread/2, 'SIM_B': common - spread/2,
              'SIM_C': np.cumsum(rng.normal(0,.011,n))+4}
    bars, funding = {}, {}
    for symbol, lv in series.items():
        # Each candle's open known at timestamp; close after the full 4h interval.
        opening = np.exp(lv)
        close = opening * np.exp(rng.normal(0, .002, n))
        df = pd.DataFrame({'open': opening, 'high': np.maximum(opening,close)*1.001,
                           'low': np.minimum(opening,close)*.999, 'close': close,
                           'volume': 100_000.}, index=idx)
        bars[symbol] = df
        # Synthetic event funding, explicit zero only for validating the accounting pipeline.
        funding[symbol] = pd.Series(np.zeros(n), index=idx)
    snapshots = pd.date_range(idx[0], idx[-1], freq='3D')
    membership = pd.DataFrame([{'timestamp': ts, 'symbol': sym, 'active': True, 'linear': True, 'quote': 'USDT', 'open_interest_usd': 20_000_000.0,
                               'volume_24h_usd': 150_000_000.0}
                               for ts in snapshots for sym in bars])
    return ResearchData(bars, funding, membership, {'venue':'SYNTHETIC',
                        'quote_currency':'USDT', 'contract_type':'linear_perpetual',
                        'funding_complete':True, 'universe_point_in_time':True,
                        'synthetic':True, 'warning':'NOT a strategy performance dataset'})
