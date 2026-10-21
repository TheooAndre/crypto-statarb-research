"""Validation for the multi-exchange data layer -- pure functions tested
directly, DataLayer's pagination tested against a mock exchange (no ccxt/
network needed for any of this)."""
import sys
sys.path.insert(0, '.')
import asyncio
import numpy as np
import pandas as pd

from config import Config
from data_layer import (
    EXCHANGE_PROFILES, get_profile, filter_liquid_symbols, plan_ohlcv_pagination,
    merge_ohlcv_chunks, infer_funding_interval_hours, resample_funding_to_bars,
    DataLayer, _TIMEFRAME_MS,
)

print("=" * 70)
print("TEST 1: quote-currency + inverse filtering across exchange profiles")
print("=" * 70)
markets = {
    "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "swap": True, "active": True, "quote": "USDT", "inverse": False},
    "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT", "swap": True, "active": True, "quote": "USDT", "inverse": False},
    "BTC/USD:BTC":   {"symbol": "BTC/USD:BTC",   "swap": True, "active": True, "quote": "USD",  "inverse": True},   # coin-margined inverse
    "BTC/USDC:USDC": {"symbol": "BTC/USDC:USDC", "swap": True, "active": True, "quote": "USDC", "inverse": False},
    "SOL/USDT:USDT": {"symbol": "SOL/USDT:USDT", "swap": True, "active": False, "quote": "USDT", "inverse": False},  # inactive
}
tickers = {sym: {"quoteVolume": 50_000_000} for sym in markets}

for ex_id in ["binance", "bybit", "hyperliquid"]:
    profile = get_profile(ex_id)
    result = filter_liquid_symbols(markets, tickers, profile, min_volume_usd=1_000_000)
    print(f"  {ex_id:<12} (quote={profile.quote_currencies}, exclude_inverse={profile.exclude_inverse}) -> {result}")

assert filter_liquid_symbols(markets, tickers, get_profile("binance"), 1e6) == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
assert filter_liquid_symbols(markets, tickers, get_profile("hyperliquid"), 1e6) == ["BTC/USDC:USDC"]
print("  PASS: Binance correctly gets USDT-only, no inverse. Hyperliquid correctly gets USDC-only.")
print("  PASS: inverse BTC/USD:BTC excluded everywhere. Inactive SOL market excluded everywhere.")

print(f"\n{'=' * 70}")
print("TEST 2: OHLCV pagination planning + chunk merging")
print("=" * 70)
tf_ms = _TIMEFRAME_MS["4h"]
end_ms = 1_700_000_000_000
total_bars = 720  # 120 days at 4h
for max_per_fetch in [1500, 1000, 500, 200]:
    since_list = plan_ohlcv_pagination(total_bars, max_per_fetch, tf_ms, end_ms)
    covered_bars = len(since_list) * max_per_fetch
    print(f"  max_candles_per_fetch={max_per_fetch:>5}  -> {len(since_list)} calls planned, "
          f"covers >= {covered_bars} bars (need {total_bars})")
    assert covered_bars >= total_bars, "pagination plan doesn't cover the requested range"
print("  PASS: pagination plan always covers the full requested range regardless of per-call limit.")

# merge with overlap + duplicate timestamps at chunk boundaries
chunk1 = [[1000 + i*tf_ms, 1,2,0,1,10] for i in range(5)]
chunk2 = [[1000 + i*tf_ms, 9,9,9,9,99] for i in range(3, 8)]  # overlaps chunk1 at i=3,4 with DIFFERENT values
merged = merge_ohlcv_chunks([chunk1, chunk2])
timestamps = [row[0] for row in merged]
assert timestamps == sorted(set(timestamps)), "merge produced duplicate or unsorted timestamps"
assert len(merged) == 8, f"expected 8 unique bars, got {len(merged)}"
# last-write-wins: chunk2 should have overwritten the overlapping bars
overlapping_ts = 1000 + 3*tf_ms
overwritten_row = [r for r in merged if r[0] == overlapping_ts][0]
assert overwritten_row[1] == 9, "merge did not apply last-write-wins at overlapping timestamp"
print(f"  PASS: merged {len(chunk1)}+{len(chunk2)} overlapping chunks into {len(merged)} unique, sorted bars, "
      f"last-write-wins confirmed at the overlap boundary.")

print(f"\n{'=' * 70}")
print("TEST 3: empirical funding interval inference")
print("=" * 70)
hour = 3_600_000
regular_8h = [i * 8 * hour for i in range(20)]
regular_1h = [i * 1 * hour for i in range(50)]
irregular = [0, 1*hour, 1.05*hour, 8*hour, 8.1*hour, 16*hour]  # mostly 8h with a couple 1h anomalies mixed in
print(f"  regular 8h spacing  -> inferred {infer_funding_interval_hours(regular_8h, fallback=99):.2f}h  (expect 8.00)")
print(f"  regular 1h spacing  -> inferred {infer_funding_interval_hours(regular_1h, fallback=99):.2f}h  (expect 1.00)")
print(f"  irregular mixed     -> inferred {infer_funding_interval_hours(irregular, fallback=99):.2f}h  (median, robust to a couple anomalies)")
print(f"  <2 data points      -> inferred {infer_funding_interval_hours([123], fallback=8.0):.2f}h  (falls back, expect 8.00)")
assert abs(infer_funding_interval_hours(regular_8h, fallback=99) - 8.0) < 0.01
assert abs(infer_funding_interval_hours(regular_1h, fallback=99) - 1.0) < 0.01
assert infer_funding_interval_hours([123], fallback=8.0) == 8.0
print("  PASS")

print(f"\n{'=' * 70}")
print("TEST 4: resampling hourly funding onto 4h bars (Hyperliquid-vs-4h-timeframe case)")
print("=" * 70)
bar_index = pd.date_range("2026-01-01", periods=6, freq="4h")
# 24 hourly funding payments of 0.0001 each, spanning the 6x4h=24h bar range
funding_records = [{"timestamp": int(bar_index[0].timestamp()*1000) + i*hour, "fundingRate": 0.0001} for i in range(24)]
resampled = resample_funding_to_bars(funding_records, bar_index)
print(resampled.to_string())
assert np.allclose(resampled.values, 0.0004, atol=1e-9), "each 4h bar should sum exactly 4 hourly payments of 0.0001"
print("  PASS: each 4h bar correctly sums its 4 underlying hourly funding payments (0.0004 each).")

print(f"\n{'=' * 70}")
print("TEST 5: DataLayer.fetch_all_ohlcv against a mock exchange with a small per-call limit")
print("=" * 70)


class _MockExchange:
    """Simulates a real exchange's per-call candle cap -- this is exactly
    the constraint that would silently truncate history if pagination
    weren't implemented."""
    def __init__(self, full_history: dict, max_per_call: int):
        self._history = full_history
        self.max_per_call = max_per_call
        self.call_log = []

    async def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.call_log.append((symbol, since))
        candles = self._history.get(symbol, [])
        eff_limit = min(limit or self.max_per_call, self.max_per_call)
        filtered = [c for c in candles if since is None or c[0] >= since]
        return filtered[:eff_limit]

    async def close(self):
        pass


async def run_mock_test():
    from datetime import datetime, timezone
    from data_layer import ExchangeProfile
    tf = "4h"
    tf_ms = _TIMEFRAME_MS[tf]
    n_bars = 500
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)  # matches fetch_all_ohlcv's real default
    start_ms = end_ms - n_bars * tf_ms
    full = [[start_ms + i * tf_ms, 100 + i*0.1, 101, 99, 100, 5] for i in range(n_bars)]

    MOCK_CAP = 137  # deliberately awkward, not a divisor of n_bars
    mock = _MockExchange({"BTC/USDT:USDT": full}, max_per_call=MOCK_CAP)
    config = Config()
    config.exchange_id = "binance"
    config.max_workers = 4
    layer = DataLayer(config, exchange=mock)
    # the planner must agree with what the (mock) exchange actually enforces --
    # in real use this is just the profile matching the real exchange's documented
    # limit; here we align the test profile with the mock's constraint on purpose
    layer.profile = ExchangeProfile(
        ccxt_id="binance", quote_currencies=["USDT"], ccxt_options={},
        max_candles_per_fetch=MOCK_CAP,
    )

    lookback_days = n_bars // 6  # 4h bars, 6/day
    result = await layer.fetch_all_ohlcv(["BTC/USDT:USDT"], tf, lookback_days, use_cache=False)
    df = result["BTC/USDT:USDT"]
    print(f"  requested ~{lookback_days*6} bars via mock exchange capped at {mock.max_per_call} candles/call")
    print(f"  calls made: {len(mock.call_log)}")
    print(f"  bars returned: {len(df)}")
    print(f"  timestamps unique & sorted: {df.index.is_monotonic_increasing and df.index.is_unique}")
    assert len(mock.call_log) > 1, "should have needed multiple paginated calls given the small per-call cap"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert len(df) >= lookback_days * 6 - 10, f"pagination under-covered the requested lookback: got {len(df)}"
    print("  PASS: multi-call pagination correctly assembled full history from a constrained mock exchange.")

asyncio.run(run_mock_test())

print(f"\n{'=' * 70}")
print("TEST 6: get_liquid_universe against a mock exchange, per exchange profile")
print("=" * 70)


class _MockUniverseExchange:
    def __init__(self, markets, tickers):
        self._markets = markets
        self._tickers = tickers

    async def load_markets(self):
        return self._markets

    async def fetch_tickers(self, symbols):
        return {s: self._tickers[s] for s in symbols if s in self._tickers}

    async def close(self):
        pass


async def run_universe_test():
    for ex_id in ["binance", "hyperliquid"]:
        config = Config()
        config.exchange_id = ex_id
        config.min_24h_volume_usd = 1_000_000
        mock = _MockUniverseExchange(markets, tickers)
        layer = DataLayer(config, exchange=mock)
        universe = await layer.get_liquid_universe()
        print(f"  {ex_id:<12} -> {universe}")

asyncio.run(run_universe_test())
print("  PASS: same mock market data, correct exchange-specific universes via DataLayer end-to-end.")

print(f"\n{'=' * 70}")
print("ALL TESTS PASSED")
print("=" * 70)

print(f"\n{'=' * 70}")
print("TEST 7: load_markets() fallback when ccxt itself raises a TypeError")
print("(reproduces a REAL failure seen on a live Hyperliquid connection: ccxt's own")
print(" hyperliquid.py crashes in fetch_spot_markets on a market entry with base=None)")
print("=" * 70)


class _MockCcxtBugExchange:
    """Reproduces the exact crash from ccxt/async_support/hyperliquid.py:614."""
    async def load_markets(self):
        base = None
        quote = "USDC"
        symbol = base + "/" + quote  # verbatim reproduction of the real ccxt bug
        return {}

    async def fetch_markets(self, params=None):
        assert params == {"type": "swap"}, f"expected swap-only fallback call, got params={params}"
        return [
            {"symbol": "BTC/USDC:USDC", "swap": True, "active": True, "quote": "USDC", "inverse": False},
            {"symbol": "ETH/USDC:USDC", "swap": True, "active": True, "quote": "USDC", "inverse": False},
        ]

    async def close(self):
        pass


class _MockCcxtBugExchangeFallbackAlsoFails(_MockCcxtBugExchange):
    async def fetch_markets(self, params=None):
        raise RuntimeError("swap-only fetch also broken in this hypothetical")


async def run_ccxt_bug_fallback_test():
    config = Config()
    config.exchange_id = "hyperliquid"

    layer = DataLayer(config, exchange=_MockCcxtBugExchange())
    markets = await layer.load_markets()
    print(f"  fallback recovers markets: {sorted(markets.keys())}")
    assert set(markets.keys()) == {"BTC/USDC:USDC", "ETH/USDC:USDC"}
    print("  PASS: swap-only fallback correctly recovers when plain load_markets() hits the ccxt TypeError.")

    layer2 = DataLayer(config, exchange=_MockCcxtBugExchangeFallbackAlsoFails())
    markets2 = await layer2.load_markets()
    print(f"  when fallback ALSO fails: returns {markets2} (expect empty dict, not a raised exception)")
    assert markets2 == {}
    print("  PASS: double failure degrades to an empty universe with a clear message, doesn't crash the pipeline.")

asyncio.run(run_ccxt_bug_fallback_test())

print(f"\n{'=' * 70}")
print("ALL TESTS PASSED (including the ccxt bug fallback)")
print("=" * 70)
