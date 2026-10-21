"""
Async OHLCV + funding-rate fetching via ccxt, across Binance / Bybit /
Hyperliquid perpetuals.

IMPORTANT: the ccxt-dependent code paths here (anything using a real
exchange, not the injected _MockExchange in validate_data_layer.py) could
not be executed in the sandbox used to build this -- no network access to
install ccxt, and none available. What COULD be tested, and was:
  - filter_liquid_symbols, plan_ohlcv_pagination, merge_ohlcv_chunks,
    infer_funding_interval_hours, resample_funding_to_bars -- all pure
    functions, no exchange needed, exercised directly with synthetic input.
  - DataLayer.fetch_all_ohlcv's PAGINATION LOGIC end-to-end, against a mock
    exchange (_MockExchange in validate_data_layer.py) that simulates a
    per-call candle limit smaller than the requested lookback -- this is
    exactly the failure mode that would silently truncate history on an
    exchange with tight per-call limits (Hyperliquid, or Bybit under some
    endpoint versions) if pagination weren't implemented.
See validate_data_layer.py for what actually ran and what it found.

Why a single ExchangeProfile per venue, not just an exchange_id string:
  - Hyperliquid perpetuals settle in USDC, not USDT. A universe filter
    hardcoded to quote=='USDT' (which the original script had) silently
    returns ZERO Hyperliquid symbols -- not an error, just an empty
    universe that looks like "no liquid symbols today."
  - Bybit's ccxt 'swap' market type includes both linear (USDT-margined)
    and inverse (coin-margined) contracts. Inverse P&L is denominated in
    the base asset, not USD -- if one slips into the universe, the entire
    spread P&L calculation in backtest.py (which assumes USD-quoted
    returns) is silently wrong, not obviously broken. Both quote-currency
    AND inverse-exclusion filters matter here, not just one.
  - Exchanges cap how many candles you get per fetch_ohlcv call, and the
    cap differs (Binance ~1500, Bybit ~1000, Hyperliquid unconfirmed here
    -- profile value is a conservative placeholder, verify it). Request a
    lookback longer than one call's worth and you silently get truncated,
    recent-only history with no error, on whichever exchange has the
    smaller limit. Fixed via real pagination (plan_ohlcv_pagination +
    merge_ohlcv_chunks), not a single fetch_ohlcv call.
  - Funding intervals differ AND change over time (most CEXs ~8h, but this
    isn't universal or permanent; Hyperliquid is hourly by default). Rather
    than hardcode per-exchange constants and trust them,
    infer_funding_interval_hours computes the actual interval from
    consecutive funding timestamps, falling back to the profile default
    only when there isn't enough history to infer it.

Bugs fixed vs. the original script (unrelated to multi-exchange support,
carried over from the first rewrite pass):
  1. `except ccxt.NetworkError` referenced an unimported bare `ccxt` name
     (only `ccxt.async_support` and the exception classes were imported) --
     NameError exactly when the handler was needed. Fixed: import and catch
     the exception classes directly.
  2. `self.config.exchange` was logged but never existed on Config
     (`exchange_id` does). Fixed.
  3. The disk cache only checked symbol coverage, not whether cached data
     matched the requested time range/exchange -- silently served stale
     data across reruns or across DIFFERENT EXCHANGES with the same symbol
     names. Fixed: cache key now includes exchange_id, symbols, timeframe,
     and bar count.
  4. Signal-generation defaults read from the Config CLASS rather than the
     passed instance, so changing config at runtime silently had no effect.
     Fixed in pair_selection.py / backtest.py, not here, but noted since
     it's the same class of bug.
"""

import asyncio
import hashlib
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config

try:
    from ccxt.async_support import NetworkError, ExchangeError
except ImportError:
    # ccxt isn't installed in this sandbox. Placeholder exception types so
    # the module (and everything testable in it) still imports and runs --
    # a mock exchange in tests won't raise ccxt-specific errors anyway. When
    # ccxt IS installed, the real import above wins and real errors are
    # caught correctly.
    class NetworkError(Exception):
        pass

    class ExchangeError(Exception):
        pass

CACHE_DIR = Path("./ohlcv_cache")

_TIMEFRAME_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


# ---------------------------------------------------------------------------
# Exchange profiles
# ---------------------------------------------------------------------------

@dataclass
class ExchangeProfile:
    ccxt_id: str
    quote_currencies: List[str]
    ccxt_options: dict
    exclude_inverse: bool = True
    max_candles_per_fetch: int = 1000
    fallback_funding_interval_hours: float = 8.0
    notes: str = ""


EXCHANGE_PROFILES: Dict[str, ExchangeProfile] = {
    "binance": ExchangeProfile(
        ccxt_id="binance",
        quote_currencies=["USDT"],
        ccxt_options={"defaultType": "future"},
        exclude_inverse=True,
        max_candles_per_fetch=1500,
        fallback_funding_interval_hours=8.0,
        notes="USDT-margined linear perpetuals (ccxt defaultType='future'). "
              "COIN-M inverse contracts excluded.",
    ),
    "bybit": ExchangeProfile(
        ccxt_id="bybit",
        quote_currencies=["USDT"],
        ccxt_options={"defaultType": "swap", "defaultSubType": "linear"},
        exclude_inverse=True,
        max_candles_per_fetch=1000,
        fallback_funding_interval_hours=8.0,
        notes="Bybit's 'swap' market type mixes linear and inverse contracts -- "
              "defaultSubType='linear' AND exclude_inverse both matter, redundantly "
              "on purpose (belt and suspenders on a mistake that silently corrupts P&L).",
    ),
    "hyperliquid": ExchangeProfile(
        ccxt_id="hyperliquid",
        quote_currencies=["USDC"],
        ccxt_options={"defaultType": "swap"},
        exclude_inverse=True,
        max_candles_per_fetch=500,
        fallback_funding_interval_hours=1.0,
        notes="USDC-settled, not USDT -- would silently produce an empty universe "
              "under a hardcoded USDT filter. Funding is hourly by default (this "
              "pipeline infers the real interval from data rather than trusting this "
              "constant -- see infer_funding_interval_hours). UNVERIFIED against a "
              "live connection: max_candles_per_fetch is a conservative placeholder, "
              "and Hyperliquid's ccxt auth pattern (wallet address / private key, not "
              "traditional apiKey+secret) is untested here. Public market-data calls "
              "typically don't require auth, but confirm before relying on it.",
    ),
}


def get_profile(exchange_id: str) -> ExchangeProfile:
    profile = EXCHANGE_PROFILES.get(exchange_id)
    if profile is None:
        raise ValueError(
            f"No ExchangeProfile for '{exchange_id}'. Known: {list(EXCHANGE_PROFILES)}. "
            f"Add one to EXCHANGE_PROFILES before pointing this pipeline at a new exchange -- "
            f"quote currency and inverse-contract handling are exchange-specific and this "
            f"pipeline should not guess at them."
        )
    return profile


# ---------------------------------------------------------------------------
# Pure, testable logic (no exchange object needed)
# ---------------------------------------------------------------------------

def filter_liquid_symbols(markets: dict, tickers: dict, profile: ExchangeProfile,
                           min_volume_usd: float) -> List[str]:
    liquid = []
    for m in markets.values():
        if not m.get("swap") or not m.get("active"):
            continue
        if m.get("quote") not in profile.quote_currencies:
            continue
        if profile.exclude_inverse and m.get("inverse"):
            continue
        sym = m["symbol"]
        t = tickers.get(sym, {})
        if (t.get("quoteVolume") or 0) >= min_volume_usd:
            liquid.append(sym)
    return liquid


def plan_ohlcv_pagination(total_bars: int, max_candles_per_fetch: int,
                           timeframe_ms: int, end_time_ms: int) -> List[int]:
    """Returns `since` timestamps (ms), oldest first, for enough successive
    fetch_ohlcv calls to cover total_bars ending at end_time_ms."""
    if total_bars <= 0 or max_candles_per_fetch <= 0:
        return []
    n_chunks = math.ceil(total_bars / max_candles_per_fetch)
    start_time = end_time_ms - total_bars * timeframe_ms
    return [start_time + i * max_candles_per_fetch * timeframe_ms for i in range(n_chunks)]


def merge_ohlcv_chunks(chunks: List[list]) -> list:
    """Dedups by timestamp (last write wins) and sorts -- pagination chunks
    commonly overlap by one bar at the boundary."""
    seen = {}
    for chunk in chunks:
        for row in chunk:
            seen[row[0]] = row
    return [seen[k] for k in sorted(seen.keys())]


def infer_funding_interval_hours(funding_timestamps_ms: List[float], fallback: float) -> float:
    """Median gap between consecutive funding events, not a hardcoded
    per-exchange constant -- intervals differ by exchange AND change over
    time (exchanges have altered funding schedules before). Falls back only
    when there's not enough history to infer anything."""
    if len(funding_timestamps_ms) < 2:
        return fallback
    ts = np.array(sorted(funding_timestamps_ms), dtype=float)
    diffs_hours = np.diff(ts) / (1000 * 3600)
    diffs_hours = diffs_hours[diffs_hours > 0]
    if len(diffs_hours) == 0:
        return fallback
    return float(np.median(diffs_hours))


def resample_funding_to_bars(funding_records: List[dict], bar_index: pd.DatetimeIndex) -> pd.Series:
    """Sums funding payments falling within each OHLCV bar's interval.
    Matters whenever the funding interval doesn't equal the trading
    timeframe -- e.g. hourly Hyperliquid funding against 4h trading bars,
    where each bar should reflect ~4 summed funding payments, not one."""
    if len(bar_index) == 0:
        return pd.Series(dtype=float)
    if not funding_records:
        return pd.Series(0.0, index=bar_index)
    df = pd.DataFrame(funding_records)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("ts")
    bar_step = bar_index[1] - bar_index[0] if len(bar_index) > 1 else pd.Timedelta(hours=4)
    edges = list(bar_index) + [bar_index[-1] + bar_step]
    binned = pd.cut(df["ts"], bins=edges, labels=bar_index, right=False)
    summed = df.groupby(binned, observed=True)["fundingRate"].sum()
    return summed.reindex(bar_index).fillna(0.0)


def _cache_key(exchange_id: str, symbols: List[str], timeframe: str, total_bars: int) -> str:
    raw = f"{exchange_id}|{sorted(symbols)}|{timeframe}|{total_bars}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Exchange I/O (untestable here -- see module docstring)
# ---------------------------------------------------------------------------

class DataLayer:
    def __init__(self, config: Config, exchange=None):
        """exchange: inject a pre-built client (real ccxt, or a test mock).
        Mainly here for testability -- validate_data_layer.py injects
        _MockExchange to exercise the pagination/assembly logic without ccxt."""
        self.config = config
        self.profile = get_profile(config.exchange_id)
        if exchange is not None:
            self.exchange = exchange
        else:
            import ccxt.async_support as ccxt_async
            exchange_cls = getattr(ccxt_async, self.profile.ccxt_id)
            self.exchange = exchange_cls({
                "enableRateLimit": True,
                "options": dict(self.profile.ccxt_options),
            })

    async def close(self):
        close_fn = getattr(self.exchange, "close", None)
        if close_fn is not None:
            await close_fn()

    async def load_markets(self) -> dict:
        try:
            return await self.exchange.load_markets()
        except (NetworkError, ExchangeError) as e:
            print(f"[{self.config.exchange_id}] failed to load markets: {e}")
            return {}

    async def get_liquid_universe(self) -> List[str]:
        markets = await self.load_markets()
        if not markets:
            return []
        candidate_symbols = [
            m["symbol"] for m in markets.values()
            if m.get("swap") and m.get("active")
            and m.get("quote") in self.profile.quote_currencies
            and not (self.profile.exclude_inverse and m.get("inverse"))
        ]
        if not candidate_symbols:
            return []
        try:
            tickers = await self.exchange.fetch_tickers(candidate_symbols)
        except (NetworkError, ExchangeError) as e:
            print(f"[{self.config.exchange_id}] failed to fetch tickers: {e}")
            return []
        return filter_liquid_symbols(markets, tickers, self.profile, self.config.min_24h_volume_usd)

    async def fetch_symbol_ohlcv(self, symbol: str, timeframe: str, total_bars: int,
                                  end_time_ms: Optional[int] = None) -> list:
        tf_ms = _TIMEFRAME_MS.get(timeframe)
        if tf_ms is None:
            raise ValueError(f"Unknown timeframe '{timeframe}' -- add it to _TIMEFRAME_MS.")
        if end_time_ms is None:
            end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_list = plan_ohlcv_pagination(total_bars, self.profile.max_candles_per_fetch, tf_ms, end_time_ms)
        chunks = []
        for since in since_list:
            try:
                raw = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since,
                                                        limit=self.profile.max_candles_per_fetch)
            except (NetworkError, ExchangeError) as e:
                print(f"[{symbol}] fetch_ohlcv failed (since={since}): {e}")
                continue
            if raw:
                chunks.append(raw)
        merged = merge_ohlcv_chunks(chunks)
        return merged[-total_bars:] if len(merged) > total_bars else merged

    async def fetch_all_ohlcv(self, symbols: List[str], timeframe: str, lookback_days: int,
                               use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 6)
        total_bars = lookback_days * bars_per_day

        CACHE_DIR.mkdir(exist_ok=True)
        key = _cache_key(self.config.exchange_id, symbols, timeframe, total_bars)
        cache_path = CACHE_DIR / f"{key}.pkl"
        if use_cache and cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if set(symbols).issubset(cached.keys()):
                return {s: cached[s] for s in symbols}

        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def _fetch(sym):
            async with semaphore:
                raw = await self.fetch_symbol_ohlcv(sym, timeframe, total_bars)
                if not raw:
                    return sym, None
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("timestamp")
                return sym, df

        results = await asyncio.gather(*[_fetch(s) for s in symbols])
        out = {sym: df for sym, df in results if df is not None and len(df) >= self.config.min_data_points}

        if use_cache and out:
            with open(cache_path, "wb") as f:
                pickle.dump(out, f)
        return out

    async def get_open_interest_dollar(self, symbol: str) -> Optional[float]:
        try:
            oi = await self.exchange.fetch_open_interest(symbol)
            if oi.get("openInterestValue") is not None:
                return oi["openInterestValue"]  # already USD-quoted where the exchange provides it directly
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or 0
            contracts = oi.get("openInterestAmount") or oi.get("openInterest") or 0
            return contracts * price
        except (NetworkError, ExchangeError) as e:
            print(f"[{self.config.exchange_id}] open interest fetch failed for {symbol}: {e}")
            return None

    async def get_funding_rate_history(self, symbol: str, since: Optional[int] = None,
                                        limit: int = 500) -> list:
        try:
            return await self.exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
        except (NetworkError, ExchangeError) as e:
            print(f"[{symbol}] funding rate history fetch failed: {e}")
            return []

    async def get_funding_series(self, symbol: str, bar_index: pd.DatetimeIndex) -> pd.Series:
        """Per-symbol funding series resampled onto the OHLCV bar grid --
        this is the shape backtest.walk_forward's funding_panel expects
        (it computes the beta-weighted diff itself per pair); see
        resample_funding_to_bars for why resampling matters whenever the
        funding interval doesn't match the trading timeframe."""
        history = await self.get_funding_rate_history(symbol)
        return resample_funding_to_bars(history, bar_index)


async def build_universe_and_data(config: Config) -> Dict[str, pd.DataFrame]:
    layer = DataLayer(config)
    try:
        universe = await layer.get_liquid_universe()
        if not universe:
            print(f"No symbols passed the liquidity filter on {config.exchange_id} "
                  f"(profile quote currencies: {layer.profile.quote_currencies}). "
                  f"Check min_24h_volume_usd / exchange connectivity / that the profile is right.")
            return {}
        ohlcv = await layer.fetch_all_ohlcv(universe, config.timeframe, config.lookback_days)
        return ohlcv
    finally:
        await layer.close()
