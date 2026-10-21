# Crypto stat-arb research pipeline

Rewrite of the original single-file pairs scanner. The rewrite is organized
around one idea: **a discovery p-value should never be the sole gate between
"tradeable" and "not."** Everything else follows from that.

## Two-layer validation

**Layer 1 -- pair_selection.py (statistical pre-filter, deliberately lenient)**
Log-price ADF prescreen + Engle-Granger cointegration + Benjamini-Hochberg
FDR correction across every pair tested in one pass, at a lenient default
alpha (0.10). `validate_fdr_concept.py` proves two things empirically on
synthetic data with known ground truth: (1) the original script's flat
p<0.05-per-pair approach produces real false discoveries at roughly the rate
you'd expect from testing hundreds of pairs with no correction, and (2) a
*strict* FDR correction (alpha=0.05) at a realistic universe size can erase
genuinely cointegrated pairs along with the noise, because one p-value
threshold is being asked to do more separating work than it can reliably do
at scale. Run it: `python validate_fdr_concept.py`.

**Layer 2 -- backtest.py (the actual arbiter)**
Multi-fold walk-forward simulation with a 1-bar execution lag, transaction
costs on both legs (weighted by hedge ratio), and funding-rate differential
as a first-class P&L input. Realized OOS Sharpe across pairs gets its own
multiple-testing correction (block-bootstrap p-value + FDR) -- a simplified
stand-in for Bailey & Lopez de Prado's deflated Sharpe ratio. A pair only
reaches `portfolio.py` sizing if it clears *both* layers.

## What was actually tested vs. not

This was built in a sandboxed environment with no network access, so
statsmodels and ccxt could not be installed. That constrains what could be
verified before handing this to you:

| File | Tested here? | How |
|---|---|---|
| `backtest.py` | **Yes** | `validate_backtest.py` -- true pair vs. 6 noise pairs, cost-sensitivity monotonicity, funding sign convention (deterministic check), Layer-2 FDR correctly separating signal from noise |
| `portfolio.py` | **Yes** | Same script -- Kelly sizing, vol-targeting, and a deliberately-concentrated 3-pair example confirming the concentration flag fires correctly |
| `data_layer.py` (pure functions + pagination/universe logic) | **Yes** | `validate_data_layer.py` -- quote-currency/inverse filtering per exchange profile, OHLCV pagination against a mock exchange with a deliberately small per-call cap, chunk-merge dedup at overlap boundaries, empirical funding-interval inference, hourly-funding-onto-4h-bars resampling, and `DataLayer.fetch_all_ohlcv`/`get_liquid_universe` run end-to-end against mock exchanges (not real ccxt) |
| `data_layer.py` (actual ccxt calls: `fetch_ohlcv`, `load_markets`, `fetch_funding_rate_history`, exchange auth) | **No** | Needs a real or testnet connection to Binance/Bybit/Hyperliquid; the Hyperliquid profile in particular (auth pattern, real per-call candle limit) is unverified -- see its entry in `EXCHANGE_PROFILES` |
| `validate_fdr_concept.py` | **Yes** | Self-contained (numpy-only), includes its own sanity check that p-values are ~Uniform(0,1) under a true null |
| `pair_selection.py` | **No** | Needs statsmodels; logic mirrors the validated concept script with statsmodels swapped in for properly-calibrated p-values (Monte Carlo p-values don't resolve the tail multiple-testing needs at real universe scale -- see validate_fdr_concept.py's own output for why) |

## Switching exchanges

`config.exchange_id` selects one of `binance` / `bybit` / `hyperliquid`
(`data_layer.EXCHANGE_PROFILES`). Each profile encodes the exchange-specific
details that a single `exchange_id` string can't safely paper over:

- **Quote currency** -- Hyperliquid perps settle in USDC, not USDT. A
  universe filter hardcoded to USDT (which the original script had) doesn't
  error on Hyperliquid, it just silently returns zero symbols.
- **Inverse contracts** -- Bybit's `swap` market type mixes linear
  (USDT-margined) and inverse (coin-margined) contracts. Inverse P&L is
  denominated in the base asset, not USD -- letting one into the universe
  wouldn't crash anything, it would just silently corrupt the spread P&L
  math in `backtest.py`, which assumes USD-quoted returns throughout.
- **Per-call candle limits** -- differ by exchange, so `fetch_all_ohlcv`
  paginates (`plan_ohlcv_pagination` + `merge_ohlcv_chunks`) rather than
  trusting one `fetch_ohlcv` call to return a full lookback window.
- **Funding interval** -- most CEXs are ~8h, Hyperliquid defaults to hourly,
  and this isn't a permanent fact about any exchange (schedules have
  changed before). Rather than hardcode it, `infer_funding_interval_hours`
  computes the actual interval from consecutive funding timestamps, and
  `resample_funding_to_bars` correctly sums multiple funding payments into
  each trading bar when the two intervals don't match (e.g. hourly funding
  against 4h trading bars).

Adding a new exchange means adding a new `ExchangeProfile` entry with these
filled in deliberately, not just dropping a new ccxt id into a string field.

Run `python main.py demo` first -- it exercises the fully-tested path
(backtest + portfolio, on synthetic data) with zero external dependencies
beyond numpy/pandas, so you have a known-good reference before pointing
`python main.py scan` at a real exchange.

## Real limitations, stated plainly

- **This is a research/signal pipeline, not an execution system.** No order
  management, no position reconciliation, no live risk controls. A real
  trading firm treats those as a separate, more carefully engineered layer
  built after the research signal is trusted -- not a follow-on feature of
  the same script.
- **Kelly sizing requires 30+ pooled OOS trades before it engages**
  (`portfolio.py`), by design -- a 100% hit rate over 12 trades is a
  small-sample artifact, not an edge you should size against. Until a pair
  accumulates enough trade history, it's correctly sized to zero even after
  passing both statistical layers.
- **Funding rates must be supplied**, not fabricated. `data_layer.py`
  exposes `get_funding_rate_history`; nothing downstream invents a number if
  you don't wire it through.
- **Hedge ratio refits periodically** (`config.beta_refit_every_bars`, ~weekly
  by default at 4h bars), not every bar -- a per-bar-adaptive hedge ratio
  back tests beautifully and is not something you can actually execute
  without constant rebalancing costs.

## Usage

```
pip install -r requirements.txt
python main.py demo                          # synthetic, no network needed
python main.py scan --exchange binance --timeframe 4h --lookback-days 120
python main.py scan --exchange bybit --timeframe 4h
python main.py scan --exchange hyperliquid --timeframe 1h   # unverified profile -- smoke-test first
```
