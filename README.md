# Crypto Perpetual Pairs: bias-conscious stat-arb refactor

**Research only. Zero live exchange order placement. No real market dataset was supplied.**

Refactors the supplied `backtest.py`, `pair_selection(1).py`, `data_layer.py`, `portfolio.py`, `main.py`, `config.py` and validation scripts; untouched copies are under `legacy/`. The new code fixes major modeling errors rather than rubber-stamping existing retrospective Sharpe scores. See [RESEARCH_MEMO.md](RESEARCH_MEMO.md) for the hypothesis fixed **before** testing the new implementation.

## Design decisions

- **Cointegrated crypto-perp pairs** instead of single-asset indicators: preserves the project intent and economically meaningful two-leg hedging.
- **Causal periodic OLS**, not Kalman: more auditable than continuously changing beta; adding Kalman tuning would be an additional research experiment, not a free upgrade. The hedge is estimated on LOG price levels, *but P&L uses dollar notional weights and both actual OHLC opens*. A log spread's point difference must NEVER be treated as dollars earned.
- **Custom minimal event-style loop**, not vectorbt: sequential entries, exits, funding and kill-switches cannot be represented accurately by a naive shifted Series. This is a research simulator, **not Backtrader/NautilusTrader live parity**.
- **Strict provenance / PIT inclusion** with a point-in-time membership and 24h-volume / OI snapshot, funding events and linear-perp venue manifest. Missing data aborts instead of being repaired with forward fills. Data before and after an asset's listed life are permitted (no current-constituent restriction); a selected asset missing a bar during evaluation **invalidates** that run instead of silently dropping the losing trade.
- **No tuning sweep**: one preregistered set of signal parameters, source logged. Holdout must be explicitly opened once and is locked in the report folder. Running in a different folder bypasses file locks, so operational discipline and immutable dataset hashes matter.
- **Deterministic Python tests** run without networking or brokerage credentials.

## Install

Python 3.11+ (tested with installed 3.11 environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
statarb demo --out reports/demo                # mechanical demo only
statarb demo --out reports/synthetic-holdout --open-holdout
```

The demo generates *synthetic*, artificially mean-reverting prices. **Any Sharpe there is manufactured ground truth, not a prediction or validation of a crypto strategy.** Never use it to select pairs. The `--open-holdout` command is for testing output plumbing on synthetic data only.

## Real research dataset format

`statarb research --data /path/to/historical_export --out reports/real` performs **development-only** walk-forward. Once you have frozen hypothesis, dataset and risk configuration, use `--open-holdout` exactly once with the same `--out` path. After seeing holdout, do not change parameters; a new test requires a preregistered new experiment and genuinely new unseen data.

Directory layout:

```text
historical_export/
  manifest.json
  membership.csv
  bars/
    BTC_USDT.csv
    ETH_USDT.csv
    ...including historically-delisted assets...
  funding/
    BTC_USDT.csv
    ETH_USDT.csv
```

`manifest.json` example (YOU certify each true statement; don't invent coverage):

```json
{
  "venue": "binance",
  "contract_type": "linear_perpetual",
  "quote_currency": "USDT",
  "funding_complete": true,
  "universe_point_in_time": true,
  "benchmark_symbol": "BTC_USDT",
  "data_vendor": "your-pit-vendor",
  "extracted_at_utc": "2026-10-19T00:00:00Z"
}
```

- Bars CSV: `timestamp,open,high,low,close,volume`, timestamps ISO8601 with `Z`/UTC (candle START time), one complete bar every 4h throughout **each instrument's actual listed life**. Quote/size volume in **base units**, as returned by the vendor; value turnover estimated from price × base volume. No missing candles permitted within a listed span; no incomplete latest candle.
- Funding CSV: `timestamp,rate`, timestamp ISO8601 UTC of each **realized settlement**; positive = long pays short. A `funding_complete:true` declaration is an explicit provenance assertion that the provider contains **ALL** events in the requested date range. The loader verifies formatting, not external exchange completeness.
- Membership CSV: `timestamp,symbol,active,linear,quote,open_interest_usd,volume_24h_usd`. Each timestamp is a **complete snapshot of all historically known markets**, with the financial fields as measured *at the snapshot timestamp*, including delisting decisions. Snapshots must be no more than 7 days apart. Missing/delisted assets should appear in relevant older snapshots. Quoted market universe is NOT inferred from current CCXT markets.
- All symbols in `bars/` must belong to the same exchange and settlement type. **Do not mix inverse perps, spot, USDT/USDC settlement or different venues.**
- Startup manifest and snapshot accuracy are **not** independently verified by this library. An inaccurate manifest can still bias results. Independent audits remain required.
- Short sale borrow costs of spot are irrelevant to linear perps, but trading fees, impact, and funding are modeled.

## Exact execution convention

1. At close of 4h candle *t*, update the rolling statistic and, on schedule, beta using data known no later than *t*.
2. Decision executes at **open t+1**; target = `direction * gross * (1,-beta)/(1+abs(beta))` in USD equity-normalized notionals, where `gross<=20%` for each pair and vol-targeted at 10% annualized.
3. Mark open(t+1)→open(t+2) using simple returns on BOTH legs. Actual exchange trades/fills not known, so add fee + volatility and bar-size-dependent slippage on **changes to each leg's position**, with 1% maximum order participation assumption.
4. Debit/rebate realized timestamped funding on both signed legs. Funding is not guessed zero when provider omitted records.
5. Flat/entry/exit state machine plus z-stop (4), 90-bar time stop, abnormal spread-vol filter. Stops execute **one opening later**, not as magical intrabar guaranteed orders.
6. Force exit at last in-window OPEN, charging fees; do not mark beyond evaluation slice.
7. Roll development in nonoverlapping 30-day segments; discover candidate pairs independently from the 180-day past ending strictly before each segment. Holdout uses one locked pair selection made at the split, not any look-ahead discovery inside holdout.

### Measurement and safeguards

- Logs trial count, EG FDR p-values, coefficient stability, trades, costs, realized funding, gross/net returns, exposures and expected fill prices.
- Explicitly separates development (rolling walk-forward) and untouched holdout (30% of chronological observations).
- Reports net Sharpe, Sortino, Calmar, worst drawdown with dates, max duration, profit factor, hit rate, average win/loss, annual turnover, cost drag, monthly heatmap and per-fold results. Fields are null / N/A where statistically unsupported, **not auto-passed**.
- **Not a genuine Deflated Sharpe Ratio**: complete research-trial genealogy / effective trial count isn't available. A reported high synthetic Sharpe is NOT sufficient for paper-trading approval.
- Correlation and concentration control: first stage forbids multiple selected pairs sharing the same underlying; caps max pair 20%, portfolio 100% and max single asset 30% equity gross. Per-pair volatility target uses historical completed-bar data.
- Max DD kill switch persists on disk for paper; model-wide simulator flattens at the following open after threshold breach (with liquidation cost approximation), not the violating price.

## Live/paper integration boundary

`statarb.paper_reconcile.reconcile_csv()` imports an externally exported simulator-fill CSV (never guesses fills). `statarb.paper.PaperLedger` stores simulated fills and checks persistent kill-switch conditions. **It is not connected to Binance/Bybit/Hyperliquid and does not generate or transmit live orders.** See [PAPER_RUNBOOK.md](PAPER_RUNBOOK.md) for minimum 30-session external broker/testnet paper procedure, alarms, kill-switch, and rollback.

The previous `data_layer.py` is preserved only for reference in `legacy/`: it fetches present-day universes, caches with pickle and has no historical universe snapshots or complete funding history certification. Do **not** plug it into the new strict engine without rebuilding provenance and event reconciliation.

## Project layout

```text
src/statarb/
  config.py      # immutable preregistered defaults
  data.py        # UTC OHLCV, delisting-safe PIT snapshot, realized funding loading
  selection.py   # formation-only Engle-Granger, BH FDR, stability screening
  signals.py     # causal signals and state machine
  execution.py   # both-leg executable-notional P&L & costs, event funding
  risk.py        # no shared legs, exposure caps, persistent paper kill
  research.py    # nonoverlap WF dev + once-only locked holdout
  report.py      # metrics, net equity, monthly heatmap
  paper.py       # simulated fill ledger, health checks only
  synthetic.py   # explicitly fake market for regressions
  cli.py         # demo/research CLI
notebooks/research.ipynb
legacy/          # original upload snapshot, NOT safe to trade directly
PAPER_RUNBOOK.md
RESEARCH_MEMO.md
```

## Reproducibility caveats

- Tested locally on deterministic fixtures; **no live exchange connection tested** and **no actual historical crypto trade sample supplied**, so no empirically validated positive edge or real IS/OOS report can be claimed.
- No intrabar stop promises, latency-based queue priority, exact L2 order-book market impact, catastrophic gap guarantees, liquidation/margin modeling or partial-fills engine. A 4h strategy has no credible need to promise VPS colocation; reliable data/time sync and risk controls matter more than microseconds.
- Not enough validated data to establish a Deflated OOS Sharpe > 1; paper authorization remains blocked.
- If historical timestamps reference candle *close* rather than *start*, shift them in your import process using the exchange's documented convention before loading.
