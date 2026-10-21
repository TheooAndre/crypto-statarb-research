# Engineering validation report — 2026-10-19

## Ground truth versus performance evidence

**No historical crypto OHLCV, funding, delisting records, or broker fills were uploaded.** The user-provided `backtest.py`, `pair_selection.py`, `data_layer.py`, etc. are code, not a point-in-time market dataset. Therefore real-market IS, OOS, holdout, walk-forward profitability, real cost drag and live Sharpe **cannot honestly be computed**. The refactor makes no positive-edge claim.

## Concrete original defects addressed

| Original concern | Refactor action |
|---|---|
| `backtest.py`: `spread.diff()` of log prices used as realized P&L | Execution uses signed USD notional weights times **two observed open-to-open simple returns**; opening on following bar only. |
| Prior `walk_forward` tested a globally shortlisted pair in all historical folds, including periods before the shortlist was known | New `discover()` reruns using only each fold's past formation; holdout shortlist locks at the split. |
| Original `step_days < trading_days` could double-count 4h observations across folds | Nonoverlapping test windows; uniqueness asserted and tested. |
| No final untouched chronological 30% holdout | 70/30 split; explicit one-time `--open-holdout` command with refusal on second run in same result directory. |
| Funding missing may be silently treated as zero | Manifest attestation + individual symbol event series required; absence fails closed. Completeness attestation remains externally auditable. |
| Lookback 120 days may be too short for multiple robust formation and testing folds | Default 180-day formation + 30-day disjoint folds; fails clearly on insufficient history; recommend 2+ years. |
| Present-day liquidity universe and missing delisted markets | Point-in-time membership + USD open interest and volume snapshots, multiple historic listings/delistings in data model. |
| Kelly sizing from noisy small sample and merely warning on concentration | Fixed vol target, 20% pair gross cap, 30% per asset, no shared-leg selected pairs and portfolio drawdown stop. |
| Flat 2bps slippage and simple two-leg fee approximation | Commissions per changed notional + historical volatility + order size/ADV model, hard 1% bar participation rejection. |
| Original statistical FDR at discovery + realized Sharpe BH described as surrogate DSR | True DSR is explicitly NOT claimed or computed without trial genealogy, skew/kurtosis and predeclared comparison benchmark. |
| Missing true position reconciliation and live risk controls | Persisted manual-reset paper kill switch; paper fill audit, no real exchange order integration. |

## Reproducible validation

Command:

```bash
pip install -e '.[test]'
pytest -q
statarb demo --out reports/demo
statarb demo --out reports/synthetic_holdout --open-holdout
```

**16 tests passed** in the build environment (offline; no ccxt credentials, no real venue data). Verified: data gaps rejected, future close mutation does not change prior signals, two-leg open execution, funding signs, cost monotonicity, trading caps, no overlapping folds, time-split separation, historical delistings represented, exit of window, persistent kill switch, paper fill journal, deliberate stop after portfolio drawdown, and holdout fixed-pair freeze.

The `reports/demo` and `reports/synthetic_holdout` subdirectories contain only **synthetic-process** runs. These runs must never be used for investment decisions. Some synthetic folds select no pairs, and the synthetic holdout produced no trades: correct abstention rather than manufacture of statistical evidence.

## Gating checklist

| Guard | Status |
|---|---|
| Hypothesis before market data | PASS: memo preregistered |
| Causal signals/open fills | PASS: deterministic unit tests (not exchange audit) |
| Nonoverlapping validation folds | PASS: unit and integration tests |
| Actual exchange fee schedule | UNVERIFIED: user needs their account-tier fee |
| Exchange-specific slippage realism | PROXY ONLY: needs actual order-book/fills |
| Complete timestamped funding | NOT PROVIDED |
| Historical delisted/PIT universe | NOT PROVIDED |
| Actual 70/30 market dataset | NOT PROVIDED |
| Deflated OOS Sharpe > 1 | NOT COMPUTABLE |
| Drawdown within risk tolerance on real market data | NOT COMPUTABLE |
| External venue paper sessions >=30 | NOT STARTED |
| Live readiness | **BLOCKED** |

## Interpretation

Strong performance in the synthetic demo **does not demonstrate a market edge**. The appropriate substantive result is: **no tradable edge established**. The refactor is an experimental apparatus, ready for a properly sourced dataset and independent paper integration, not a trade recommendation.
