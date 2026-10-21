# Preregistered research memo · H0: no tradable edge

**Date:** 2026-10-19. **Status:** design preregistration, not evidence of profitability.

## Hypothesis (specified before inspecting any new empirical market data)

Among sufficiently liquid *linear USD-quoted perpetuals* on the **same venue**, some economically related crypto assets have a common stochastic trend. Temporary relative mispricing arising from dealer inventory, ETF/spot-perp flows, cross-exchange lag, liquidation cascades, and constrained arbitrage capital may revert within days. We take the opposite side of temporarily imbalanced flow; in return we provide liquidity and bear tail risk, funding, and execution costs. There is no guarantee any pair is stably cointegrated or profitable. An OOS failure means **no edge**.

### Null hypothesis
After realistic costs, funding, point-in-time liquidity controls and holdout evaluation, expected net return <= 0. Reject only with stable walk-forward results and an untouched final holdout. No positive validation ⇒ no permission to paper trade.

## Strategy choice
**Option 3, pairs cointegration**, with Engle–Granger/BH discovery and a *piecewise-frozen* rolling OLS hedge. A Kalman filter is deliberately **not** used: the user's original code has periodic beta refits and the high-frequency hedge churn, tuning and mark-to-market accounting introduced by Kalman would need independent validation. The hedge is an elasticity on log-price levels, mapped into USD notionals (long $1 of Y, short $beta of X, normalized by 1+|beta|); not a dollar price difference. This is a model simplification, not a proof of equivalence to a self-financing cointegration spread.

### Signal, exact convention
At close t: s_t = ln(C_y,t) - alpha_t - beta_t ln(C_x,t). m_t and sd_t are rolling statistics of past/current observed spread, known only at close t. z_t=(s_t-m_t)/sd_t. At OPEN t+1 execute position direction -sign(z_t) when |z_t| >= 2, exit when |z_t| <= 0.5, |z_t| >= 4, or after 90 bars (15 days at 4h). Hedge fitted from completed historical candles before each refit; no data from the execution candle.

Long spread means long Y and short beta*X (where negative beta results in a same-direction position and is rejected by the selector). Open-to-next-open mark-to-market uses *both* legs' simple returns and normalized notionals; turnover charges each leg at execution; realized funding is timestamped, denominated in portfolio equity, and **never assumed zero** when the required funding feed is absent. New positions are blocked on missing/corrupt bars, stale data and negative volume. No intrabar stops are claimed: signal stops have the stated next-open reaction lag.

## Fixed rules (not grid searched)
- 4h bars, 180-calendar-day formation, 30-day non-overlapping validation, development 70%, final holdout 30%.
- Statistical formation: both individual log prices should behave I(1), Engle–Granger p-value with FDR q<=0.10 across **all pairs tested in the fold**, beta>0, two-subsample stability, 2<=half-life<=90 bars. ADF is a screening heuristic, not proof of I(1).
- Entry |z| >=2, mean exit |z|<=0.5, stop |z|>=4, time stop 90 4h bars.
- No parameter optimization in v1 (number of tried parameter variants = **1**). Number of eligible pair hypotheses/fold is logged; ordinary Sharpe is NOT a mathematically valid Deflated Sharpe Ratio.
- Portfolio: per-pair target annualized volatility 10%, max pair gross 20% equity; max portfolio gross 100%; max single asset gross 30% equity; max drawdown 10% with permanent human-reset kill switch. No Kelly estimates.

## Required data provenance
Historical OHLCV for *both legs including open and volume*, funding EVENTS with historical timestamps, **point-in-time market membership and delistings**, and historical 24h dollar-volume/open-interest snapshots for formation-time filtering (both REQUIRED). Today's filtered universe cannot be used to backtest earlier years without survivorship caveat. Exchange and margin/quote currencies must agree. Crypto perpetuals have no equity dividend/split adjustment, but contract migrations, redenominations and listings/delistings require point-in-time normalization. No OHLCV-only backtest is labeled executable if funding or historical universe snapshots are missing.

## Protocol
1. Review tests on synthetic stationary/nonstationary fixtures; these are **mechanical tests, not performance evidence**.
2. Obtain 2+ years of venue-specific 4h OHLCV + event funding + point-in-time universe, freeze hash/date and thresholds.
3. Run repeated NONOVERLAPPING development folds (formation window strictly before test). Report every fold including empty/no-trade folds; don't pick retrospectively selected survivors.
4. Lock experiment, open final 30% HOLDOUT **once**, prohibit tuning after opening; compare to the preregistered thresholds. If holdout disappoints, report no edge.
5. Run >=30 paper sessions using reconciliation + fills vs expected; only then consider broker integrations. No real-money order placement in this package.

## Deflated Sharpe / multiple testing
Do not call bootstrap/BH a 'deflated Sharpe': it is not. Full DSR requires effective number of trials, non-normal return skew/kurtosis and an adjusted benchmark. The number of **pairs searched** is disclosed in reports and all individual experiments must be tracked across repeated research runs. v1 reports a conservative minimum-across-fold Sharpe and an optional *probabilistic Sharpe* estimate only with sufficient data; it does not assert DSR > 1.

## Failure modes and leading indicators
1. **Structural breakup:** changing exposures, tokenomic revisions, regulatory news => frequent EG rejection, deteriorating two-half beta, rising OOS correlation of residuals. Stop new trades on failing formation checks.
2. **Execution and funding decay:** wider book spreads, large size/ADV, unpredictable funding => net/gross return gap, realized slippage > stressed estimate, adverse fills. Stop trading when fills exceed limits.
3. **Crowded unwind / directional regime:** joint deleveraging / trending spread => z repeatedly hits stop, half-life rises, volatility spikes, drawdown acceleration. Regime filter and 10% portfolio kill switch prevent new risk but do not guarantee fill prices.

## Bias audit status (as of implementation)
- Lookahead and execution: unit-tested synthetic cases; real exchange timestamp mapping remains to be audited.
- Survivorship: **NOT PASSED on current uploads** (no point-in-time delisted universe).
- Funding coverage: **NOT PASSED on current uploads** (no complete historical realized funding dataset).
- Holdout: **NOT OPENED**; no valid market dataset attached.
- Commission and slippage: engine implemented; venue-specific values require verification.
- DSR: **NOT COMPUTED**; no claim of its threshold being met.
- Regimes: report slices supported; pre/post-2020 not meaningful for short-lived perpetual histories and explicitly N/A.
- Live deployment: **NOT AUTHORIZED**.
