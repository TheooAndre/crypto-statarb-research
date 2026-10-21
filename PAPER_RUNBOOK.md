# Paper-trading runbook: no live orders permitted yet

## Gate 0: research signoff

Keep research memo and instrument universe frozen and version-controlled. Required: actual historical point-in-time contract membership, adjusted contract migrations, 24h USD volume and OI snapshots, complete realized funding settlements, venue-verified fees, and slippage stress tests. Development fold distribution must be reported, with the worst folds included. **Explicit holdout pass and credible Deflated Sharpe analysis are not yet available; therefore current status: BLOCKED.**

## Gate 1: paper environment

Use the venue's own *testnet or paper simulator* once authenticated and confirmed compatible. Do not use live API credentials. Connect a new broker/testnet adapter (not present in this repository), convert exchange `contractSize`, min lot, tick sizes, leverage/margin and reduceOnly rules, and pass its executions to `PaperLedger.record(PaperFill(...))`. Orders from this package NEVER reach a venue.

1. Check NTP time sync; candle-open conventions; confirm API timestamp tolerance and signing. Prefer read-only API first.
2. Backfill and reconcile every position before allowing a new order. Implement idempotent client order IDs, partial fills, cancel/replace and restart recovery.
3. On disconnect >120s with open positions, **HALT & trigger actual venue reduce-only flatten** in a future, separately audited execution layer. Offline code here only locks trading; it cannot liquidate real positions.
4. Hard gross<=100% of equity, pair gross<=20%, asset gross<=30%, max drawdown 10%. Funding and fees in real account currency. Mandatory position reconciliation before every order.
5. Max per-leg order participation 1% of previous 4h dollar volume; record actual spread and slippage versus estimated every execution. If sustained realized slippage exceeds 2x expected, halt pending review.
6. Mobile alerting and systemd/Docker `restart: unless-stopped` are execution-adapter responsibilities; restart **must not** reset the persisted kill switch.

## >=30 paper sessions

Record every signal, intention timestamp, estimated open fill, hypothetical requested size, venue paper fill, venue actual funding, rejects, position ledger, and net equity. Compare assumed vs realized slippage by symbol and size/ADV bucket; compare annualized vol/hold-time; report >95% fill/reconciliation success (predeclared quality criterion). Sessions with API failure count as failures, not skipped data.

## Gate 2: tiny live (NOT IMPLEMENTED)

Only after research+paper pass, minimum size and full independently audited execution adapter. The prompt's “1 share equivalent” becomes exchange minimum **contract quantity** in perps. Require >=60 sessions before increasing size; realized live Sharpe within 30% of paper with uncertainty caveat. Any kill event ⇒ automatically cancel/reduce-only flatten using verified venue support, disable and page the operator. No automatic reset.

## Rollback

1. Disable new entries at broker/testnet AND locally; don't rely solely on local `KillSwitch`.
2. Cancel resting orders, query actual positions, independently reduce-only flatten every remaining leg (confirm rejects/partials).
3. Verify zero broker position AND zero open orders; reconcile fills/funding, persist immutable incident report.
4. Re-enable only after a human documents root cause and signs off `KillSwitch.manual_reset(explicit_human_approval=True)`.

## Top three decay indicators

- Breakup: unstable beta/EG failures and half-life rising across neighboring windows.
- Funding/cost blowout: realized-vs-expected slippage ratio, negative funding differential, high participation.
- Trending unwind: unusually many z-stop exits, drawdown slope, sudden vol >2x trailing median.
