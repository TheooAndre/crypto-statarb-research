"""Conservative event-style open-to-open simulator, explicitly NOT live routing."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import ResearchConfig
from .data import ResearchData, DataIntegrityError, common_bars
from .selection import Pair
from .signals import rolling_signals


@dataclass
class Simulation:
    returns: pd.Series
    gross_returns: pd.Series
    fees_and_slippage: pd.Series
    funding_returns: pd.Series
    exposure_y: pd.Series
    exposure_x: pd.Series
    fills: pd.DataFrame
    signals: pd.DataFrame
    stopped: bool


def simulate_pair(data: ResearchData, pair: Pair, start: int, end: int,
                  cfg: ResearchConfig) -> Simulation:
    """Bar timestamps represent candle starts. Signal close(t-1) -> open(t).

    pnl for bar t is open(t) to open(t+1), with both legs and funding events.
    The final bar within the evaluation window is flattened at its OPEN,
    removing the need to invent a next-open price beyond the evaluation window.
    """
    timeline = data.index
    if not (cfg.formation_bars <= start < end <= len(timeline)):
        raise ValueError("Invalid global execution window")
    history_start = start - cfg.formation_bars
    grid = timeline[history_start:end]
    dy, dx = data.bars[pair.y].reindex(grid), data.bars[pair.x].reindex(grid)
    if dy[list(('open','high','low','close','volume'))].isna().any().any() or dx[list(('open','high','low','close','volume'))].isna().any().any():
        raise DataIntegrityError("Selected pair has missing/delisted candles during eval; result invalid, do not discard silently")
    local_start, local_end = cfg.formation_bars, len(grid)
    if cfg.require_funding and (pair.y not in data.funding or pair.x not in data.funding or
                                not data.provenance.get('funding_complete')):
        raise DataIntegrityError("Realized funding history not attested complete")
    if end - start < 3:
        raise ValueError("Evaluation window must contain at least 3 bars")
    # IMPORTANT: no unseen post-end bars are included in signals.
    logy, logx = np.log(dy.close.to_numpy()), np.log(dx.close.to_numpy())
    signals = rolling_signals(logy, logx, cfg, first_decision_bar=local_start)
    iy, ix = [], []
    returns, gross_list, cost_list, fund_list, fill_log, signal_log = [], [], [], [], [], []
    prev_y, prev_x, high_water, equity, killed = 0.0, 0.0, 1.0, 1.0, False
    idx = dy.index
    for t in range(local_start, local_end):
        ts = idx[t]
        sig = signals[t-1]
        if killed or t == local_end-1 or sig.target_direction == 0 or not np.isfinite(sig.beta):
            want_y = want_x = 0.0
        else:
            vol = sig.vol_estimate
            gross = min(cfg.max_pair_gross, cfg.target_vol_annual / vol) if np.isfinite(vol) and vol > 0 else 0.0
            want_y = sig.target_direction * gross / (1.0 + abs(sig.beta))
            want_x = -sig.target_direction * gross * sig.beta / (1.0 + abs(sig.beta))
        # Venue-specific participation: estimate each order relative to previous bars' traded USD.
        def impact(old: float, new: float, df: pd.DataFrame) -> float:
            change = abs(new-old)
            if change == 0:
                return 0.0
            historical = df.iloc[max(0, t-30):t]
            if len(historical) < 10:
                raise DataIntegrityError("Insufficient ADV history")
            adv = float((historical.close * historical.volume).mean())
            if adv <= 0:
                raise DataIntegrityError("No historical market depth proxy")
            participation = change * cfg.starting_equity / adv
            if participation > 0.01:
                raise DataIntegrityError("Estimated participation >1% per 4h bar; do not assume fill")
            local_vol = np.log(historical.close).diff().std()
            volatility_bps = float(local_vol) * 10_000 if np.isfinite(local_vol) else 0.0
            bps = (cfg.fee_bps + cfg.base_slippage_bps +
                   cfg.vol_slippage_factor * volatility_bps +
                   cfg.participation_impact_bps * np.sqrt(participation))
            return change * bps / 10_000
        cost = impact(prev_y, want_y, dy) + impact(prev_x, want_x, dx)
        if abs(want_y-prev_y) > 1e-12 or abs(want_x-prev_x) > 1e-12:
            fill_log.append({"time":ts.isoformat(), "asset_y":pair.y, "asset_x":pair.x,
                             "target_y":want_y, "target_x":want_x,
                             "delta_y":want_y-prev_y, "delta_x":want_x-prev_x,
                             "expected_open_y":float(dy.open.iloc[t]),
                             "expected_open_x":float(dx.open.iloc[t]),
                             "estimated_cost_fraction":cost, "reason": "window_end" if t==end-1 else sig.reason})
        prev_y, prev_x = want_y, want_x
        gross_ret, f_ret = 0.0, 0.0
        if t < local_end-1:
            ry = float(dy.open.iloc[t+1] / dy.open.iloc[t] - 1)
            rx = float(dx.open.iloc[t+1] / dx.open.iloc[t] - 1)
            gross_ret = want_y * ry + want_x * rx
            if cfg.require_funding:
                left, right = idx[t], idx[t+1]
                yf = data.funding[pair.y]
                xf = data.funding[pair.x]
                rate_y = float(yf.loc[(yf.index >= left) & (yf.index < right)].sum())
                rate_x = float(xf.loc[(xf.index >= left) & (xf.index < right)].sum())
                # Positive funding: long pays, short receives.
                f_ret = -want_y * rate_y - want_x * rate_x
        net = gross_ret + f_ret - cost
        equity *= (1 + net)
        if equity <= 0:
            raise DataIntegrityError("Portfolio equity nonpositive")
        high_water = max(high_water, equity)
        if 1 - equity / high_water >= cfg.max_drawdown:
            killed = True  # flattened at next available open; cannot magically exit here
        returns.append(net)
        gross_list.append(gross_ret)
        cost_list.append(cost)
        fund_list.append(f_ret)
        iy.append(want_y)
        ix.append(want_x)
        signal_log.append({"time":ts.isoformat(), "z_signal_prev_close":sig.z,
                           "beta_signal_prev_close":sig.beta, "direction":sig.target_direction,
                           "reason":sig.reason, "stopped":killed})
    return Simulation(
        returns=pd.Series(returns, index=idx[local_start:local_end], name=f"{pair.y}-{pair.x}"),
        gross_returns=pd.Series(gross_list, index=idx[local_start:local_end]),
        fees_and_slippage=pd.Series(cost_list, index=idx[local_start:local_end]),
        funding_returns=pd.Series(fund_list, index=idx[local_start:local_end]),
        exposure_y=pd.Series(iy, index=idx[local_start:local_end]),
        exposure_x=pd.Series(ix, index=idx[local_start:local_end]),
        fills=pd.DataFrame(fill_log), signals=pd.DataFrame(signal_log), stopped=killed,
    )
