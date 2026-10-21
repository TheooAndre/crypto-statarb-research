"""Causal indicators and a single explicit spread position state machine."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import ResearchConfig
from .selection import estimate_beta


@dataclass
class Signal:
    z: float = float("nan")
    beta: float = float("nan")
    target_direction: int = 0
    reason: str = "warmup"
    vol_estimate: float = float("nan")


def rolling_signals(ly: np.ndarray, lx: np.ndarray, config: ResearchConfig,
                    first_decision_bar: int) -> list[Signal]:
    """At bar i close uses ONLY price indices <= i. Executes on i+1 open.

    Beta is refit on a trailing formation window at a fixed schedule, not
    recomputed from data subsequent to the signal being evaluated.
    """
    n = len(ly)
    out = [Signal() for _ in range(n)]
    direction, held = 0, 0
    beta = float("nan")
    for i in range(config.formation_bars - 1, n):
        # Don't trade inside formation, but fit indicator warm-up on completed closes.
        if (i - (config.formation_bars - 1)) % config.beta_refit_bars == 0:
            j = i - config.formation_bars + 1
            try:
                beta, _ = estimate_beta(ly[j:i + 1], lx[j:i + 1])
            except ValueError:
                beta = float("nan")
        if not np.isfinite(beta) or beta <= 0:
            continue
        j = i - config.z_window + 1
        if j < 0:
            continue
        spread = ly[j:i + 1] - beta * lx[j:i + 1]
        sd = float(np.std(spread, ddof=1))
        if not np.isfinite(sd) or sd < 1e-12:
            continue
        z = float((spread[-1] - spread.mean()) / sd)
        # Realized volatility measured in normalized fractional price returns.
        rr = (np.diff(ly[max(0, i - config.z_window):i + 1]) -
              beta * np.diff(lx[max(0, i - config.z_window):i + 1])) / (1 + abs(beta))
        vol = float(np.std(rr, ddof=1) * np.sqrt(config.bars_per_year)) if len(rr) > 3 else float("nan")
        reason = "flat"
        if i < first_decision_bar - 1:
            out[i] = Signal(z, beta, 0, "formation", vol)
            continue
        # Simple trailing volatility shock filter; no tuned ADX filters.
        baseline = []
        look = config.z_window
        start = max(config.formation_bars, i - 6 * look)
        for k in range(start, i - look + 1, look):
            b = ly[k:k+look] - beta * lx[k:k+look]
            baseline.append(float(np.std(np.diff(b), ddof=1)))
        current = float(np.std(np.diff(spread), ddof=1))
        trending = bool(baseline and np.median(baseline) > 0 and
                        current > config.max_vol_regime_multiplier * np.median(baseline))
        if direction == 0:
            if not trending and z >= config.entry_z:
                direction, held, reason = -1, 0, "enter_short_spread"
            elif not trending and z <= -config.entry_z:
                direction, held, reason = +1, 0, "enter_long_spread"
            else:
                reason = "regime_block" if trending else "flat"
        else:
            held += 1
            if abs(z) >= config.stop_z:
                direction, held, reason = 0, 0, "z_stop"
            elif abs(z) <= config.exit_z:
                direction, held, reason = 0, 0, "mean_exit"
            elif held >= config.max_holding_bars:
                direction, held, reason = 0, 0, "time_stop"
            else:
                reason = "holding"
        out[i] = Signal(z, beta, direction, reason, vol)
    return out
