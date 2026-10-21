"""Portfolio constraints and explicit persistent paper-trading kill switch."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
from .selection import Pair
from .config import ResearchConfig


def allocate_nonoverlapping(pairs: list[Pair], cfg: ResearchConfig) -> list[Pair]:
    """Prevents an underlying being stacked across different spreads.

    Conservative correlation proxy: pairs sharing a leg are rejected rather
    than estimating an unstable cross-pair covariance on a short formation.
    """
    chosen, used = [], set()
    for p in sorted(pairs, key=lambda x: x.adjusted_p):
        if p.y in used or p.x in used:
            continue
        if len(chosen) * cfg.max_pair_gross + cfg.max_pair_gross > cfg.max_portfolio_gross + 1e-12:
            break
        if cfg.max_pair_gross > cfg.max_asset_gross:
            raise ValueError("Impossible constraints: pair gross > asset cap")
        chosen.append(p)
        used.update((p.y, p.x))
    return chosen


def combine(simulations: list, idx: pd.DatetimeIndex, cfg: ResearchConfig) -> pd.Series:
    """Aggregate capped pair gross exposures and enforce portfolio drawdown stop.

    On breach at end of bar t, flatten at OPEN(t+1), conservatively charge
    an *additional* liquidation fee/slippage estimate. Can't claim an
    instantaneous exit at the violating price or back-adjust losing bar t.
    This research stop is not a replacement for a broker emergency close.
    """
    if not simulations:
        return pd.Series(0.0, index=idx, name="net_return")
    net = sum((sim.returns.reindex(idx, fill_value=0.0) for sim in simulations),
              pd.Series(0.0, index=idx))
    eq, peak, halted = 1., 1., False
    for i, ts in enumerate(idx):
        if halted:
            if i == first_halted:
                # Approximate liquidation cost at the first available opening.
                # Model gap risk conservatively in each pair's preceding bar return.
                gross_w = sum(float(abs(s.exposure_y.iloc[i-1]) + abs(s.exposure_x.iloc[i-1]))
                              for s in simulations)
                net.loc[ts] = -gross_w * (cfg.fee_bps + cfg.base_slippage_bps * 2) / 10_000
            else:
                net.loc[ts] = 0.
            continue
        eq *= (1 + float(net.loc[ts]))
        peak = max(peak, eq)
        if 1 - eq / peak >= cfg.max_drawdown:
            halted = True
            first_halted = i + 1
    return net.rename("net_return")


@dataclass
class KillSwitch:
    path: Path
    max_drawdown: float = 0.10
    disconnect_seconds: int = 120
    slippage_multiple: float = 2.0

    def _load(self) -> dict:
        if not self.path.exists():
            return {"disabled": False, "reason": None, "peak": 1.0}
        return json.loads(self.path.read_text())

    def is_disabled(self) -> bool:
        return bool(self._load()["disabled"])

    def observe(self, equity: float, last_connection_seconds: float = 0,
                has_position: bool = False, slippage_ratio: float = 1) -> bool:
        state = self._load()
        if state["disabled"]:
            return True
        state["peak"] = max(state["peak"], equity)
        why = None
        if equity < (1-self.max_drawdown) * state["peak"]:
            why = "portfolio_drawdown"
        elif has_position and last_connection_seconds > self.disconnect_seconds:
            why = "connection_stale_with_open_position"
        elif slippage_ratio > self.slippage_multiple:
            why = "persistent_slippage_breach"
        if why:
            state.update(disabled=True, reason=why, time=datetime.now(timezone.utc).isoformat())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2))
        return bool(state["disabled"])

    def manual_reset(self, explicit_human_approval: bool = False) -> None:
        if not explicit_human_approval:
            raise PermissionError("Human authorization required to re-enable trading")
        self.path.write_text(json.dumps({"disabled": False, "reason": None, "peak": 1.0}))
