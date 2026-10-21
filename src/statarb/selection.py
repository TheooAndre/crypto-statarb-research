"""Formation-only discovery with all-trials multiplicity correction."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.multitest import multipletests
from .config import ResearchConfig
from .data import ResearchData, eligible_at, common_bars


@dataclass(frozen=True)
class Pair:
    y: str
    x: str
    beta: float
    eg_p: float
    adjusted_p: float
    half_life: float


def estimate_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS on historical log closes, independent of current test bar."""
    if len(y) != len(x) or len(y) < 20 or np.var(x) < 1e-12:
        raise ValueError("Insufficient beta fit")
    beta, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(beta):
        raise ValueError("Non-finite beta")
    return float(beta), float(intercept)


def ar_half_life(residual: np.ndarray) -> float:
    lag = residual[:-1]
    nxt = residual[1:]
    if len(lag) < 20 or np.var(lag) < 1e-12:
        return float("inf")
    rho = np.cov(lag, nxt, ddof=1)[0, 1] / np.var(lag, ddof=1)
    return float(-np.log(2) / np.log(rho)) if 0 < rho < 1 else float("inf")


def discover(data: ResearchData, end_idx: int, config: ResearchConfig, *,
             candidate_symbols: list[str] | None = None) -> tuple[list[Pair], dict]:
    """Uses [end-formation, end) only. end bar never inspected."""
    timeline = data.index
    if end_idx < config.formation_bars or end_idx >= len(timeline):
        raise ValueError("Not enough formation history or bad decision index")
    ts = timeline[end_idx - 1]
    formation_grid = timeline[end_idx-config.formation_bars:end_idx]
    eligible = eligible_at(data, ts, min_open_interest_usd=config.min_open_interest_usd,
                           min_volume_24h_usd=config.min_volume_24h_usd)
    if candidate_symbols is not None:
        eligible = sorted(set(eligible) & set(candidate_symbols))
    # 1st stage I(1) screening, may reject good pairs (conservative).
    stationary_screen = []
    for sym in eligible:
        df = data.bars[sym].reindex(formation_grid)
        if len(df) != config.formation_bars or df.close.isna().any():
            continue
        p = np.log(df.close.to_numpy())
        try:
            if adfuller(p, maxlag=6, autolag="BIC")[1] > .10 and adfuller(np.diff(p), maxlag=6, autolag="BIC")[1] < .05:
                stationary_screen.append(sym)
        except (ValueError, np.linalg.LinAlgError):
            continue
    records = []
    tried = 0
    for y, x in combinations(stationary_screen, 2):
        try:
            yy = np.log(data.bars[y].reindex(formation_grid).close.to_numpy())
            xx = np.log(data.bars[x].reindex(formation_grid).close.to_numpy())
            tried += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p, _ = coint(yy, xx, maxlag=6, autolag="BIC")
            if not np.isfinite(p):
                p = 1.0
            beta, intercept = estimate_beta(yy, xx)
            res = yy - beta * xx - intercept
            hl = ar_half_life(res)
            # Refuse beta instability within development formation; not OOS-tuned.
            middle = len(yy) // 2
            b1, _ = estimate_beta(yy[:middle], xx[:middle])
            b2, _ = estimate_beta(yy[middle:], xx[middle:])
            stable = beta > 0 and b1 > 0 and b2 > 0 and abs(b1 - b2) / max(abs(beta), 1e-6) < .35
            records.append((y, x, beta, float(p), hl, stable))
        except (ValueError, np.linalg.LinAlgError):
            continue
    audit = {"decision_timestamp": ts.isoformat(), "eligible_symbols": len(eligible),
             "adf_passed_symbols": len(stationary_screen), "pairs_tested": tried,
             "raw_eg_tests": len(records), "selected_pairs": 0}
    if not records:
        return [], audit
    reject, adjusted, _, _ = multipletests([r[3] for r in records], alpha=config.fdr_q, method="fdr_bh")
    pairs = [Pair(r[0], r[1], r[2], r[3], float(q), r[4]) for r, yes, q in zip(records, reject, adjusted)
             if yes and r[5] and config.min_half_life <= r[4] <= config.max_half_life]
    audit["selected_pairs"] = len(pairs)
    return sorted(pairs, key=lambda p: p.adjusted_p), audit
