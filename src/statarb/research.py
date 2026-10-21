"""Repeated formation-only screening; NONoverlapping OOS folds; locked holdout."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import ResearchConfig
from .data import ResearchData, DataIntegrityError
from .selection import discover
from .execution import simulate_pair, Simulation
from .risk import allocate_nonoverlapping, combine
from .report import metrics, regime_breakdown, month_matrix


@dataclass
class Evaluation:
    name: str
    net_returns: pd.Series
    simulations: list[Simulation]
    audits: list[dict]
    fold_metrics: list[dict]
    statistics: dict


def _evaluate(data: ResearchData, cfg: ResearchConfig, lo: int, hi: int,
              name: str, fixed_pairs=None) -> Evaluation:
    idx = data.index
    pieces, sims, audits, folds = [], [], [], []
    cursor = lo
    while cursor < hi:
        end = min(hi, cursor + cfg.test_bars)
        if end - cursor < 3:
            break
        if fixed_pairs is None:
            pairs, audit = discover(data, cursor, cfg)
            pairs = allocate_nonoverlapping(pairs, cfg)
        else:
            pairs = fixed_pairs
            audit = {"decision_timestamp": str(idx[cursor-1]), "locked_holdout_pairs":
                     [f"{p.y}/{p.x}" for p in pairs], "pairs_tested": 0, "selected_pairs": len(pairs)}
        audit['start'] = str(idx[cursor]); audit['end_exclusive'] = str(idx[end-1] + pd.Timedelta(hours=cfg.timeframe_hours))
        audits.append(audit)
        step_sims = []
        for pair in pairs:
            sim = simulate_pair(data, pair, cursor, end, cfg)
            step_sims.append(sim)
        step = combine(step_sims, idx[cursor:end], cfg)
        pieces.append(step)
        sims.extend(step_sims)
        fold_record = metrics(step, cfg, step_sims)
        fold_record.update({"start": str(idx[cursor]), "end": str(idx[end-1]),
                            "selected_pairs": [f"{p.y}/{p.x}" for p in pairs]})
        folds.append(fold_record)
        cursor = end  # CRITICAL: nonoverlap; never step every 14 days into a 21d window.
    if not pieces:
        raise DataIntegrityError("No evaluable walk-forward folds")
    ret = pd.concat(pieces)
    if ret.index.has_duplicates or len(ret) != hi - lo:
        raise DataIntegrityError("Overlapping/missing OOS windows")
    stats = metrics(ret, cfg, sims)
    stats['fold_sharpes'] = [x.get('sharpe') for x in folds]
    finite_fold_sharpes = [v for v in stats['fold_sharpes'] if v is not None and np.isfinite(v)]
    stats['median_fold_sharpe'] = float(np.median(finite_fold_sharpes)) if finite_fold_sharpes else None
    stats['worst_fold_sharpe'] = float(min(finite_fold_sharpes)) if finite_fold_sharpes else None
    stats['trials_searched_total'] = sum(x.get('pairs_tested', 0) for x in audits)
    benchmark_name = data.provenance.get('benchmark_symbol')
    if benchmark_name in data.bars:
        opens = data.bars[benchmark_name].open
        benchmark = (opens.shift(-1)/opens - 1).reindex(ret.index)
        if benchmark.notna().all():
            stats['regime_breakdown'] = regime_breakdown(ret, benchmark)
        else:
            stats['regime_breakdown'] = regime_breakdown(ret)
    else:
        stats['regime_breakdown'] = regime_breakdown(ret)
    stats['paper_readiness'] = 'BLOCKED: historical PIT+funding provenance and genuine held-out market performance required'
    return Evaluation(name, ret, sims, audits, folds, stats)


def run_research(data: ResearchData, cfg: ResearchConfig, *,
                 open_holdout: bool = False) -> dict[str, Evaluation]:
    if cfg.require_point_in_time_universe and not data.provenance.get('universe_point_in_time'):
        raise DataIntegrityError('Point-in-time universe missing')
    if cfg.require_funding and not data.provenance.get('funding_complete'):
        raise DataIntegrityError('Funding coverage missing')
    common_idx = data.index
    bar_step = pd.Timedelta(hours=cfg.timeframe_hours)
    if len(common_idx)>1 and not (common_idx.to_series().diff().iloc[1:] == bar_step).all():
        raise DataIntegrityError('Venue timeline has missing market bars')
    n = len(common_idx)
    split = int(n * (1 - cfg.holdout_fraction))
    if split < cfg.formation_bars + cfg.test_bars * 3:
        raise DataIntegrityError('Insufficient history for >=3 development folds; extend dataset')
    dev = _evaluate(data, cfg, cfg.formation_bars, split, 'development')
    output = {'development': dev}
    if open_holdout:
        # An immutable artifact should be committed BEFORE invoking --open-holdout.
        # Fit exactly once using only formation history ending before split.
        locked, lock_audit = discover(data, split, cfg)
        locked = allocate_nonoverlapping(locked, cfg)
        hold = _evaluate(data, cfg, split, n, 'holdout', fixed_pairs=locked)
        hold.audits.insert(0, {'locked_at': str(common_idx[split-1]),
                               'pre_holdout_formation_only': True, 'lock_audit': lock_audit})
        a = dev.statistics['sharpe']; b = hold.statistics['sharpe']
        hold.statistics['holdout_vs_dev_sharpe_ratio'] = (b/a if a is not None and np.isfinite(a) and a>0 else None)
        hold.statistics['OOS_less_than_half_IS_flag'] = bool(np.isfinite(a) and a>0 and np.isfinite(b) and b < .5*a)
        output['holdout'] = hold
    return output
