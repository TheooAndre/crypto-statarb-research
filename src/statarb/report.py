"""Honest, reproducible risk metrics; empty evaluation is no evidence."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ResearchConfig


def _closed_trades(sim) -> list[float]:
    active = (sim.exposure_y.abs() + sim.exposure_x.abs()) > 1e-10
    values = sim.returns.to_numpy()
    trades, running, opened = [], 0., False
    for idx, (a, r) in enumerate(zip(active.to_numpy(), values)):
        if a and not opened:
            running, opened = 0., True
        if opened:
            running += float(r)
        if not a and opened:
            trades.append(running)
            running, opened = 0., False
    if opened:
        trades.append(running)
    return trades


def metrics(ret: pd.Series, cfg: ResearchConfig, simulations: list | None = None) -> dict:
    if len(ret) == 0:
        return {"status": "NO_DATA"}
    r = ret.fillna(0.0).astype(float)
    periods = cfg.bars_per_year
    eq = (1 + r).cumprod()
    baseline = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(hours=cfg.timeframe_hours)]), eq])
    rolling_high = baseline.cummax()
    dd = baseline / rolling_high - 1
    max_dd = float(dd.min())
    trough = dd.idxmin()
    peak = baseline.loc[:trough].idxmax()
    nav = float(eq.iloc[-1])
    duration_years = len(r) / periods
    cagr = nav ** (1/duration_years) - 1 if nav > 0 and duration_years > 0 else -1.
    stdev = float(r.std(ddof=1)) if len(r)>1 else 0.
    sharpe = float(r.mean()/stdev*np.sqrt(periods)) if stdev > 0 else float('nan')
    downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)))
    sortino = float(r.mean()/downside*np.sqrt(periods)) if downside > 0 else float('nan')
    calmar = float(cagr/abs(max_dd)) if max_dd < 0 else float('nan')
    # Maximum continuous count of observations underwater
    under = dd.iloc[1:].to_numpy() < -1e-12
    longest, current = 0, 0
    for val in under:
        current = current + 1 if val else 0
        longest = max(longest, current)
    result = {"status": "RESEARCH_ONLY", "sharpe": sharpe, "sortino": sortino,
              "calmar": calmar, "cagr": float(cagr), "max_drawdown": max_dd,
              "max_drawdown_peak": str(peak), "max_drawdown_trough": str(trough),
              "longest_drawdown_bars": longest, "ending_equity_multiple": nav,
              "bars": len(r), "years": duration_years,
              "deflated_sharpe": None, "deflated_sharpe_reason":
              "Requires independent market trials/effective number of research attempts; no substitute metric claimed."}
    if simulations:
        trades = [p for sim in simulations for p in _closed_trades(sim)]
        gains = [t for t in trades if t > 0]
        losses = [-t for t in trades if t <= 0]
        result.update({
            "trades": len(trades), "win_rate": len(gains)/len(trades) if trades else None,
            "profit_factor": sum(gains)/sum(losses) if sum(losses) > 0 else None,
            "avg_win_loss_ratio": (np.mean(gains)/np.mean(losses)) if gains and losses else None,
            "annual_turnover": float(sum(abs(d) for s in simulations for col in ("delta_y","delta_x")
                for d in s.fills.get(col, pd.Series(dtype=float))) / max(duration_years, 1e-8)),
            "total_cost_fraction": float(sum(s.fees_and_slippage.sum() for s in simulations)),
            "total_gross_fraction": float(sum(s.gross_returns.sum() for s in simulations)),
            "total_funding_fraction": float(sum(s.funding_returns.sum() for s in simulations)),
            "cost_drag_fraction": float(sum(s.fees_and_slippage.sum() for s in simulations)),
            "cost_drag_pct_gross_pnl": (100 * sum(s.fees_and_slippage.sum() for s in simulations) /
                        sum(s.gross_returns.sum() for s in simulations)
                        if sum(s.gross_returns.sum() for s in simulations) > 0 else None),
        })
    return result


def regime_breakdown(ret: pd.Series, benchmark: pd.Series | None = None) -> dict:
    if ret.empty:
        return {}
    if benchmark is None or len(benchmark) != len(ret):
        return {"vol_high": "N/A (benchmark required)", "vol_low": "N/A (benchmark required)",
                "bull": "N/A (benchmark required)", "bear": "N/A (benchmark required)",
                "pre_2020": "N/A unless period covered"}
    b = benchmark.reindex(ret.index)
    rv = b.rolling(30, min_periods=30).std().shift(1)
    trend = b.rolling(180, min_periods=180).sum().shift(1)
    median = rv.expanding(min_periods=180).median().shift(1)
    groups = {"vol_high": rv > median, "vol_low": rv <= median,
              "bull": trend > 0, "bear": trend <= 0}
    out = {}
    for k, mask in groups.items():
        group = ret.loc[mask.fillna(False)]
        out[k] = {"n_bars": len(group), "mean_bar_return": float(group.mean())} if len(group) else None
    out["pre_2020"] = ("N/A" if (ret.index >= pd.Timestamp("2020-01-01", tz="UTC")).all()
                        else "Present, inspect historical venue viability")
    return out


def month_matrix(ret: pd.Series) -> dict:
    if ret.empty:
        return {}
    eq = (1 + ret).resample('ME').prod() - 1
    return {i.strftime('%Y-%m'): float(v) for i, v in eq.items()}


def save_report(folder: Path, title: str, ret: pd.Series, summary: dict,
                audits: list[dict], simulations: list, cfg: ResearchConfig) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    full = dict(summary)
    full.update({"label": title, "discovery_audit": audits,
                 "monthly_returns": month_matrix(ret)})
    (folder / f'{title}.json').write_text(json.dumps(full, indent=2, default=str, allow_nan=False))
    ret.to_csv(folder / f'{title}_equity.csv', header=['net_bar_return'])
    if simulations:
        fills = [sim.fills for sim in simulations if not sim.fills.empty]
        if fills:
            pd.concat(fills, ignore_index=True).to_csv(folder / f'{title}_fills.csv', index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,4))
        ((1+ret).cumprod()).plot(ax=ax)
        ax.set(title=f'{title}: net equity multiple (research only)', xlabel='UTC', ylabel='Equity / initial')
        fig.tight_layout()
        fig.savefig(folder / f'{title}_equity.png', dpi=140)
        plt.close(fig)
        mon = month_matrix(ret)
        if mon:
            years = sorted({int(k[:4]) for k in mon})
            matrix = np.full((len(years),12), np.nan)
            for month, value in mon.items():
                y, m = map(int, month.split('-'))
                matrix[years.index(y),m-1] = value * 100
            fig, ax = plt.subplots(figsize=(11, max(2.5, .8*len(years)+1)))
            im=ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
            ax.set_xticks(range(12), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
            ax.set_yticks(range(len(years)),[str(y) for y in years])
            ax.set(title=f'{title}: monthly NET returns %; worst month outlined', xlabel='Month')
            for j in range(len(years)):
                for k in range(12):
                    if np.isfinite(matrix[j,k]):
                        ax.text(k,j,f'{matrix[j,k]:+.1f}',ha='center',va='center',fontsize=8)
            minimum = np.unravel_index(np.nanargmin(matrix),matrix.shape)
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((minimum[1]-.5,minimum[0]-.5),1,1,fill=False,edgecolor='black',linewidth=2))
            fig.colorbar(im,ax=ax,label='Net return %')
            fig.tight_layout()
            fig.savefig(folder / f'{title}_monthly_heatmap.png',dpi=130)
            plt.close(fig)
    except ImportError:
        pass
