"""Reproducible, offline data research entry point. Real orders not supported."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import ResearchConfig
from .data import load_folder
from .synthetic import synthetic_market
from .research import run_research
from .report import save_report


def _clean(v):
    if isinstance(v, dict):
        return {k:_clean(x) for k,x in v.items()}
    if isinstance(v,list):
        return [_clean(x) for x in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):
        return None
    return v


def main(argv=None):
    p = argparse.ArgumentParser(description='Bias-conscious pairs research; never places orders')
    p.add_argument('command', choices=['demo','research'])
    p.add_argument('--data', help='Folder containing OHLCV, funding, provenance, historical membership')
    p.add_argument('--out', default='reports')
    p.add_argument('--open-holdout', action='store_true', help='Open untouched holdout ONCE after freezing config')
    args = p.parse_args(argv)
    cfg = ResearchConfig()
    if args.command == 'research' and not args.data:
        p.error('research requires --data with point-in-time historical files')
    data = synthetic_market(days=420) if args.command=='demo' else load_folder(args.data, cfg.timeframe_hours)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock_payload = json.dumps({'config':asdict(cfg), 'dataset_hash':data.provenance.get('sha256_input'),
                               'hypothesis': 'RESEARCH_MEMO.md: 2026-10-19'}, sort_keys=True)
    fingerprint = hashlib.sha256(lock_payload.encode()).hexdigest()
    lock = out/'holdout_lock.json'
    if args.open_holdout:
        if lock.exists():
            raise SystemExit('HOLDOUT already opened for this report path. Refusing to rerun/retest.')
        lock.write_text(json.dumps({'fingerprint':fingerprint,'warning':'HOLDOUT OPENED; do not tune after this'},indent=2))
    result = run_research(data, cfg, open_holdout=args.open_holdout)
    for name, evaluation in result.items():
        summ = _clean(evaluation.statistics)
        save_report(out,name,evaluation.net_returns,summ,evaluation.audits,evaluation.simulations,cfg)
        (out/f'{name}_folds.json').write_text(json.dumps(_clean(evaluation.fold_metrics), indent=2, default=str))
        print(f'{name}: {summ["status"]}; Sharpe={summ.get("sharpe")}; DD={summ.get("max_drawdown")}; '
              f'trades={summ.get("trades")}; folds={len(evaluation.fold_metrics)}')
    (out/'experiment.json').write_text(json.dumps({'fingerprint':fingerprint,'config':asdict(cfg),
                'provenance':data.provenance,'synthetic':bool(data.provenance.get('synthetic')),
                'holdout_opened':args.open_holdout,'live_permission':False}, indent=2))
    print('Research-only. No evidence of live profitability or trading authorization.')


if __name__=='__main__':
    main()
