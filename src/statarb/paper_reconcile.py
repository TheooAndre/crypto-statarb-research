"""Reconcile external paper-exchange fills exported as CSV; no order placement."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .paper import PaperFill, PaperLedger


def reconcile_csv(csv_path: Path, journal_dir: Path) -> dict:
    """Required columns: timestamp_utc,symbol,side,expected_price,
    simulated_price,quantity,fee. This must be broker simulator/export data,
    never filled with fictional prices. Returns aggregate real paper-fill gaps.
    """
    df = pd.read_csv(csv_path)
    required = set(PaperFill.__dataclass_fields__)
    if not required.issubset(df.columns):
        raise ValueError(f'Missing paper fill columns: {sorted(required-set(df.columns))}')
    if df.empty:
        raise ValueError('No venue paper fills to reconcile')
    ledger = PaperLedger(journal_dir)
    fills = []
    for _, row in df.iterrows():
        f = PaperFill(timestamp_utc=str(row.timestamp_utc), symbol=str(row.symbol),
                      side=str(row.side), expected_price=float(row.expected_price),
                      simulated_price=float(row.simulated_price), quantity=float(row.quantity),
                      fee=float(row.fee))
        ledger.record(f)
        fills.append(f)
    adverse = [f.slippage_bps for f in fills]
    result = {'fills':len(fills), 'mean_signed_adverse_slippage_bps':sum(adverse)/len(adverse),
              'worst_adverse_slippage_bps':max(adverse),
              'broker_connected':False, 'venue_data_trust':'USER_SUPPLIED_PAPER_EXPORT',
              'no_live_orders':True}
    (journal_dir/'reconciliation.json').write_text(json.dumps(result,indent=2))
    return result
