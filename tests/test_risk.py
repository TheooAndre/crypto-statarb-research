from pathlib import Path
import pytest
from statarb.config import ResearchConfig
from statarb.selection import Pair
from statarb.risk import KillSwitch, allocate_nonoverlapping
from statarb.paper import PaperLedger, PaperFill


def test_shared_leg_cap():
    c=ResearchConfig()
    pairs=[Pair('BTC','ETH',1,.001,.001,10),Pair('BTC','SOL',1,.002,.002,12),Pair('ADA','SOL',1,.003,.003,10)]
    chosen=allocate_nonoverlapping(pairs,c)
    assert len(chosen)==2
    assert chosen[0].y=='BTC'


def test_kill_switch_persisted_and_manual_reset(tmp_path):
    k=KillSwitch(tmp_path/'kill.json')
    assert not k.observe(1.0)
    assert k.observe(.89)
    assert k.is_disabled()
    with pytest.raises(PermissionError):
        k.manual_reset()
    k.manual_reset(explicit_human_approval=True)
    assert not k.is_disabled()


def test_ledger_reconciles_slippage(tmp_path):
    l=PaperLedger(tmp_path)
    f=PaperFill('2026-01-01T00:00:00Z','BTC','buy',100,100.1,1,0.05)
    l.record(f)
    assert round(f.slippage_bps,4)==10
    assert l.events.exists()
    assert l.health(1,0,False,3,2)['status']=='PAPER_OK'
    assert l.health(1,0,True,20,2)['status']=='HALTED'


def test_portfolio_dd_stops_future_returns_and_charges_liquidation():
    import pandas as pd
    from types import SimpleNamespace
    from statarb.risk import combine
    idx=pd.date_range('2026-01-01', periods=5, freq='4h', tz='UTC')
    fake=SimpleNamespace(returns=pd.Series([-.11,.5,.5,.5,0], index=idx),
        exposure_y=pd.Series([.1]*5,index=idx),exposure_x=pd.Series([-.1]*5,index=idx))
    net=combine([fake],idx,ResearchConfig())
    assert net.iloc[0]==-.11
    assert net.iloc[1] < 0
    assert (net.iloc[2:]==0).all()


def test_external_paper_fill_reconciliation(tmp_path):
    import pandas as pd
    from statarb.paper_reconcile import reconcile_csv
    inp=tmp_path/'paper.csv'
    pd.DataFrame([{'timestamp_utc':'2026-01-01T00:00:00Z', 'symbol':'SIM_A',
        'side':'buy', 'expected_price':100, 'simulated_price':100.1,
        'quantity':1.0, 'fee':0.02}]).to_csv(inp,index=False)
    report=reconcile_csv(inp,tmp_path/'journal')
    assert report['fills']==1
    assert report['no_live_orders'] is True
