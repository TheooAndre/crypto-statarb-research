import numpy as np
import pandas as pd
import pytest
from statarb.config import ResearchConfig
from statarb.synthetic import synthetic_market
from statarb.selection import Pair
from statarb.signals import rolling_signals
from statarb.execution import simulate_pair
from statarb.data import DataIntegrityError, validate_bars


def cfg():
    return ResearchConfig(formation_bars=180, test_bars=45, z_window=35)


def test_no_future_signal_lookahead():
    rng=np.random.default_rng(7)
    ly=np.cumsum(rng.normal(size=400)*.01)+5
    lx=ly+ rng.normal(size=400)*.03
    c=cfg()
    s1=rolling_signals(ly, lx, c, 180)
    mutated=ly.copy(); mutated[280:]+=100
    s2=rolling_signals(mutated, lx, c, 180)
    for t in range(279):
        a,b=s1[t],s2[t]
        assert a.target_direction == b.target_direction
        assert (np.isnan(a.z) and np.isnan(b.z)) or np.isclose(a.z,b.z)


def test_open_execution_and_two_legs_costs():
    d=synthetic_market(days=60)
    c=cfg()
    p=Pair('SIM_A','SIM_B',1.,.01,.01,10.)
    sim=simulate_pair(d,p,180,280,c)
    assert sim.returns.index.equals(d.bars['SIM_A'].index[180:280])
    assert (sim.fees_and_slippage >= 0).all()
    assert (sim.exposure_y.abs() + sim.exposure_x.abs() <= c.max_pair_gross+1e-9).all()
    assert sim.exposure_y.iloc[-1] == 0
    assert sim.exposure_x.iloc[-1] == 0
    assert np.allclose(sim.returns, sim.gross_returns + sim.funding_returns - sim.fees_and_slippage)


def test_funding_positive_costs_long_and_income_short():
    d=synthetic_market(days=60)
    c=cfg()
    p=Pair('SIM_A','SIM_B',1.,.01,.01,10.)
    s0=simulate_pair(d,p,180,340,c)
    d.funding['SIM_A'][:]=0.002
    s1=simulate_pair(d,p,180,340,c)
    expected = -s1.exposure_y.iloc[:-1].to_numpy()*.002
    got = (s1.funding_returns - s0.funding_returns).iloc[:-1].to_numpy()
    assert np.allclose(got,expected)


def test_missing_bars_refused():
    d=synthetic_market(days=35)
    df=d.bars['SIM_A'].drop(d.bars['SIM_A'].index[20])
    with pytest.raises(DataIntegrityError):
        validate_bars(df)


def test_incomplete_funding_fails_closed():
    d=synthetic_market(days=60)
    d.provenance['funding_complete']=False
    with pytest.raises(DataIntegrityError):
        simulate_pair(d,Pair('SIM_A','SIM_B',1.,.01,.01,10.),180,270,cfg())


def test_higher_commissions_never_increase_return_if_signals_unchanged():
    from dataclasses import replace
    d=synthetic_market(days=65)
    base=cfg()
    high=replace(base, fee_bps=base.fee_bps+15)
    p=Pair('SIM_A','SIM_B',1.,.01,.01,10.)
    a=simulate_pair(d,p,180,340,base)
    b=simulate_pair(d,p,180,340,high)
    assert b.returns.sum()<=a.returns.sum()+1e-12
    assert b.fees_and_slippage.sum()>=a.fees_and_slippage.sum()-1e-12
