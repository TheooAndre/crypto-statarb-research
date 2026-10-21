import numpy as np
import pandas as pd
from statarb.config import ResearchConfig
from statarb.synthetic import synthetic_market
from statarb.research import run_research
from statarb.selection import discover
from statarb.data import eligible_at


def tiny():
    return ResearchConfig(formation_bars=180, test_bars=60, z_window=40,
                          beta_refit_bars=30, max_holding_bars=40)


def test_pit_membership():
    d=synthetic_market(days=70)
    ts=d.bars['SIM_A'].index[200]
    assert 'SIM_A' in eligible_at(d,ts)


def test_folds_do_not_overlap_holdout_not_opened():
    d=synthetic_market(days=125)
    c=tiny()
    res=run_research(d,c,open_holdout=False)
    assert 'holdout' not in res
    dev=res['development']
    assert dev.net_returns.index.is_unique
    assert len(dev.net_returns)==int(len(d.bars['SIM_A'])*.7)-c.formation_bars
    assert all('pairs_tested' in a for a in dev.audits)


def test_future_prices_cannot_change_pair_discovery():
    d=synthetic_market(days=70)
    c=tiny()
    a,aa=discover(d, 200, c)
    d.bars['SIM_A'].loc[d.bars['SIM_A'].index[250]:,'close'] *= 10
    b,bb=discover(d, 200, c)
    assert a==b
    assert aa==bb


def test_delisted_history_is_representable_and_not_silently_screened_out():
    from statarb.data import validate_bars
    d=synthetic_market(days=125)
    # C was delisted after 65d. Its historical prices remain searchable in earlier folds.
    cutoff=65*6
    d.bars['SIM_C']=validate_bars(d.bars['SIM_C'].iloc[:cutoff])
    d.membership.loc[(d.membership.symbol=='SIM_C') &
                     (d.membership.timestamp > d.bars['SIM_C'].index[-1]), 'active']=False
    assert len(d.index)==125*6
    assert 'SIM_C' not in eligible_at(d, d.index[-1])
    c=tiny()
    res=run_research(d,c)
    assert res['development'].net_returns.index.is_unique


def test_holdout_is_disjoint_and_locked_pair_set():
    d=synthetic_market(days=125)
    c=tiny()
    res=run_research(d,c,open_holdout=True)
    dev=res['development'].net_returns
    hold=res['holdout'].net_returns
    assert len(dev.index.intersection(hold.index))==0
    assert dev.index[-1] < hold.index[0]
    locked_lists=[a.get('locked_holdout_pairs') for a in res['holdout'].audits if 'locked_holdout_pairs' in a]
    assert len(set(map(tuple,locked_lists)))==1


def test_real_loader_requires_provenance(tmp_path):
    from statarb.data import load_folder, DataIntegrityError
    import pytest
    with pytest.raises(DataIntegrityError, match='manifest'):
        load_folder(tmp_path)
