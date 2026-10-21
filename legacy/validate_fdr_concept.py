"""
Standalone, dependency-free (numpy/pandas/scipy only) proof of the core claim:
naive pairwise cointegration screening at p<0.05 across a large universe
produces false discoveries at roughly the base rate, and Benjamini-Hochberg
FDR correction fixes it.

Why this file exists separately from the real pipeline: this sandbox has no
network access, so statsmodels (which the real pair_selection.py uses, same
as your original script) can't be installed here. Rather than hand statsmodels
calls to you unverified, I built a minimal ADF/Engle-Granger implementation
from the actual regression definitions and Monte-Carlo-calibrated the null
distribution myself (instead of trusting a from-memory reproduction of
MacKinnon's asymptotic critical-value tables, which I have no way to check
against a reference here). This is slower than the real thing but is
self-verifying: under a true null, p-values should be ~Uniform(0,1), which
you can see checked below.
"""

import numpy as np
import pandas as pd
from itertools import combinations
import time


# ---------------------------------------------------------------------------
# Minimal ADF / Engle-Granger test statistics (regression from first principles)
# ---------------------------------------------------------------------------

def adf_stat(y, lags: int = 1) -> float:
    """t-stat on the lagged-level coefficient in the ADF regression:
    dy_t = c + gamma*y_{t-1} + sum(delta_i * dy_{t-i}) + eps_t
    More negative => stronger evidence against a unit root (stationary)."""
    y = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
    dy = y.diff()
    df = pd.DataFrame({"dy": dy, "y_lag1": y.shift(1)})
    for i in range(1, lags + 1):
        df[f"dy_lag{i}"] = dy.shift(i)
    df = df.dropna()
    if len(df) < 10:
        return np.nan
    Y = df["dy"].values
    X = np.column_stack([np.ones(len(df)), df["y_lag1"].values]
                         + [df[f"dy_lag{i}"].values for i in range(1, lags + 1)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return np.nan
    sigma2 = (resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan
    se = np.sqrt(sigma2 * xtx_inv[1, 1])
    return beta[1] / se if se > 0 else np.nan


def eg_stat(y, x, lags: int = 1):
    """Engle-Granger 2-step: OLS y on x, then ADF stat on residuals.
    Returns (test_stat, hedge_ratio)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return adf_stat(resid, lags=lags), beta[1]


def calibrate_null(n: int, kind: str, n_sims: int = 2000, lags: int = 1, seed: int = 0):
    """Monte Carlo null distribution of the test stat under the TRUE null
    (independent random walks -> no unit-root rejection / no cointegration).
    Calibrated once per sample length, reused across every candidate pair --
    this is what makes testing ~1000 pairs computationally cheap."""
    rng = np.random.default_rng(seed)
    stats_ = np.empty(n_sims)
    for i in range(n_sims):
        if kind == "adf":
            series = np.cumsum(rng.normal(size=n))
            stats_[i] = adf_stat(series, lags=lags)
        else:
            x = np.cumsum(rng.normal(size=n))
            y = np.cumsum(rng.normal(size=n))  # independent -- no true relationship
            s, _ = eg_stat(y, x, lags=lags)
            stats_[i] = s
    return np.sort(stats_[~np.isnan(stats_)])


def mc_pvalue(observed: float, null_dist: np.ndarray) -> float:
    """(count + 1)/(n_sims + 1) -- standard Monte Carlo p-value estimator
    (Davison & Hinkley). Avoids ever reporting exactly p=0, which a finite
    simulation can never actually justify."""
    if np.isnan(observed) or len(null_dist) == 0:
        return 1.0
    count_le = np.searchsorted(null_dist, observed, side="right")
    return float(count_le + 1) / (len(null_dist) + 1)


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> np.ndarray:
    """Standard BH step-up FDR procedure. Returns boolean reject-null mask."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    cutoff_p = sorted_p[np.max(np.where(below)[0])]
    return p <= cutoff_p


# ---------------------------------------------------------------------------
# Sanity check: under a KNOWN true null, p-values should be ~Uniform(0,1).
# If this fails, the test statistic/calibration has a bug -- don't trust
# anything downstream until this passes.
# ---------------------------------------------------------------------------

def _sanity_check_uniform_pvalues(n=400, n_sims=1500, n_check=500, seed=99):
    null_dist = calibrate_null(n, kind="eg", n_sims=n_sims, seed=seed)
    rng = np.random.default_rng(seed + 1)
    pvals = []
    for _ in range(n_check):
        x = np.cumsum(rng.normal(size=n))
        y = np.cumsum(rng.normal(size=n))  # true null: unrelated
        s, _ = eg_stat(y, x)
        pvals.append(mc_pvalue(s, null_dist))
    pvals = np.array(pvals)
    frac_below_05 = (pvals < 0.05).mean()
    print(f"[sanity check] P(p<0.05 | true null), n_check={n_check}: {frac_below_05:.3f}  "
          f"(should be close to 0.05 if calibration is correct)")
    return frac_below_05


# ---------------------------------------------------------------------------
# Synthetic universe: a few genuinely cointegrated pairs buried in noise
# ---------------------------------------------------------------------------

def make_synthetic_universe(n_true_pairs=4, n_noise=40, n_bars=400,
                             half_life=10.0, seed=42):
    rng = np.random.default_rng(seed)
    assets = {}
    true_pairs = []
    phi = 0.5 ** (1.0 / half_life)  # AR(1) coeff giving exactly this half-life
    for k in range(n_true_pairs):
        trend = np.cumsum(rng.normal(0, 1, n_bars))
        spread = np.zeros(n_bars)
        for t in range(1, n_bars):
            spread[t] = phi * spread[t - 1] + rng.normal(0, 0.5)
        name_a, name_b = f"TRUE{k}_A", f"TRUE{k}_B"
        assets[name_a] = trend + 0.5 * spread
        assets[name_b] = trend - 0.5 * spread
        true_pairs.append(tuple(sorted([name_a, name_b])))
    for j in range(n_noise):
        assets[f"NOISE{j}"] = np.cumsum(rng.normal(0, 1, n_bars))
    return assets, true_pairs, phi


# ---------------------------------------------------------------------------
# Run the experiment: naive p<0.05 vs FDR-corrected, on the SAME data
# ---------------------------------------------------------------------------

def run_experiment():
    print(f"{'='*70}\nSANITY CHECK: does the p-value machinery behave correctly?\n{'='*70}")
    _sanity_check_uniform_pvalues()

    print(f"\n{'='*70}\nEXPERIMENT: naive vs FDR-corrected pair selection\n{'='*70}")
    assets, true_pairs, true_phi = make_synthetic_universe(n_noise=24)
    names = list(assets.keys())
    n_bars = len(next(iter(assets.values())))
    pairs = list(combinations(names, 2))
    n_sims = 40000
    required_resolution = (1 / len(pairs)) * 0.05
    print(f"Universe: {len(names)} assets ({len(true_pairs)} true cointegrated pairs, "
          f"{len(names) - 2*len(true_pairs)} pure-noise assets)")
    print(f"Total candidate pairs tested: {len(pairs)}")
    print(f"True embedded half-life: {np.log(2)/(-np.log(true_phi)):.1f} bars")
    print(f"BH rank-1 threshold at this m: {required_resolution:.2e}  |  "
          f"MC resolution at n_sims={n_sims}: {1/n_sims:.2e}\n")

    t0 = time.time()
    null_dist = calibrate_null(n_bars, kind="eg", n_sims=n_sims, seed=1)
    print(f"Null calibration: {time.time()-t0:.1f}s for {n_sims} sims (reused across all {len(pairs)} pairs)")

    t0 = time.time()
    records = []
    for a, b in pairs:
        stat, beta = eg_stat(assets[a], assets[b])
        p = mc_pvalue(stat, null_dist)
        is_true = tuple(sorted([a, b])) in true_pairs
        records.append({"pair": (a, b), "stat": stat, "pvalue": p, "beta": beta, "is_true_pair": is_true})
    df = pd.DataFrame(records)
    print(f"Tested all {len(pairs)} pairs in {time.time()-t0:.1f}s\n")

    n_true = df["is_true_pair"].sum()
    n_noise_pairs = len(df) - n_true

    # --- naive uncorrected p<0.05 ---
    naive_selected = df[df["pvalue"] < 0.05]
    naive_true_found = naive_selected["is_true_pair"].sum()
    naive_false_pos = len(naive_selected) - naive_true_found

    # --- FDR corrected ---
    df["fdr_reject"] = benjamini_hochberg(df["pvalue"].values, alpha=0.05)
    fdr_selected = df[df["fdr_reject"]]
    fdr_true_found = fdr_selected["is_true_pair"].sum()
    fdr_false_pos = len(fdr_selected) - fdr_true_found

    print(f"{'-'*70}")
    print(f"Ground truth: {n_true} true cointegrated pairs, {n_noise_pairs} noise pairs (no real relationship)\n")
    print(f"{'Method':<28}{'Selected':>10}{'True found':>12}{'False positives':>18}")
    print(f"{'-'*70}")
    print(f"{'Naive p<0.05':<28}{len(naive_selected):>10}{naive_true_found:>12}/{n_true:<3}{naive_false_pos:>15}")
    print(f"{'FDR-corrected (BH, a=0.05)':<28}{len(fdr_selected):>10}{fdr_true_found:>12}/{n_true:<3}{fdr_false_pos:>15}")
    print(f"{'-'*70}")
    expected_naive_fp = 0.05 * n_noise_pairs
    print(f"\nExpected false positives from {n_noise_pairs} noise pairs at p<0.05 by chance alone: "
          f"~{expected_naive_fp:.0f}")
    print(f"Actually observed with naive thresholding: {naive_false_pos}")
    print(f"\nThis is the exact failure mode in the original script: it applies a flat "
          f"p<{0.05} cutoff across every pair in the universe with zero correction for "
          f"how many pairs were tested. At ~{len(pairs)} candidate pairs, that's not a "
          f"detail -- it's the dominant source of 'signal' in the output.")
    return df


if __name__ == "__main__":
    run_experiment()
