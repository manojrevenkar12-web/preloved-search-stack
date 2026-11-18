import numpy as np

from preloved.measure import abtest


def buyers(n, rate, seed, propensity=None, most=6):
    r = np.random.default_rng(seed)
    searches = r.integers(1, most, n)
    p = rate if propensity is None else propensity
    return r.binomial(searches, np.clip(p, 0, 1)).astype(float), searches.astype(float)


def test_detects_a_real_lift_and_not_a_fake_one():
    ya, na = buyers(4000, 0.05, 1)
    yb, nb = buyers(4000, 0.065, 2)
    t = abtest.ratio_test(ya, na, yb, nb)
    assert t["low"] > 0 and abs(t["lift"] - 0.3) < 0.15
    yc, nc = buyers(4000, 0.05, 3)
    assert abtest.ratio_test(ya, na, yc, nc)["low"] < 0 < abtest.ratio_test(ya, na, yc, nc)["high"]


def test_aa_false_positive_rate_is_near_alpha():
    y, n = buyers(3000, 0.05, 4)
    aa = abtest.aa_false_positives(np.arange(3000), y, n, np.zeros(3000), splits=300, alpha=0.05)
    assert 0.02 <= aa["plain"] <= 0.09


def test_cuped_narrows_the_interval_when_the_past_predicts():
    r = np.random.default_rng(5)
    prop = r.gamma(1.0, 0.08, 8000)  # some buyers buy far more often than others, and keep doing so
    pre = r.binomial(50, np.clip(prop, 0, 1)).astype(float)
    y, n = buyers(8000, None, 6, propensity=prop, most=30)
    b = abtest.split(np.arange(8000), 7)
    plain = abtest.ratio_test(y[~b], n[~b], y[b], n[b])
    cuped = abtest.ratio_test(y[~b], n[~b], y[b], n[b], pre[~b], pre[b])
    assert cuped["high"] - cuped["low"] < 0.95 * (plain["high"] - plain["low"])


def test_split_is_deterministic_and_balanced():
    u = np.arange(10000)
    assert (abtest.split(u, 3) == abtest.split(u, 3)).all()
    assert 0.47 < abtest.split(u, 3).mean() < 0.53
    assert 0.4 < (abtest.split(u, 3) == abtest.split(u, 4)).mean() < 0.6


def test_searches_needed_scales_with_z():
    assert abtest.searches_needed(2.8, 1000) == 1000
    assert abtest.searches_needed(5.6, 1000) == 250
