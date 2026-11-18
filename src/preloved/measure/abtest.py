"""A/B tests on a ratio metric (purchases per search), randomised by buyer.

A search is not independent of the buyer's other searches, so the variance is the delta-method variance of a ratio
of per-buyer sums. CUPED subtracts what each buyer's purchases before the test already predict.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _linearised(y: np.ndarray, n: np.ndarray) -> tuple[float, np.ndarray]:
    r = y.sum() / n.sum()
    return float(r), (y - r * n) / n.mean()


def ratio_test(y_a, n_a, y_b, n_b, x_a=None, x_b=None, alpha: float = 0.05) -> dict:
    """y: per-buyer purchases, n: per-buyer searches, x: per-buyer pre-period covariate (CUPED when given).
    Returns the relative lift of B over A with its interval and p-value."""
    y_a, n_a, y_b, n_b = (np.asarray(v, dtype=float) for v in (y_a, n_a, y_b, n_b))
    r_a, l_a = _linearised(y_a, n_a)
    r_b, l_b = _linearised(y_b, n_b)
    if x_a is not None:
        x_a, x_b = np.asarray(x_a, dtype=float), np.asarray(x_b, dtype=float)
        x_all, l_all = np.concatenate([x_a, x_b]), np.concatenate([l_a, l_b])
        theta = float(np.cov(l_all, x_all)[0, 1] / x_all.var(ddof=1)) if x_all.var() > 0 else 0.0
        mx = x_all.mean()
        r_a, r_b = r_a - theta * (x_a.mean() - mx), r_b - theta * (x_b.mean() - mx)
        l_a, l_b = l_a - theta * (x_a - x_a.mean()), l_b - theta * (x_b - x_b.mean())
    diff = r_b - r_a
    se = float(np.sqrt(l_a.var(ddof=1) / len(l_a) + l_b.var(ddof=1) / len(l_b)))
    z = diff / se if se > 0 else 0.0
    q = stats.norm.ppf(1 - alpha / 2)
    return {
        "control": r_a,
        "treatment": r_b,
        "lift": diff / r_a,
        "low": (diff - q * se) / r_a,
        "high": (diff + q * se) / r_a,
        "z": z,
        "p": float(2 * stats.norm.sf(abs(z))),
        "buyers": len(y_a) + len(y_b),
    }


def split(users: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic 50/50 assignment of buyers to B."""
    with np.errstate(over="ignore"):
        z = users.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(
            (salt * 0xBF58476D1CE4E5B9 + 1) & 0xFFFFFFFFFFFFFFFF
        )
        z ^= z >> np.uint64(31)
        z *= np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(29)
    return (z & np.uint64(1)).astype(bool)


def aa_false_positives(users, y, n, x, splits: int, alpha: float) -> dict:
    """Split one arm's buyers into two fake arms many times: the share called significant should be about alpha."""
    users, y, n, x = (np.asarray(v) for v in (users, y, n, x))
    hits = {"plain": 0, "cuped": 0}
    for s in range(splits):
        b = split(users, 1000 + s)
        hits["plain"] += ratio_test(y[~b], n[~b], y[b], n[b], alpha=alpha)["p"] < alpha
        hits["cuped"] += ratio_test(y[~b], n[~b], y[b], n[b], x[~b], x[b], alpha=alpha)["p"] < alpha
    return {k: v / splits for k, v in hits.items()} | {"splits": splits, "alpha": alpha}


def searches_needed(z: float, n: int, power_z: float = 2.8) -> float:
    """Searches for an 80%-power test at 5%, scaled from the z a test reached on n searches."""
    return float("inf") if z == 0 else n * (power_z / abs(z)) ** 2
