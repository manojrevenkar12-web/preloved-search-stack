"""Team-draft interleaving: one results page built from two rankings, each click credited to the ranking that put the
item there. A buyer comparing both on the same query needs far fewer searches to show which is better than an A/B
split, where each buyer sees only one."""

from __future__ import annotations

import numpy as np


def team_draft(a: list[int], b: list[int], k: int, seed: int) -> tuple[list[int], dict[int, str]]:
    rng = np.random.default_rng(seed)
    out: list[int] = []
    team: dict[int, str] = {}
    ia = ib = na = nb = 0
    while len(out) < k and (ia < len(a) or ib < len(b)):
        pick_a = na < nb or (na == nb and rng.random() < 0.5)
        if pick_a and ia >= len(a):
            pick_a = False
        if not pick_a and ib >= len(b):
            pick_a = True
        src, i = (a, ia) if pick_a else (b, ib)
        while i < len(src) and src[i] in team:
            i += 1
        if i < len(src):
            out.append(src[i])
            team[src[i]] = "a" if pick_a else "b"
            na, nb = (na + 1, nb) if pick_a else (na, nb + 1)
        if pick_a:
            ia = i + 1
        else:
            ib = i + 1
    return out, team


def preference(wins_a: np.ndarray, wins_b: np.ndarray) -> dict:
    """Share of decided searches won by B, with a binomial interval."""
    decided = wins_a != wins_b
    n = int(decided.sum())
    p = float((wins_b[decided] > wins_a[decided]).mean()) if n else 0.5
    se = float(np.sqrt(p * (1 - p) / n)) if n else 0.5
    return {"decided": n, "b_share": round(p, 4), "low": round(p - 1.96 * se, 4), "high": round(p + 1.96 * se, 4)}
