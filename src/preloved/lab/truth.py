"""Lab-only judgments from the world's truth: what a panel of raters would say, for the offline study.

Nothing under preloved.search or preloved.measure may import this module (a test checks).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from preloved.world.sim import World


def intents(config: Path, days: range) -> dict[int, dict]:
    """Rebuild what each logged buyer was looking for. Searches are drawn per day and user, so a fresh world
    replays them exactly; the query text is checked against the log by the caller."""
    w = World.load(config)
    out: dict[int, dict] = {}
    for day in range(days.stop):
        for s in w.searches_for_day(day):
            if day >= days.start:
                out[s.search_id] = {"query": s.query, "intent": s.intent}
    return out


def judged_dcg(world: World, ranked: dict[int, list], truth: dict[int, dict], k: int = 10) -> float:
    """Graded relevance (0-3) of the top k, as a share of a page where every result is exactly what was asked."""
    disc = 1.0 / np.log2(np.arange(k) + 2)
    vals = []
    for sid, items in ranked.items():
        g = np.array([world.relevance(truth[sid]["intent"], world.items[i]) / 3 for i in items[:k]] + [0.0] * k)[:k]
        vals.append(float((g * disc).sum() / disc.sum()))
    return float(np.mean(vals))
