"""Offline evaluation from the marketplace's own event log: what a team can compute before an A/B test.

Every metric replays the logged test searches through a design and scores its ranking against what the buyer did.
They differ in the catalogue the design is allowed to see and in how clicks are weighted:

    ndcg_clicks     clicks as relevance, unweighted (the default in most offline harnesses)
    ndcg_ips        clicks weighted by 1 / how likely their logged position was to be looked at
    recall_pit      the bought item in the top 20, ranking only what was for sale at the time (point in time)
    recall_today    the same against today's tables: every listing, sold or not yet listed, and buyer profiles that
                    already contain the click being predicted (the trap)

Nothing here reads the simulation: only the parquet log, the listings table and the search stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from preloved.search.stack import Design, Stack


@dataclass
class OfflineSet:
    searches: list[tuple]  # (search_id, day, user_id, market, query)
    clicks: dict[int, list[tuple[int, int]]]  # search -> [(item, logged position)]
    purchases: dict[int, int]  # search -> item
    listings: list[dict]
    listed: np.ndarray
    sold: np.ndarray  # day sold here, or a large number
    gone: np.ndarray  # day it left otherwise, or a large number
    day_clicks: dict[int, list[tuple[int, int]]]  # day -> [(user, item)], the log the profiles are built from

    def available(self, day: int) -> np.ndarray:
        """For sale on `day`, as the listings table says now: an item sold that day was still for sale before."""
        return (self.listed <= day) & (self.sold >= day) & (self.gone > day)


def load(logs: Path, days: range, listing_cols: list[str]) -> OfflineSet:
    db = duckdb.connect()
    for t in ("searches", "impressions", "clicks", "purchases", "listings"):
        db.execute(f"create view {t} as select * from read_parquet('{logs / f'{t}.parquet'}')")
    lo, hi = days.start, days.stop
    searches = db.execute(f"""select search_id, day, user_id, market, query from searches
                              where day >= {lo} and day < {hi} order by search_id""").fetchall()
    clicks: dict[int, list] = {}
    for sid, item, pos in db.execute(f"""
            select c.search_id, c.item_id, i.pos from clicks c join searches s using (search_id)
            join impressions i on i.search_id = c.search_id and i.item_id = c.item_id
            where s.day >= {lo} and s.day < {hi} order by c.search_id, i.pos""").fetchall():
        clicks.setdefault(int(sid), []).append((int(item), int(pos)))
    purchases = {
        int(s): int(i)
        for s, i in db.execute(f"""
            select p.search_id, p.item_id from purchases p join searches s using (search_id)
            where s.day >= {lo} and s.day < {hi}""").fetchall()
    }
    day_clicks: dict[int, list] = {}
    for d, u, i in db.execute("""select s.day, s.user_id, c.item_id from clicks c join searches s using (search_id)
                                 order by s.day, c.search_id, c.item_id""").fetchall():
        day_clicks.setdefault(int(d), []).append((int(u), int(i)))
    df = db.execute(f"select {', '.join(listing_cols)}, sold_day, gone_day from listings order by item_id").df()
    big = 10**6
    listed = df["listed_day"].to_numpy(dtype=np.int64)
    sold = df["sold_day"].fillna(big).to_numpy(dtype=np.int64)
    gone = df["gone_day"].fillna(big).to_numpy(dtype=np.int64)
    rows = df[listing_cols].to_dict("records")
    listings = [{k: (None if v != v else v) for k, v in r.items()} for r in rows]
    for x in listings:
        x["item_id"], x["listed_day"] = int(x["item_id"]), int(x["listed_day"])
    return OfflineSet([tuple(r) for r in searches], clicks, purchases, listings, listed, sold, gone, day_clicks)


def replay(stack: Stack, designs: dict[str, Design], off: OfflineSet, depth: int = 20) -> dict[str, dict[int, list]]:
    """Point in time: the stack sees listings as they arrive and the clicks before each day, and ranks only what
    was for sale. `stack` must hold the listings and clicks from before the first test day."""
    out: dict[str, dict[int, list]] = {n: {} for n in designs}
    days = sorted({s[1] for s in off.searches})
    for day in days:
        n = len(stack.cat.listings)
        new = [x for x in off.listings[n:] if x["listed_day"] <= day]
        if new:
            stack.add_listings(new)
        avail = off.available(day)
        for sid, _d, user, market, query in (s for s in off.searches if s[1] == day):
            for name, des in designs.items():
                out[name][sid] = stack.search(des, query, user, market, day, avail).items[:depth]
        stack.observe_clicks(off.day_clicks.get(day, []))
    return out


def replay_today(
    stack: Stack, designs: dict[str, Design], off: OfflineSet, sids: set[int], depth: int = 20
) -> dict[str, dict[int, list]]:
    """Against today's tables: `stack` holds every listing and every click in the log; nothing is filtered."""
    everything = np.ones(len(stack.cat.listings), dtype=bool)
    out: dict[str, dict[int, list]] = {n: {} for n in designs}
    for sid, d, user, market, query in (s for s in off.searches if s[0] in sids):
        for name, des in designs.items():
            out[name][sid] = stack.search(des, query, user, market, d, everything).items[:depth]
    return out


def ndcg(ranked: list[int], gains: dict[int, float], k: int = 20) -> float:
    if not gains:
        return 0.0
    dcg = sum(gains.get(i, 0.0) / np.log2(r + 2) for r, i in enumerate(ranked[:k]))
    ideal = sum(g / np.log2(r + 2) for r, g in enumerate(sorted(gains.values(), reverse=True)[:k]))
    return float(dcg / ideal) if ideal > 0 else 0.0


def metrics(ranked: dict[int, list], off: OfflineSet, exam: np.ndarray, today: dict[int, list] | None = None) -> dict:
    with_clicks = [s for s in ranked if s in off.clicks]
    naive = [ndcg(ranked[s], {i: 1.0 for i, _ in off.clicks[s]}) for s in with_clicks]
    ips = [ndcg(ranked[s], {i: 1.0 / exam[min(p, len(exam) - 1)] for i, p in off.clicks[s]}) for s in with_clicks]
    bought = [s for s in ranked if s in off.purchases]
    out = {
        "ndcg_clicks": float(np.mean(naive)),
        "ndcg_ips": float(np.mean(ips)),
        "recall_pit": float(np.mean([off.purchases[s] in ranked[s] for s in bought])),
        "searches": len(ranked),
        "with_clicks": len(with_clicks),
        "with_purchase": len(bought),
    }
    if today is not None:
        out["recall_today"] = float(np.mean([off.purchases[s] in today[s] for s in bought]))
    return out
