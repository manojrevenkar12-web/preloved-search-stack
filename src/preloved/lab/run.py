"""Running the marketplace through a design: the bridge between the simulated world and the search stack.

The stack gets what the marketplace would give it: listings as they arrive, which items are still for sale, the
buyer's query, and afterwards its own event log (clicks). The world answers each results page.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from preloved.measure.interleave import team_draft
from preloved.search.stack import Design, Stack
from preloved.world.sim import World, _mix


def sync_listings(world: World, stack: Stack, day: int) -> None:
    n = len(stack.cat.listings)
    new = [it.listing() for it in world.items[n:] if it.listed_day <= day]
    if new:
        stack.add_listings(new)


def run_days(
    world: World,
    stack: Stack,
    days: range,
    policy: Callable[[object], tuple[str, Design]],
    randomized_share: float = 0.0,
    interleave: tuple[Design, Design] | None = None,
) -> list[dict]:
    """Serve every search of `days`. `policy(search)` names the arm and its design (an A/B split is a policy that
    looks at the user). With `interleave`, each page is a team-draft mix of the two designs instead."""
    out = []
    for day in days:
        sync_listings(world, stack, day)
        avail = world.available_mask(day)
        day_clicks = []
        for s in world.searches_for_day(day):
            t0 = time.perf_counter()
            teams = None
            if interleave:
                a = stack.search(interleave[0], s.query, s.user_id, s.market, day, avail).items
                b = stack.search(interleave[1], s.query, s.user_id, s.market, day, avail).items
                ranked, teams = team_draft(a, b, stack.s["page_size"], seed=s.search_id)
                arm = "interleaved"
            else:
                arm, design = policy(s)
                ranked = stack.search(design, s.query, s.user_id, s.market, day, avail).items
            ms = (time.perf_counter() - t0) * 1000
            randomized = bool(_mix(np.array([s.search_id]), 0, 9)[0] < randomized_share)
            if randomized:  # the top 10 shuffled: how position bias is measured without guessing
                head = ranked[:10]
                np.random.default_rng(s.search_id).shuffle(head)
                ranked = head + ranked[10:]
            res = world.serve(s, ranked, arm, randomized)
            for i in res["purchases"]:
                avail[i] = False
            day_clicks += [(s.user_id, i) for i in res["clicks"]]
            rec = {
                "search_id": s.search_id,
                "day": day,
                "user_id": s.user_id,
                "arm": arm,
                "ms": ms,
                "shown": len(res["shown"]),
                "clicks": len(res["clicks"]),
                "purchases": len(res["purchases"]),
                "gmv": res["gmv"],
                "relevant10": sum(world.relevance(s.intent, world.items[i]) > 0 for i in res["shown"][:10]),
            }
            if teams is not None:
                rec["wins_a"] = sum(teams.get(i) == "a" for i in res["clicks"])
                rec["wins_b"] = sum(teams.get(i) == "b" for i in res["clicks"])
            out.append(rec)
        stack.observe_clicks(day_clicks)
        world.end_of_day(day)
    return out


def write_log(world: World, directory: Path) -> None:
    """The marketplace's event log and listings, as the data warehouse would hold them."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in world.log.items():
        pd.DataFrame(rows).to_parquet(directory / f"{name}.parquet", index=False)
    pd.DataFrame([it.record() for it in world.items]).to_parquet(directory / "listings.parquet", index=False)
