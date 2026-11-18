"""The marketplace's own state at the end of the history: listings, buyer profiles from its click log, models."""

from __future__ import annotations

import duckdb

from preloved.lab.common import data_dir, models_dir, settings
from preloved.search.ranker import Ranker
from preloved.search.stack import Stack
from preloved.search.vectors import TwoTower

TOWERS = ("content", "ids")
RANKERS = ("ips", "naive", "anonymous")


def load_models(world: str) -> tuple[dict, dict]:
    m = models_dir(world)
    return ({k: TwoTower.load(m / k) for k in TOWERS}, {k: Ranker.load(m / k) for k in RANKERS})


def stack_after_history(world, name: str, models: tuple[dict, dict] | None = None) -> Stack:
    """A stack that has seen every listing so far and every click in the log, as production would on day one."""
    towers, rankers = models or load_models(name)
    stack = Stack(settings(), towers=towers, rankers=rankers)
    stack.add_listings([it.listing() for it in world.items])
    logs = data_dir(name) / "logs"
    clicks = duckdb.sql(f"""
        select s.user_id, c.item_id from read_parquet('{logs / "clicks.parquet"}') c
        join read_parquet('{logs / "searches.parquet"}') s using (search_id)
        order by c.search_id, c.item_id""").fetchall()
    stack.observe_clicks([(int(u), int(i)) for u, i in clicks])
    return stack
