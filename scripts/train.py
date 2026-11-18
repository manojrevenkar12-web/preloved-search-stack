"""Train the two towers and the rankers from the event log of the first weeks.

    python3 scripts/train.py [--world dev|heldout]   -> models/<world>/{content,ids,ips,naive,anonymous}/

The warehouse is queried in SQL (DuckDB over the parquet log). Buyer profiles for the ranker are rebuilt day by day,
so a search's features only know the clicks before its day.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preloved.lab.common import WORLDS, data_dir, models_dir, settings  # noqa: E402
from preloved.search import ranker_train, vectors_train  # noqa: E402
from preloved.search.qu import QueryUnderstanding  # noqa: E402
from preloved.search.ranker import features  # noqa: E402
from preloved.search.stack import Stack  # noqa: E402
from preloved.search.vectors import TwoTower  # noqa: E402

LISTING_COLS = [
    "item_id",
    "title",
    "price",
    "market",
    "listed_day",
    "brand",
    "category",
    "model",
    "material",
    "colour",
    "size",
    "condition",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=list(WORLDS), default="dev")
    args = ap.parse_args()
    s = settings()
    logs, out = data_dir(args.world) / "logs", models_dir(args.world)
    db = duckdb.connect()
    for t in ("searches", "impressions", "clicks", "purchases", "listings"):
        db.execute(f"create view {t} as select * from read_parquet('{logs / f'{t}.parquet'}')")
    train_days = s["timeline"]["train_days"]
    qu = QueryUnderstanding()
    listings = db.execute(f"select {', '.join(LISTING_COLS)} from listings order by item_id").df()
    listings = [{k: (None if v != v else v) for k, v in r.items()} for r in listings.to_dict("records")]
    for x in listings:
        x["item_id"], x["listed_day"] = int(x["item_id"]), int(x["listed_day"])

    # ---- two towers: (query, clicked item) pairs, purchases weighted ----------------------------------------
    pairs_df = db.execute(f"""
        select s.query, c.item_id, case when p.item_id is null then 1.0 else {s["twotower"]["purchase_weight"]} end w
        from clicks c join searches s using (search_id)
        left join purchases p on p.search_id = c.search_id and p.item_id = c.item_id
        where s.day < {train_days} order by c.search_id, c.item_id""").df()
    pairs = [(q, qu.parse(q), listings[int(i)], float(w)) for q, i, w in pairs_df.itertuples(index=False, name=None)]
    t0 = time.perf_counter()
    cards = {
        name: vectors_train.train(pairs, out / name, s["twotower"], ids=(name == "ids")) for name in ("content", "ids")
    }
    print(f"two towers on {len(pairs)} pairs in {time.perf_counter() - t0:.0f} s")

    # ---- position bias from the randomised searches ---------------------------------------------------------
    rnd = db.execute("""
        select i.pos, (c.item_id is not null)::int clicked from impressions i join searches s using (search_id)
        left join clicks c on c.search_id = i.search_id and c.item_id = i.item_id
        where s.randomized and i.pos < 10""").fetchall()
    exam = ranker_train.position_bias(rnd, s["page_size"])

    # ---- ranker lists: logged pages with their features, replayed day by day ----------------------------------
    stack = Stack(s, towers={"content": TwoTower.load(out / "content")})
    stack.add_listings(listings)
    clicked_searches = db.execute(f"""
        select s.search_id, s.day, s.user_id, s.market, s.query from searches s
        where s.day < {train_days} and exists (select 1 from clicks c where c.search_id = s.search_id)
        order by s.search_id""").fetchall()
    imp = db.execute(f"""
        select i.search_id, i.item_id, i.pos, (c.item_id is not null)::int clicked, (p.item_id is not null)::int bought
        from impressions i join searches s using (search_id)
        left join clicks c on c.search_id = i.search_id and c.item_id = i.item_id
        left join purchases p on p.search_id = i.search_id and p.item_id = i.item_id
        where s.day < {train_days} order by i.search_id, i.pos""").fetchnumpy()
    day_clicks = db.execute(f"""
        select s.day, s.user_id, c.item_id from clicks c join searches s using (search_id)
        where s.day < {train_days} order by s.day, c.search_id, c.item_id""").fetchall()
    by_search: dict[int, slice] = {}
    sid = imp["search_id"]
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(sid)], strict=True):
        by_search[int(sid[a])] = slice(int(a), int(b))
    clicks_by_day: dict[int, list] = {}
    for d, u, i in day_clicks:
        clicks_by_day.setdefault(int(d), []).append((int(u), int(i)))
    lists, day = [], -1
    tower = stack.towers["content"]
    for search_id, d, user, market, query in clicked_searches:
        while day < d:  # profiles know the clicks before this search's day
            if day >= 0:
                stack.observe_clicks(clicks_by_day.get(day, []))
            day += 1
        sl = by_search[int(search_id)]
        cand = imp["item_id"][sl].astype(np.int64)
        parsed = qu.parse(query)
        bm = stack.idx["structured"].scores(_tokens(query, parsed))
        vec = np.zeros(len(stack.cat.listings), dtype=np.float32)
        vec[cand] = stack.vecs["content"][cand] @ tower.encode_query(query, parsed)
        x = features(stack.cat, cand, parsed, stack.profiles.get(int(user)), market, int(d), bm, vec)
        lists.append(
            {
                "x": x,
                "pos": imp["pos"][sl].astype(np.int64),
                "click": imp["clicked"][sl].astype(float),
                "buy": imp["bought"][sl].astype(float),
            }
        )
    t0 = time.perf_counter()
    personal = ["user_brand", "user_category", "has_history", "price_gap", "price_gap_abs"]
    for name, ips, without in (("ips", True, None), ("naive", False, None), ("anonymous", True, personal)):
        cards[name] = ranker_train.train(lists, out / name, s["ranker"], exam, ips=ips, without=without)
    print(
        f"rankers on {len(lists)} logged pages in {time.perf_counter() - t0:.0f} s; "
        f"position bias {[round(float(v), 3) for v in exam[:6]]}"
    )
    summary = (
        json.dumps(
            {
                "pairs": len(pairs),
                "pages": len(lists),
                "train_days": train_days,
                "position_bias": [round(float(v), 4) for v in exam],
            },
            indent=2,
        )
        + "\n"
    )
    (out / "training.json").write_text(summary)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"training-{args.world}.json").write_text(summary)  # for the README, which has no models


def _tokens(query, parsed):
    from preloved.search.index import query_tokens

    return query_tokens(query, parsed)


if __name__ == "__main__":
    main()
