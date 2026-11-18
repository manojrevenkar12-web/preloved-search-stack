"""The search service: FastAPI over the composable stack. One process serves every design; the design is a name.

    uvicorn preloved.service.app:app            (reads PRELOVED_HOME, default the repo root, and PRELOVED_WORLD)

It loads what production would have: the listings table, the click log for buyer profiles, the trained models and
the design configs. It never imports the simulation (a test checks).
"""

from __future__ import annotations

import json
import os
import time
import tomllib
from pathlib import Path

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from preloved.search.ranker import Ranker
from preloved.search.stack import Design, Stack
from preloved.search.vectors import TwoTower

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


def load_stack(home: Path, world: str) -> tuple[Stack, np.ndarray, int]:
    s = tomllib.loads((home / "config" / "stack.toml").read_text())
    m = home / "models" / world
    stack = Stack(
        s,
        towers={k: TwoTower.load(m / k) for k in ("content", "ids")},
        rankers={k: Ranker.load(m / k) for k in ("ips", "naive", "anonymous")},
    )
    logs = home / "data" / world / "logs"
    df = duckdb.sql(
        f"select {', '.join(LISTING_COLS)}, sold_day, gone_day "
        f"from read_parquet('{logs / 'listings.parquet'}') order by item_id"
    ).df()
    rows = [{k: (None if v != v else v) for k, v in r.items()} for r in df[LISTING_COLS].to_dict("records")]
    for x in rows:
        x["item_id"], x["listed_day"] = int(x["item_id"]), int(x["listed_day"])
    stack.add_listings(rows)
    clicks = duckdb.sql(f"""select s.user_id, c.item_id from read_parquet('{logs / "clicks.parquet"}') c
                            join read_parquet('{logs / "searches.parquet"}') s using (search_id)
                            order by c.search_id, c.item_id""").fetchall()
    stack.observe_clicks([(int(u), int(i)) for u, i in clicks])
    today = int(df["listed_day"].max())
    available = (df["sold_day"].isna() & df["gone_day"].isna()).to_numpy()
    return stack, available, today


def create_app(
    stack: Stack, designs: dict[str, Design], available: np.ndarray, today: int, report: dict | None = None
) -> FastAPI:
    app = FastAPI(title="preloved-search-stack", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "listings": len(stack.cat.listings), "for_sale": int(available.sum()), "day": today}

    @app.get("/designs")
    def list_designs() -> list[dict]:
        return [
            {
                "name": d.name,
                "understand": d.understand,
                "index": d.index,
                "hard": list(d.hard),
                "vectors": d.vectors,
                "fusion": d.fusion,
                "ranker": d.ranker,
                "depth": d.depth,
            }
            for d in designs.values()
        ]

    @app.get("/search")
    def search(
        q: str = Query(min_length=1, max_length=200), design: str = "personalised", user: int = -1, market: str = "FR"
    ) -> dict:
        if design not in designs:
            raise HTTPException(404, f"no design named {design!r}; GET /designs lists them")
        t0 = time.perf_counter()
        r = stack.search(designs[design], q, user, market, today, available)
        return {
            "query": q,
            "design": design,
            "understood": r.parsed.slots() if r.parsed else None,
            "corrected": [list(f) for f in r.parsed.fuzzy] if r.parsed else [],
            "relaxed": r.relaxed,
            "candidates": r.candidates,
            "ms": round((time.perf_counter() - t0) * 1000, 2),
            "items": [stack.cat.listings[i] for i in r.items],
        }

    @app.get("/report")
    def get_report() -> dict:
        if report is None:
            raise HTTPException(404, "no report yet: run scripts/evaluate.py")
        return report

    return app


def _default() -> FastAPI:
    home = Path(os.environ.get("PRELOVED_HOME", Path(__file__).resolve().parents[3]))
    world = os.environ.get("PRELOVED_WORLD", "dev")
    cfg = tomllib.loads((home / "config" / "designs.toml").read_text())
    designs = {n: Design.from_config(n, c) for n, c in {**cfg["designs"], **cfg.get("variants", {})}.items()}
    stack, available, today = load_stack(home, world)
    rp = home / "reports" / f"{world}.json"
    return create_app(stack, designs, available, today, json.loads(rp.read_text()) if rp.exists() else None)


def __getattr__(name: str):  # `uvicorn preloved.service.app:app` builds the app on first access, not on import
    if name == "app":
        globals()["app"] = _default()
        return globals()["app"]
    raise AttributeError(name)
