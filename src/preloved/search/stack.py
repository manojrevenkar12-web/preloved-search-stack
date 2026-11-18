"""The multistage search as composable stages. A design is a config of the stages; the same code serves them all.

    understand   query -> slots (preloved.search.qu)
    filter       available items only; hard slots (brand, category) as filters, relaxed if they leave too few
    retrieve     BM25 over the title (today) or over title + canonical slots; vectors from the two towers
    fuse         reciprocal rank fusion of the retrievers, or BM25 alone; a recency boost
    rerank       the personalised ranker over the top of the fused list

Every listing, profile and model here is what the marketplace itself has: listings, its own event log, and models
trained on that log. The world's truth is never imported (a test checks).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from preloved.search.index import BM25Index, query_tokens
from preloved.search.qu import Parsed, QueryUnderstanding
from preloved.search.ranker import Catalogue, Profile, Ranker, features
from preloved.search.vectors import TwoTower


@dataclass
class Design:
    name: str
    understand: bool = False
    index: str = "title"  # "title" (today) or "structured" (title + canonical slots)
    hard: tuple[str, ...] = ()
    relax: bool = True
    vectors: str | None = None  # name of a two-tower model
    fusion: str = "bm25"  # "bm25" or "rrf"
    vector_weight: float = 1.0
    k_lexical: int = 200
    k_vector: int = 200
    recency: float = 0.3
    ranker: str | None = None
    depth: int = 200
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> Design:
        c = dict(cfg)
        if "hard" in c:
            c["hard"] = tuple(c["hard"])
        return cls(name=name, **c)


@dataclass
class Result:
    items: list[int]
    parsed: Parsed | None
    candidates: int
    relaxed: list[str]


class Stack:
    def __init__(
        self, settings: dict, towers: dict[str, TwoTower] | None = None, rankers: dict[str, Ranker] | None = None
    ):
        self.s = settings
        self.qu = QueryUnderstanding()
        self.towers = towers or {}
        self.rankers = rankers or {}
        self.cat = Catalogue()
        self.vecs: dict[str, np.ndarray] = {
            k: np.zeros((0, t.q.shape[1]), dtype=np.float32) for k, t in self.towers.items()
        }
        self.profiles: dict[int, Profile] = defaultdict(Profile)
        self.idx: dict[str, BM25Index] = {}

    # ---- keeping up with the marketplace ----------------------------------------------------------------
    def add_listings(self, listings: list[dict]) -> None:
        for x in listings:
            assert x["item_id"] == len(self.cat.listings), "listings arrive in order"
            self.cat.add(x, self.qu.parse(x["title"]).slots())
        for k, t in self.towers.items():
            if listings:
                self.vecs[k] = np.vstack([self.vecs[k], t.encode_items(listings)])
        self.cat.arrays()
        self.idx = {
            "title": BM25Index(self.cat.listings, structured=False),
            "structured": BM25Index(self.cat.listings, structured=True),
        }

    def observe_clicks(self, clicks: list[tuple[int, int]]) -> None:
        """(user, item) clicks from the event log: the buyer profiles the ranker reads."""
        for user, item in clicks:
            p, x = self.profiles[user], self.cat.listings[item]
            p.clicks += 1
            p.brands[x["brand"]] = p.brands.get(x["brand"], 0) + 1
            p.categories[x["category"]] = p.categories.get(x["category"], 0) + 1
            p.log_prices.append(float(np.log(max(x["price"], 1.0))))

    # ---- serving ----------------------------------------------------------------------------------------
    def search(self, d: Design, query: str, user: int, market: str, day: int, available: np.ndarray) -> Result:
        n = len(self.cat.listings)
        page = self.s["page_size"]
        parsed = self.qu.parse(query) if d.understand else None
        mask = available[:n].copy()
        relaxed: list[str] = []
        if parsed is not None and d.hard:
            slots = parsed.slots()
            active = [h for h in d.hard if h in slots]
            m = mask.copy()
            for h in active:
                m &= self.cat.a_slots[h] == slots[h]
            while d.relax and active and m.sum() < page:  # too few: drop the narrowest filter first
                relaxed.append(active.pop())
                m = mask.copy()
                for h in active:
                    m &= self.cat.a_slots[h] == slots[h]
            mask = m
        bm = self.idx[d.index].scores(query_tokens(query, parsed if d.index == "structured" else None))
        bm = np.where(mask, bm, 0.0)
        lex = np.flatnonzero(bm > 0)
        lex = lex[np.argsort(-bm[lex], kind="stable")[: d.k_lexical]]
        vec_scores = np.zeros(n, dtype=np.float32)
        cand = lex
        if d.vectors:
            qv = self.towers[d.vectors].encode_query(query, parsed)
            vec_scores = self.vecs[d.vectors] @ qv
            ok = np.flatnonzero(mask)
            vtop = ok[np.argsort(-vec_scores[ok], kind="stable")[: d.k_vector]]
            cand = np.unique(np.concatenate([lex, vtop]))
        if not len(cand):
            return Result([], parsed, 0, relaxed)
        if d.fusion == "rrf":
            rank_b = {int(i): r for r, i in enumerate(lex)}
            order_v = cand[np.argsort(-vec_scores[cand], kind="stable")]
            rank_v = {int(i): r for r, i in enumerate(order_v)}
            k = self.s["rrf_k"]
            fused = np.array(
                [
                    (1 / (k + rank_b[int(i)]) if int(i) in rank_b else 0.0) + d.vector_weight / (k + rank_v[int(i)])
                    for i in cand
                ]
            )
        else:
            fused = bm[cand]
        age = np.maximum(day - self.cat.a_listed[cand], 0)
        fused = fused * (1 + d.recency * np.exp(-age / self.s["recency_days"]))
        order = cand[np.argsort(-fused, kind="stable")]
        if d.ranker:
            top = order[: d.depth]
            x = features(
                self.cat, top, parsed or self.qu.parse(""), self.profiles.get(user), market, day, bm, vec_scores
            )
            top = top[np.argsort(-self.rankers[d.ranker].score(x), kind="stable")]
            order = np.concatenate([top, order[d.depth :]])
        return Result([int(i) for i in order[:page]], parsed, len(cand), relaxed)
