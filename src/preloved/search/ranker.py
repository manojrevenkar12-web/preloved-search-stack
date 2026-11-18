"""Personalised re-ranking at serving time (NumPy). Training is in `preloved.search.ranker_train` (PyTorch).

Features of a candidate for a query and a buyer:
- retrieval: BM25 and vector scores, each scaled within the candidate set
- slots: does the listing match the brand, model, category, colour, material, style and size the query asked for,
  read from the seller's form and, where the form is empty, from the title
- price: log price, and how far it sits from what this buyer usually clicks
- condition, days since listing, same market as the buyer
- the buyer: share of their past clicks on this brand and this category, and whether they have any history
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preloved.search.qu import Parsed
from preloved.taxonomy import CONDITIONS

FEATURES = [
    "bm25",
    "vector",
    "brand",
    "model",
    "category",
    "colour",
    "material",
    "style",
    "size",
    "log_price",
    "price_gap",
    "price_gap_abs",
    "condition",
    "age",
    "same_market",
    "user_brand",
    "user_category",
    "has_history",
]
SLOTS = ("brand", "model", "category", "colour", "material", "style", "size")


@dataclass
class Profile:
    """What the marketplace knows about a buyer from their own past clicks (point in time)."""

    clicks: int = 0
    brands: dict[str, int] = field(default_factory=dict)
    categories: dict[str, int] = field(default_factory=dict)
    log_prices: list[float] = field(default_factory=list)

    def median_log_price(self) -> float | None:
        return float(np.median(self.log_prices)) if self.log_prices else None


class Catalogue:
    """Listings the stack has seen, as arrays: slots from the seller's form, filled from the title when empty."""

    def __init__(self) -> None:
        self.listings: list[dict] = []
        self.slots: dict[str, list] = {k: [] for k in SLOTS}
        self.log_price: list[float] = []
        self.condition: list[int] = []
        self.listed_day: list[int] = []
        self.market: list[str] = []

    def add(self, listing: dict, title_slots: dict) -> None:
        self.listings.append(listing)
        for k in SLOTS:
            self.slots[k].append(listing.get(k) or title_slots.get(k))
        self.log_price.append(float(np.log(max(listing["price"], 1.0))))
        self.condition.append(CONDITIONS.index(listing["condition"]))
        self.listed_day.append(listing["listed_day"])
        self.market.append(listing["market"])

    def arrays(self) -> None:
        self.a_slots = {k: np.array(v, dtype=object) for k, v in self.slots.items()}
        self.a_log_price = np.array(self.log_price)
        self.a_condition = np.array(self.condition, dtype=np.float32)
        self.a_listed = np.array(self.listed_day)
        self.a_market = np.array(self.market, dtype=object)
        self.a_brand = self.a_slots["brand"]
        self.a_category = self.a_slots["category"]


def features(
    cat: Catalogue,
    cand: np.ndarray,
    parsed: Parsed,
    prof: Profile | None,
    market: str,
    day: int,
    bm25: np.ndarray,
    vec: np.ndarray,
) -> np.ndarray:
    n = len(cand)
    x = np.zeros((n, len(FEATURES)), dtype=np.float32)

    def scaled(v: np.ndarray) -> np.ndarray:
        top = v.max() if len(v) else 0.0
        return v / top if top > 0 else v

    x[:, 0], x[:, 1] = scaled(bm25[cand]), vec[cand]
    asked = parsed.slots()
    for j, k in enumerate(SLOTS, start=2):
        if k in asked:
            have = cat.a_slots[k][cand]
            x[:, j] = np.where(have == asked[k], 1.0, np.where(have == None, 0.0, -1.0))  # noqa: E711
    lp = cat.a_log_price[cand]
    x[:, 9] = lp
    med = prof.median_log_price() if prof else None
    if med is not None:
        x[:, 10], x[:, 11] = lp - med, np.abs(lp - med)
    x[:, 12] = cat.a_condition[cand]
    x[:, 13] = np.log1p(np.maximum(day - cat.a_listed[cand], 0))
    x[:, 14] = cat.a_market[cand] == market
    if prof and prof.clicks:
        b, c = cat.a_brand[cand], cat.a_category[cand]
        x[:, 15] = np.array([prof.brands.get(v, 0) for v in b]) / prof.clicks
        x[:, 16] = np.array([prof.categories.get(v, 0) for v in c]) / prof.clicks
        x[:, 17] = 1.0
    return x


class Ranker:
    def __init__(self, weights: dict[str, np.ndarray], card: dict):
        self.w1, self.b1, self.w2, self.b2 = weights["w1"], weights["b1"], weights["w2"], weights["b2"]
        self.mean, self.std = weights["mean"], weights["std"]
        self.card = card
        self.drop = [FEATURES.index(f) for f in card.get("without", [])]

    @classmethod
    def load(cls, directory: Path) -> Ranker:
        with np.load(directory / "ranker.npz") as z:
            return cls({k: z[k] for k in z.files}, json.loads((directory / "card.json").read_text()))

    def score(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        if self.drop:
            z[:, self.drop] = 0.0
        h = np.maximum(z @ self.w1.T + self.b1, 0.0)
        return (h @ self.w2.T + self.b2).ravel()
