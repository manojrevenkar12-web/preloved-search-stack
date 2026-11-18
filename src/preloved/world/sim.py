"""A simulated pre-loved luxury marketplace with its own truth.

Sellers list one-of-a-kind items in their own language, with structured attributes that are sometimes missing or
wrong. Buyers have hidden tastes (brands, budget, size, condition) and type queries in their language, with typos,
nicknames and filler words. A results page is examined top-down with a position bias; a click and a purchase depend
on how well the item fits what the buyer wanted and on a hidden appeal (photos, description). A sold item is gone.

Only the evaluation scripts and the tests read `truth`, `Item.true`, `Item.appeal` or `User`. The search stack sees
listings (`Item.listing()`), queries and the event log.
"""

from __future__ import annotations

import copy
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preloved.taxonomy import (
    BAG_SIZES,
    BRANDS,
    CATEGORIES,
    CLOTHING_SIZES,
    COLOURS,
    CONDITIONS,
    MARKET_LANG,
    MATERIALS,
    SHOE_SIZES,
    STYLES,
    category_of_model,
)

# what a brand sells, and how popular it is on the marketplace
BRAND_WEIGHT = {
    "Chanel": 9,
    "Hermès": 6,
    "Louis Vuitton": 10,
    "Dior": 6,
    "Gucci": 8,
    "Prada": 6,
    "Celine": 4,
    "Bottega Veneta": 4,
    "Saint Laurent": 6,
    "Fendi": 4,
    "Loewe": 3,
    "Balenciaga": 4,
    "Burberry": 3,
    "Christian Louboutin": 3,
    "Valentino": 3,
    "Miu Miu": 3,
    "Cartier": 4,
    "Rolex": 3,
    "Max Mara": 3,
    "Zadig & Voltaire": 4,
}
BRAND_CATS = {
    "Chanel": ["bag", "shoes", "jacket", "jewellery"],
    "Hermès": ["bag", "shoes", "jewellery"],
    "Louis Vuitton": ["bag", "shoes"],
    "Dior": ["bag", "shoes", "dress"],
    "Gucci": ["bag", "shoes", "jacket"],
    "Prada": ["bag", "shoes", "jacket"],
    "Celine": ["bag", "jacket"],
    "Bottega Veneta": ["bag", "shoes"],
    "Saint Laurent": ["bag", "shoes", "jacket"],
    "Fendi": ["bag"],
    "Loewe": ["bag"],
    "Balenciaga": ["bag", "shoes"],
    "Burberry": ["jacket"],
    "Christian Louboutin": ["shoes"],
    "Valentino": ["shoes", "dress"],
    "Miu Miu": ["bag", "shoes"],
    "Cartier": ["jewellery", "watch"],
    "Rolex": ["watch"],
    "Max Mara": ["jacket"],
    "Zadig & Voltaire": ["bag", "jacket", "dress"],
}
BRAND_PRICE = {
    "Chanel": 3200,
    "Hermès": 6500,
    "Louis Vuitton": 900,
    "Dior": 1800,
    "Gucci": 800,
    "Prada": 700,
    "Celine": 1200,
    "Bottega Veneta": 1300,
    "Saint Laurent": 900,
    "Fendi": 1000,
    "Loewe": 1100,
    "Balenciaga": 600,
    "Burberry": 600,
    "Christian Louboutin": 350,
    "Valentino": 450,
    "Miu Miu": 600,
    "Cartier": 4500,
    "Rolex": 8000,
    "Max Mara": 700,
    "Zadig & Voltaire": 180,
}
CAT_MATERIALS = {
    "bag": ["lambskin", "canvas", "leather", "suede", "patent"],
    "shoes": ["leather", "suede", "patent"],
    "jacket": ["wool", "tweed", "leather", "silk"],
    "dress": ["silk", "wool"],
    "jewellery": ["gold", "steel"],
    "watch": ["steel", "gold"],
}
BRAND_MATERIALS = {
    "Chanel": ["lambskin", "caviar", "tweed"],
    "Hermès": ["togo", "epsom", "leather"],
    "Louis Vuitton": ["canvas"],
    "Dior": ["canvas", "lambskin"],
    "Gucci": ["canvas", "leather"],
}
COLOUR_WEIGHT = {
    "black": 30,
    "beige": 12,
    "brown": 12,
    "white": 7,
    "red": 6,
    "pink": 6,
    "blue": 8,
    "green": 4,
    "gold_colour": 5,
    "grey": 6,
}
CAT_WEIGHT = {"bag": 45, "shoes": 20, "jacket": 12, "dress": 7, "jewellery": 9, "watch": 7}
FILLER = {
    "en": ["authentic", "pre-owned", "used"],
    "fr": ["authentique", "occasion", "seconde main"],
    "de": ["original", "gebraucht", "echt"],
    "it": ["originale", "usato", "usata"],
    "es": ["original", "usado", "segunda mano"],
}
# how people write brands and models that the catalogue vocabulary does not list
NICKNAMES = {
    "Bottega Veneta": ["bv"],
    "Christian Louboutin": ["loubs", "louboutins"],
    "Chanel": ["coco chanel"],
    "Hermès": ["hermes paris"],
    "Saint Laurent": ["slp"],
    "Max Mara": ["maxmara coat"],
}


def _mix(a: np.ndarray, b: np.ndarray | int, salt: int) -> np.ndarray:
    """splitmix64 of (a, b, salt): a uniform draw in [0, 1) keyed by what it is about, the same for every design."""
    with np.errstate(over="ignore"):  # wrap-around is the point of the hash
        x = (np.asarray(a, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (
            np.asarray(b, dtype=np.uint64) + np.uint64(salt) * np.uint64(0xBF58476D1CE4E5B9)
        )
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


@dataclass
class Item:
    item_id: int
    true: dict  # brand, model, category, material, colour, style, size, condition
    price: float
    market: str
    lang: str
    title: str
    listed_day: int
    appeal: float  # photos and description: hidden
    shown: dict  # the structured attributes the seller filled in (some missing, some wrong)
    sold_day: int | None = None
    gone_day: int | None = None  # sold elsewhere or expired

    def available(self, day: int) -> bool:
        return self.listed_day <= day and self.sold_day is None and (self.gone_day is None or self.gone_day > day)

    def listing(self) -> dict:
        """What the marketplace knows about the item: never the truth."""
        return {
            "item_id": self.item_id,
            "title": self.title,
            "price": round(self.price, 2),
            "market": self.market,
            "listed_day": self.listed_day,
            **self.shown,
        }

    def record(self) -> dict:
        """The listing with its lifecycle as the warehouse holds it: when it sold here, when it left otherwise."""
        return {**self.listing(), "sold_day": self.sold_day, "gone_day": self.gone_day}


@dataclass
class User:
    user_id: int
    market: str
    lang: str
    brands: dict[str, float]
    cats: dict[str, float]
    budget: float
    shoe_size: str
    clothing_size: str
    min_condition: int
    pre_purchases: float = 0.0  # purchases per search before the experiment (for CUPED)


@dataclass
class Search:
    search_id: int
    day: int
    user_id: int
    market: str
    query: str
    intent: dict = field(repr=False)


class World:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = int(cfg["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.items: list[Item] = []
        self.users: list[User] = []
        self.day = 0
        self.next_search = 0
        self.log: dict[str, list] = {"searches": [], "impressions": [], "clicks": [], "purchases": []}
        self._users()
        for _ in range(cfg["catalog"]["initial"]):
            self._list_item(day=-int(self.rng.integers(0, cfg["catalog"]["max_age_days"])))

    @classmethod
    def load(cls, path: Path) -> World:
        cfg = tomllib.loads(path.read_text())
        if "base" in cfg:
            base = tomllib.loads((path.parent / cfg.pop("base")).read_text())
            for k, v in cfg.items():
                base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
            cfg = base
        return cls(cfg)

    def fork(self) -> World:
        return copy.deepcopy(self)

    # ---- people and listings -------------------------------------------------------------------------
    def _users(self) -> None:
        c, r = self.cfg["users"], self.rng
        markets, mw = zip(*c["markets"].items(), strict=True)
        brands = list(BRAND_WEIGHT)
        prior = np.array([BRAND_WEIGHT[b] for b in brands], dtype=float)
        for u in range(c["count"]):
            m = str(r.choice(markets, p=np.array(mw) / sum(mw)))
            lang = "en" if r.random() < c["english_share"] else MARKET_LANG[m]
            bw = r.dirichlet(prior * c["brand_concentration"] / prior.sum() * len(brands))
            cw = r.dirichlet([CAT_WEIGHT[k] / 10 for k in CAT_WEIGHT])
            self.users.append(
                User(
                    u,
                    m,
                    lang,
                    dict(zip(brands, bw, strict=True)),
                    dict(zip(CAT_WEIGHT, cw, strict=True)),
                    float(r.lognormal(math.log(c["budget_median"]), 0.8)),
                    str(r.choice(SHOE_SIZES)),
                    str(r.choice(CLOTHING_SIZES)),
                    int(r.integers(1, 4)),
                )
            )

    def _list_item(self, day: int) -> Item:
        r, c = self.rng, self.cfg["catalog"]
        brand = str(r.choice(list(BRAND_WEIGHT), p=np.array(list(BRAND_WEIGHT.values())) / sum(BRAND_WEIGHT.values())))
        cats = BRAND_CATS[brand]
        cat = str(r.choice(cats, p=np.array([CAT_WEIGHT[x] for x in cats]) / sum(CAT_WEIGHT[x] for x in cats)))
        models = [m for m in BRANDS[brand]["models"] if category_of_model(m) == cat]
        model = str(r.choice(models)) if models and r.random() < c["model_share"] else None
        mats = self._materials(brand, cat)
        material = str(r.choice(mats))
        colour = str(
            r.choice(list(COLOUR_WEIGHT), p=np.array(list(COLOUR_WEIGHT.values())) / sum(COLOUR_WEIGHT.values()))
        )
        style = (
            "quilted"
            if brand == "Chanel" and cat == "bag" and r.random() < 0.8
            else ("mini" if r.random() < 0.12 else ("vintage" if r.random() < 0.12 else None))
        )
        size = (
            str(r.choice(SHOE_SIZES))
            if cat == "shoes"
            else str(r.choice(CLOTHING_SIZES))
            if cat in ("jacket", "dress")
            else str(r.choice(BAG_SIZES))
            if model in ("Birkin", "Kelly")
            else None
        )
        condition = int(r.choice(4, p=[0.15, 0.45, 0.3, 0.1]))
        premium = {"Birkin": 2.2, "Kelly": 1.9, "Classic Flap": 1.4, "Submariner": 1.3, "Love": 1.2}.get(
            model or "", 1.0
        )
        price = BRAND_PRICE[brand] * premium * (1.25, 1.0, 0.82, 0.62)[condition] * float(r.lognormal(0, 0.35))
        markets, mw = zip(*self.cfg["users"]["markets"].items(), strict=True)
        market = str(r.choice(markets, p=np.array(mw) / sum(mw)))
        lang = MARKET_LANG[market]
        true = {
            "brand": brand,
            "model": model,
            "category": cat,
            "material": material,
            "colour": colour,
            "style": style,
            "size": size,
            "condition": condition,
        }
        shown = {
            "brand": brand,
            "category": cat,
            "model": model if r.random() > c["missing_attr"] else None,
            "material": material if r.random() > c["missing_attr"] else None,
            "colour": (colour if r.random() > c["wrong_colour"] else str(r.choice(list(COLOUR_WEIGHT))))
            if r.random() > c["missing_attr"]
            else None,
            "size": size,
            "condition": CONDITIONS[condition],
        }
        item = Item(
            len(self.items),
            true,
            price,
            market,
            lang,
            self._title(true, lang, r),
            day,
            float(r.lognormal(0, c["appeal_sigma"])),
            shown,
        )
        self.items.append(item)
        return item

    @staticmethod
    def _materials(brand: str | None, cat: str) -> list[str]:
        """Brand signatures (Chanel caviar, Hermès togo) for bags; the category's usual materials otherwise."""
        if cat == "bag" and brand in BRAND_MATERIALS:
            return BRAND_MATERIALS[brand]
        return CAT_MATERIALS[cat]

    def _title(self, t: dict, lang: str, r: np.random.Generator) -> str:
        words = [t["brand"] if r.random() < 0.8 else BRANDS[t["brand"]]["aliases"][-1]]
        if t["model"] and r.random() < 0.9:
            words.append(str(r.choice(BRANDS[t["brand"]]["models"][t["model"]])))
        words.append(str(r.choice(CATEGORIES[t["category"]][lang])))
        if r.random() < 0.75:
            words.append(str(r.choice(MATERIALS[t["material"]][lang])))
        if r.random() < 0.8:
            words.append(str(r.choice(COLOURS[t["colour"]][lang])))
        if t["style"] and r.random() < 0.7:
            words.append(str(r.choice(STYLES[t["style"]][lang])))
        if t["size"] and r.random() < 0.6:
            words.append(t["size"])
        if r.random() < 0.4:
            words.append(str(r.choice(FILLER[lang])))
        head, tail = words[:2], words[2:]
        r.shuffle(tail)
        return " ".join(head + tail)

    # ---- searches ---------------------------------------------------------------------------------------
    def searches_for_day(self, day: int) -> list[Search]:
        """Who searches today and what they type. Drawn per day and user, so every design gets the same searches."""
        c = self.cfg["searches"]
        out = []
        r = np.random.default_rng([self.seed, 7, day])
        active = np.flatnonzero(r.random(len(self.users)) < c["per_user_day"])
        for uid in active:
            u = self.users[int(uid)]
            ur = np.random.default_rng([self.seed, 11, day, int(uid)])
            for _ in range(1 + int(ur.random() < c["second_search"])):
                intent = self._intent(u, ur)
                out.append(Search(self.next_search, day, u.user_id, u.market, self._query(intent, u, ur), intent))
                self.next_search += 1
        return out

    def _intent(self, u: User, r: np.random.Generator) -> dict:
        c = self.cfg["searches"]
        brand = (
            str(r.choice(list(u.brands), p=np.array(list(u.brands.values()))))
            if r.random() < c["brand_share"]
            else None
        )
        cats = BRAND_CATS[brand] if brand else list(CAT_WEIGHT)
        cw = np.array([u.cats[x] for x in cats])
        cat = str(r.choice(cats, p=cw / cw.sum()))
        models = [m for m in BRANDS[brand]["models"] if category_of_model(m) == cat] if brand else []
        model = str(r.choice(models)) if models and r.random() < c["model_share"] else None
        colour = str(r.choice(list(COLOUR_WEIGHT))) if r.random() < c["colour_share"] else None
        material = None
        if r.random() < c["material_share"]:
            material = str(r.choice(self._materials(brand, cat)))
        style = str(r.choice(["quilted", "mini", "vintage"])) if r.random() < c["style_share"] else None
        size = u.shoe_size if cat == "shoes" else u.clothing_size if cat in ("jacket", "dress") else None
        size = size if r.random() < c["size_share"] else None
        return {
            "brand": brand,
            "model": model,
            "category": cat,
            "colour": colour,
            "material": material,
            "style": style,
            "size": size,
        }

    def _query(self, it: dict, u: User, r: np.random.Generator) -> str:
        c, lang = self.cfg["searches"], u.lang
        words = []
        if it["brand"]:
            forms = BRANDS[it["brand"]]["aliases"] + (
                NICKNAMES.get(it["brand"], []) if r.random() < c["nickname"] else []
            )
            words.append(str(r.choice(forms)))
        if it["model"]:
            words.append(str(r.choice(BRANDS[it["brand"]]["models"][it["model"]])))
        if not it["model"] or r.random() < 0.3:
            words.append(str(r.choice(CATEGORIES[it["category"]][lang])))
        for key, table in (("colour", COLOURS), ("material", MATERIALS), ("style", STYLES)):
            if it[key]:
                words.append(str(r.choice(table[it[key]][lang])))
        if it["size"]:
            words.append(it["size"])
        if r.random() < c["filler"]:
            words.append(str(r.choice(FILLER[lang])))
        r.shuffle(words)
        words = [self._typo(w, r) if r.random() < c["typo"] and len(w) > 4 else w for w in words]
        return " ".join(words)

    @staticmethod
    def _typo(w: str, r: np.random.Generator) -> str:
        i = int(r.integers(1, len(w) - 1))
        kind = r.integers(0, 3)
        if kind == 0:
            return w[:i] + w[i + 1 :]  # a letter dropped
        if kind == 1:
            return w[:i] + w[i] + w[i:]  # a letter doubled
        return w[: i - 1] + w[i] + w[i - 1] + w[i + 1 :]  # two letters swapped

    # ---- the truth -------------------------------------------------------------------------------------
    def relevance(self, it: dict, item: Item) -> int:
        """0 = not what was asked for, 3 = exactly it. Graded on the item's true attributes."""
        t = item.true
        if (
            t["category"] != it["category"]
            or (it["brand"] and t["brand"] != it["brand"])
            or (it["model"] and t["model"] != it["model"])
            or (it["size"] and t["size"] != it["size"])
        ):
            return 0
        soft = [k for k in ("colour", "material", "style") if it[k]]
        hit = sum(t[k] == it[k] for k in soft)
        return 1 + round(2 * hit / len(soft)) if soft else 3

    def utility(self, u: User, it: dict, item: Item) -> float:
        rel = self.relevance(it, item)
        if rel == 0:
            return 0.0
        price_fit = 1.0 / (1.0 + max(0.0, item.price / u.budget - 1.0) ** 2 * 3)
        cond = 1.0 if item.true["condition"] < u.min_condition + 1 else 0.6
        brand = 0.7 + 0.6 * u.brands.get(item.true["brand"], 0) * len(u.brands) / 4
        return rel / 3 * item.appeal * price_fit * cond * min(brand, 1.6)

    # ---- serving a results page --------------------------------------------------------------------------
    def serve(self, s: Search, ranked: list[int], policy: str, randomized: bool = False) -> dict:
        """The buyer looks at the page top-down, clicks, maybe buys. Sold items leave the market."""
        c, u = self.cfg["behaviour"], self.users[s.user_id]
        shown = [i for i in ranked if self.items[i].available(s.day)][: c["page_size"]]
        pos = np.arange(len(shown))
        exam = 1.0 / (1.0 + pos) ** c["exam_decay"]
        util = np.array([self.utility(u, s.intent, self.items[i]) for i in shown])
        p_click = exam * (1 - np.exp(-c["attract"] * util))
        ids = np.array(shown, dtype=np.int64)
        clicked = _mix(np.full(len(shown), s.search_id), ids, 1) < p_click
        p_buy = c["buy_base"] * (1 - np.exp(-c["buy_sens"] * util))
        bought = clicked & (_mix(np.full(len(shown), s.search_id), ids, 2) < p_buy)
        bought_ids = [int(i) for i in ids[bought]][:1]  # one purchase per search at most
        for i in bought_ids:
            self.items[i].sold_day = s.day
        self.log["searches"].append(
            {
                "search_id": s.search_id,
                "day": s.day,
                "user_id": s.user_id,
                "market": s.market,
                "query": s.query,
                "policy": policy,
                "randomized": randomized,
                "results": len(shown),
            }
        )
        self.log["impressions"] += [
            {"search_id": s.search_id, "pos": int(p), "item_id": int(i)} for p, i in zip(pos, shown, strict=True)
        ]
        self.log["clicks"] += [{"search_id": s.search_id, "item_id": int(i)} for i in ids[clicked]]
        self.log["purchases"] += [
            {"search_id": s.search_id, "item_id": i, "price": round(self.items[i].price, 2)} for i in bought_ids
        ]
        return {
            "shown": shown,
            "clicks": [int(i) for i in ids[clicked]],
            "purchases": bought_ids,
            "gmv": sum(self.items[i].price for i in bought_ids),
        }

    def end_of_day(self, day: int) -> None:
        """Items sold elsewhere or expired leave; new listings arrive for tomorrow. Both are keyed by item and day,
        so forks that served different designs stay in step."""
        c = self.cfg["catalog"]
        ids = np.arange(len(self.items))
        elsewhere = _mix(ids, day, 3) < c["sold_elsewhere"]
        for it in self.items:
            if it.available(day) and (elsewhere[it.item_id] or day - it.listed_day > c["max_age_days"]):
                it.gone_day = day + 1
        self.rng = np.random.default_rng([self.seed, 17, day])
        for _ in range(c["new_per_day"]):
            self._list_item(day + 1)
        self.day = day + 1

    def available_mask(self, day: int) -> np.ndarray:
        return np.array([it.available(day) for it in self.items], dtype=bool)
