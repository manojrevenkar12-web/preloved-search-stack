"""Query understanding: a buyer's words, in any of five languages and with typos, read into catalogue slots.

The vocabulary is the catalogue's (scout of `preloved.taxonomy`): brands and their aliases, iconic models, categories,
materials, colours, styles and sizes. Longest phrases win; a word that is not in the vocabulary is matched to the
closest vocabulary word within one or two edits. A model whose name is also an ordinary word ("boy", "city", "love",
"tank", "kate") counts only when its brand is in the query. What the vocabulary does not know ("bv", "loubs") is
left for the vector retrieval, which learns it from clicks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from preloved.taxonomy import (
    BAG_SIZES,
    CATEGORIES,
    CLOTHING_SIZES,
    COLOURS,
    MATERIALS,
    SHOE_SIZES,
    STYLES,
    brand_forms,
    category_of_model,
    model_forms,
    surface_forms,
)

AMBIGUOUS_MODELS = {
    "boy",
    "city",
    "love",
    "tank",
    "kate",
    "pouch",
    "luggage",
    "trench",
    "teddy",
    "alma",
    "jackie",
    "kelly",
    "cleo",
    "puzzle",
    "wander",
    "cassette",
    "saddle",
    "speedy",
    "constance",
    "19",
}
_WORD = re.compile(r"[a-z0-9&\-.]+")


def fold(text: str) -> str:
    """Lower case without accents: "Hermès matelassé" and "hermes matelasse" are the same words."""
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def words(text: str) -> list[str]:
    return _WORD.findall(fold(text))


@dataclass
class Parsed:
    text: str
    brand: str | None = None
    model: str | None = None
    category: str | None = None
    colour: str | None = None
    material: str | None = None
    style: str | None = None
    size: str | None = None
    rest: list[str] = field(default_factory=list)  # words the vocabulary did not know
    fuzzy: list[tuple[str, str]] = field(default_factory=list)  # (typed, read as)

    def slots(self) -> dict[str, str]:
        return {
            k: v
            for k in ("brand", "model", "category", "colour", "material", "style", "size")
            if (v := getattr(self, k))
        }


def _table(kind: str, mapping: dict[str, object]) -> dict[tuple[str, ...], tuple[str, object]]:
    return {tuple(words(w)): (kind, v) for w, v in mapping.items() if words(w)}


class QueryUnderstanding:
    def __init__(self, max_edits_short: int = 1, max_edits_long: int = 2, fuzzy: bool = True):
        self.phrases: dict[tuple[str, ...], tuple[str, object]] = {}
        for kind, mapping in [
            ("category", surface_forms(CATEGORIES)),
            ("material", surface_forms(MATERIALS)),
            ("colour", surface_forms(COLOURS)),
            ("style", surface_forms(STYLES)),
            ("model", model_forms()),
            ("brand", brand_forms()),
        ]:
            self.phrases.update(_table(kind, mapping))
        self.longest = max(len(k) for k in self.phrases)
        self.single = sorted({k[0] for k in self.phrases if len(k) == 1 and len(k[0]) >= 4})
        self.fuzzy, self.e1, self.e2 = fuzzy, max_edits_short, max_edits_long

    def parse(self, text: str) -> Parsed:
        toks = words(text)
        p = Parsed(text)
        found: list[tuple[str, object, tuple[str, ...]]] = []  # (kind, value, the words typed)
        i = 0
        while i < len(toks):
            for n in range(min(self.longest, len(toks) - i), 0, -1):
                span = tuple(toks[i : i + n])
                hit = self.phrases.get(span)
                if hit:
                    found.append((*hit, span))
                    i += n
                    break
            else:
                tok = toks[i]
                fixed = self._closest(tok) if self.fuzzy and len(tok) >= 5 and not tok.isdigit() else None
                if fixed:
                    found.append((*self.phrases[(fixed,)], (fixed,)))
                    p.fuzzy.append((tok, fixed))
                else:
                    p.rest.append(tok)
                i += 1
        for kind, value, _ in found:
            if kind == "brand" and not p.brand:
                p.brand = str(value)
            elif kind in ("category", "colour", "material", "style") and not getattr(p, kind):
                setattr(p, kind, str(value))
        for kind, value, span in found:  # models last: a model word that is also an ordinary word needs its brand
            if kind != "model" or p.model:
                continue
            brand, model = value  # type: ignore[misc]
            if p.brand == brand or (not p.brand and " ".join(span) not in AMBIGUOUS_MODELS):
                p.brand, p.model = brand, model
            else:  # not read as a model: its words may still say something ("boy bag" is a bag)
                for tok in span:
                    hit = self.phrases.get((tok,))
                    if hit and hit[0] in ("category", "colour", "material", "style") and not getattr(p, hit[0]):
                        setattr(p, hit[0], str(hit[1]))
        if p.model and not p.category:
            p.category = category_of_model(p.model)
        for tok in toks:  # sizes: numbers and letters that only mean a size in context
            up = tok.upper()
            if (p.category == "shoes" and tok in SHOE_SIZES) or (
                p.category in ("jacket", "dress") and up in CLOTHING_SIZES
            ):
                p.size = up
            elif p.model in ("Birkin", "Kelly") and tok in BAG_SIZES:
                p.size = tok
        p.rest = [t for t in p.rest if t != (p.size or "").lower()]
        return p

    def _closest(self, tok: str) -> str | None:
        limit = self.e2 if len(tok) >= 8 else self.e1
        best, best_d = None, limit + 1
        for cand in self.single:
            if abs(len(cand) - len(tok)) > limit or cand[0] != tok[0]:
                continue
            d = osa(tok, cand, limit)
            if d < best_d:
                best, best_d = cand, d
        return best


def osa(a: str, b: str, limit: int) -> int:
    """Optimal string alignment distance (insertions, deletions, substitutions, adjacent swaps), cut off at limit+1."""
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]
