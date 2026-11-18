"""Vector retrieval at serving time: NumPy only. Training is in `preloved.search.vectors_train` (PyTorch).

Both towers are bags of hashed features averaged into one vector. The query tower reads the words, their character
trigrams and the slots query understanding found; the item tower reads the title the same way plus the attributes the
seller filled in, price band and condition. Nothing in the item tower is the item's identity: every listing is one of
a kind and new, so an embedding per item would never be trained before the item sells (the `ids` ablation shows it).
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np

from preloved.search.qu import Parsed, words

BUCKETS = 1 << 15


def _grams(toks: list[str]) -> list[str]:
    out = [f"w:{t}" for t in toks]
    for t in toks:
        p = f"<{t}>"
        out += [f"c:{p[i : i + 3]}" for i in range(len(p) - 2)]
    return out


def query_features(text: str, parsed: Parsed | None) -> list[str]:
    feats = _grams(words(text))
    if parsed is not None:
        feats += [f"s:{k}={v}" for k, v in parsed.slots().items()]
    return feats or ["w:<empty>"]


def item_features(listing: dict, ids: bool = False) -> list[str]:
    if ids:
        return [f"id:{listing['item_id']}"]
    feats = _grams(words(listing["title"]))
    feats += [
        f"a:{k}={listing[k]}"
        for k in ("brand", "model", "category", "material", "colour", "size", "condition")
        if listing.get(k)
    ]
    feats.append(f"p:{int(math.log2(max(listing['price'], 1)))}")
    return feats


def hashed(feats: list[str]) -> np.ndarray:
    return np.array([zlib.crc32(f.encode()) % BUCKETS for f in feats], dtype=np.int64)


class TwoTower:
    def __init__(self, q: np.ndarray, i: np.ndarray, card: dict):
        self.q, self.i, self.card = q.astype(np.float32), i.astype(np.float32), card
        self.ids = bool(card.get("ids", False))

    @classmethod
    def load(cls, directory: Path) -> TwoTower:
        with np.load(directory / "towers.npz") as z:
            return cls(z["query"], z["item"], json.loads((directory / "card.json").read_text()))

    @staticmethod
    def _encode(table: np.ndarray, idx: np.ndarray) -> np.ndarray:
        v = table[idx].mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-8)

    def encode_query(self, text: str, parsed: Parsed | None) -> np.ndarray:
        return self._encode(self.q, hashed(query_features(text, parsed)))

    def encode_items(self, listings: list[dict]) -> np.ndarray:
        return np.stack([self._encode(self.i, hashed(item_features(x, self.ids))) for x in listings])
