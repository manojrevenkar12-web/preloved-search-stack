"""Lexical retrieval: BM25 over listing text, with the availability of each one-of-a-kind item as a filter.

In production this is an OpenSearch index (`opensearch_mapping`, `opensearch_query`); here the same analyzer and BM25
run in-process on a sparse matrix so thousands of searches can be simulated. A listing's text is its title plus the
attributes the seller filled in, written as canonical words (`brand:chanel colour:black`), so a slot read from the
query matches whatever language the seller wrote in.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from preloved.search.qu import Parsed, words


def listing_tokens(listing: dict, structured: bool) -> list[str]:
    toks = words(listing["title"])
    if structured:
        toks += [
            f"{k}:{words(str(listing[k]))[0] if words(str(listing[k])) else listing[k]}".replace(" ", "_")
            for k in ("brand", "model", "category", "material", "colour")
            if listing.get(k)
        ]
    return toks


def query_tokens(text: str, parsed: Parsed | None) -> list[str]:
    toks = words(text)
    if parsed is not None:
        toks += [
            f"{k}:{words(v)[0] if words(v) else v}".replace(" ", "_")
            for k, v in parsed.slots().items()
            if k in ("brand", "model", "category", "material", "colour")
        ]
    return toks


class BM25Index:
    def __init__(self, listings: list[dict], structured: bool = True, k1: float = 1.2, b: float = 0.75):
        self.structured = structured
        docs = [listing_tokens(x, structured) for x in listings]
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for d, toks in enumerate(docs):
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.setdefault(t, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            for t, c in counts.items():
                rows.append(d)
                cols.append(t)
                vals.append(c)
        n = len(docs)
        tf = sparse.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n, len(self.vocab)))
        dl = np.array([len(t) for t in docs], dtype=np.float32)
        df = np.bincount(np.array(cols), minlength=len(self.vocab))
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * dl / max(dl.mean(), 1e-6))
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.matrix = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.size = n

    def scores(self, toks: list[str]) -> np.ndarray:
        cols = [self.vocab[t] for t in toks if t in self.vocab]
        if not cols:
            return np.zeros(self.size, dtype=np.float32)
        return np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()


def opensearch_mapping() -> dict:
    """The production index: one document per listing, availability as a filter field."""
    keyword = {"type": "keyword"}
    return {
        "settings": {
            "index": {"knn": True},
            "analysis": {"analyzer": {"listing": {"tokenizer": "standard", "filter": ["lowercase", "asciifolding"]}}},
        },
        "mappings": {
            "properties": {
                "title": {"type": "text", "analyzer": "listing"},
                "brand": keyword,
                "model": keyword,
                "category": keyword,
                "material": keyword,
                "colour": keyword,
                "size": keyword,
                "condition": keyword,
                "market": keyword,
                "price": {"type": "float"},
                "listed_day": {"type": "integer"},
                "available": {"type": "boolean"},
                "embedding": {"type": "knn_vector", "dimension": 32},
            }
        },
    }


def opensearch_query(text: str, parsed: Parsed, k: int, hard: tuple[str, ...] = ("brand", "category")) -> dict:
    """The same retrieval as BM25Index with slot filters: hard slots filter, soft slots boost, sold items never."""
    slots = parsed.slots()
    filters = [{"term": {"available": True}}] + [{"term": {s: slots[s]}} for s in hard if s in slots]
    should = [{"match": {"title": text}}] + [
        {"term": {s: {"value": slots[s], "boost": 2.0}}} for s in ("model", "colour", "material") if s in slots
    ]
    # the title is one way to match, not a must: a French title still matches an English query through its slots
    return {"size": k, "query": {"bool": {"filter": filters, "should": should, "minimum_should_match": 1}}}
