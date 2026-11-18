import numpy as np

from preloved.search.index import BM25Index, listing_tokens, opensearch_mapping, opensearch_query, query_tokens
from preloved.search.qu import QueryUnderstanding


def L(i, title, **kw):
    return {
        "item_id": i,
        "title": title,
        "brand": None,
        "model": None,
        "category": None,
        "material": None,
        "colour": None,
        **kw,
    }


def test_structured_tokens_match_across_languages():
    items = [
        L(0, "Sac Timeless en cuir", brand="Chanel", category="bag", colour="black"),
        L(1, "Borsa in pelle", brand="Prada", category="bag", colour="black"),
    ]
    qu = QueryUnderstanding()
    q = "chanel black bag"
    plain = BM25Index(items, structured=False).scores(query_tokens(q, None))
    canon = BM25Index(items, structured=True).scores(query_tokens(q, qu.parse(q)))
    assert plain[0] == 0  # the French title shares no word with the English query
    assert canon[0] > canon[1] > 0
    assert "brand:chanel" in listing_tokens(items[0], structured=True)


def test_bm25_prefers_rarer_terms():
    items = [L(i, t) for i, t in enumerate(["red bag", "red bag", "red shoes", "vintage bag"])]
    s = BM25Index(items, structured=False).scores(["vintage", "bag"])
    assert int(np.argmax(s)) == 3


def test_opensearch_dsl_has_the_slots_as_filters():
    qu = QueryUnderstanding()
    q = opensearch_query("chanel black bag", qu.parse("chanel black bag"), k=20)
    assert "query" in q and q["size"] == 20
    assert "properties" in opensearch_mapping()["mappings"]
    flt = q["query"]["bool"]["filter"]
    assert {"term": {"available": True}} in flt and {"term": {"brand": "Chanel"}} in flt
    assert q["query"]["bool"]["minimum_should_match"] == 1
