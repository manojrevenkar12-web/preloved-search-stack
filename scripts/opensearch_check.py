"""Check the production index and query against a real OpenSearch (CI starts one in a container).

    OPENSEARCH_URL=http://localhost:9200 python3 scripts/opensearch_check.py

Creates the index from `opensearch_mapping`, indexes a few listings, and runs `opensearch_query`: a French title must
match an English query through its slots, the hard filters must hold, and sold listings must never come back.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preloved.search.index import opensearch_mapping, opensearch_query  # noqa: E402
from preloved.search.qu import QueryUnderstanding  # noqa: E402

URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
INDEX = "listings-check"


def call(method: str, path: str, body: dict | str | None = None) -> dict:
    data = body if isinstance(body, str) else json.dumps(body) if body is not None else None
    req = urllib.request.Request(
        f"{URL}/{path}",
        data=data.encode() if data else None,
        method=method,
        headers={"Content-Type": "application/x-ndjson" if isinstance(body, str) else "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or b"{}")


def main() -> None:
    docs = [
        {"title": "Sac Timeless cuir noir", "brand": "Chanel", "category": "bag", "colour": "black", "available": True},
        {"title": "Chanel bag black", "brand": "Chanel", "category": "bag", "colour": "black", "available": False},
        {"title": "Black bag", "brand": "Prada", "category": "bag", "colour": "black", "available": True},
        {
            "title": "Chanel ballerines noires",
            "brand": "Chanel",
            "category": "shoes",
            "colour": "black",
            "available": True,
        },
    ]
    try:
        call("DELETE", INDEX)
    except Exception:  # noqa: BLE001 - absent on a fresh cluster
        pass
    call("PUT", INDEX, opensearch_mapping())
    bulk = "".join(json.dumps({"index": {"_id": str(i)}}) + "\n" + json.dumps(d) + "\n" for i, d in enumerate(docs))
    call("POST", f"{INDEX}/_bulk?refresh=true", bulk)
    q = "chanel black bag"
    hits = call("POST", f"{INDEX}/_search", opensearch_query(q, QueryUnderstanding().parse(q), k=10))["hits"]["hits"]
    ids = [h["_id"] for h in hits]
    assert ids == ["0"], f"expected only the French Chanel bag that is for sale, got {ids}"
    print(f"opensearch: {q!r} -> {[docs[int(i)]['title'] for i in ids]}")


if __name__ == "__main__":
    main()
