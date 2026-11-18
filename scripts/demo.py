"""Record the console's demo: a few queries through every design, and the dev report.

    python3 scripts/demo.py   -> console/public/demo.json

Uses the service's own code path (FastAPI test client), so the demo shows exactly what the API returns.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from preloved.search.stack import Design  # noqa: E402
from preloved.service.app import create_app, load_stack  # noqa: E402

QUERIES = [
    "sac chanel noir matelassé",
    "birkn togo",
    "loubs 38",
    "bv bag",
    "borsa gucci vintage",
    "slp shoes",
    "max mara teddy M",
    "cartier love bracelet gold",
]


def main() -> None:
    cfg = tomllib.loads((ROOT / "config" / "designs.toml").read_text())
    designs = {n: Design.from_config(n, c) for n, c in {**cfg["designs"], **cfg.get("variants", {})}.items()}
    stack, available, today = load_stack(ROOT, "dev")
    report = json.loads((ROOT / "reports" / "dev.json").read_text())
    c = TestClient(create_app(stack, designs, available, today, report))
    recorded = {
        q: {d: c.get("/search", params={"q": q, "design": d, "user": -1}).json() for d in designs} for q in QUERIES
    }
    out = {"designs": c.get("/designs").json(), "recorded": recorded, "report": report}
    (ROOT / "console" / "public" / "demo.json").write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"demo: {len(QUERIES)} queries x {len(designs)} designs")


if __name__ == "__main__":
    main()
