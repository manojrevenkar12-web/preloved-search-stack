"""The rules in AGENTS.md, checked."""

import ast
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src" / "preloved"


def imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_the_truth_stays_in_the_world():
    """Serving and measuring code never imports the simulation or the lab that reads it."""
    for pkg in ("search", "measure", "service"):
        for f in (SRC / pkg).rglob("*.py"):
            bad = {m for m in imports(f) if m.startswith(("preloved.world", "preloved.lab"))}
            assert not bad, f"{f.name} imports {bad}"


def test_the_world_does_not_read_the_stack_settings():
    for f in (SRC / "world").rglob("*.py"):
        text = f.read_text()
        assert "stack.toml" not in text and "designs.toml" not in text
        assert not {m for m in imports(f) if m.startswith(("preloved.search", "preloved.measure", "preloved.lab"))}


def test_same_luck_for_every_design(small_world):
    """Two forks get the same searches, and the same page gets the same clicks and purchases."""
    a, b = small_world.fork(), small_world.fork()
    sa, sb = a.searches_for_day(0), b.searches_for_day(0)
    assert [s.query for s in sa] == [s.query for s in sb] and len(sa) > 10
    ranked = list(range(30))
    for x, y in zip(sa[:20], sb[:20], strict=True):
        assert a.serve(x, ranked, "a") == b.serve(y, ranked, "b")


def test_end_of_day_is_keyed_so_forks_stay_in_step(small_world):
    a, b = small_world.fork(), small_world.fork()
    s = a.searches_for_day(0)[0]
    a.serve(s, list(range(40)), "a")  # a different page in one fork only
    b.searches_for_day(0)
    a.end_of_day(0)
    b.end_of_day(0)
    assert [it.title for it in a.items] == [it.title for it in b.items]
    gone_a = {it.item_id for it in a.items if it.gone_day}
    gone_b = {it.item_id for it in b.items if it.gone_day}
    assert gone_a == gone_b


def test_listings_never_carry_the_truth(small_world):
    x = small_world.items[0].listing()
    assert "appeal" not in x and "true" not in x
    assert np.isfinite(x["price"])
