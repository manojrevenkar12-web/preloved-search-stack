"""Three weeks of the marketplace under today's search, logged as the warehouse would hold it.

    python3 scripts/history.py [--world dev|heldout]   -> data/<world>/logs/*.parquet, data/<world>/world.pkl

5% of searches have their top 10 shuffled, so position bias can be measured instead of assumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preloved.lab.common import WORLDS, data_dir, designs, save_world, settings  # noqa: E402
from preloved.lab.run import run_days, write_log  # noqa: E402
from preloved.search.stack import Design, Stack  # noqa: E402
from preloved.world.sim import World  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=list(WORLDS), default="dev")
    args = ap.parse_args()
    world = World.load(ROOT / "config" / WORLDS[args.world])
    s = settings()
    today = Design.from_config("lexical", designs()[0]["lexical"])
    t0 = time.perf_counter()
    out = run_days(
        world,
        Stack(s),
        range(world.cfg["timeline"]["history_days"]),
        lambda _: ("lexical", today),
        randomized_share=world.cfg["timeline"]["randomized_share"],
    )
    data_dir(args.world).mkdir(parents=True, exist_ok=True)
    write_log(world, data_dir(args.world) / "logs")
    save_world(world, args.world)
    print(
        f"{args.world}: {len(out)} searches, {sum(o['purchases'] for o in out)} purchases, "
        f"{len(world.items)} listings in {time.perf_counter() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
