"""The held-out world, run once: history, training and the full evaluation on buyers no design choice has seen.

    python3 scripts/heldout.py            -> reports/heldout.json, reports/heldout.lock

Refuses to run if the lock exists. The lock records when it ran and on which commit, so a second look is visible.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "reports" / "heldout.lock"


def main() -> None:
    if LOCK.exists():
        sys.exit(f"the held-out world already ran ({LOCK.read_text().strip()}); see docs/heldout.md")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    LOCK.parent.mkdir(exist_ok=True)
    LOCK.write_text(json.dumps({"ran": datetime.now(UTC).isoformat(timespec="seconds"), "commit": commit}) + "\n")
    for step in ("history", "train", "evaluate"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / f"{step}.py"), "--world", "heldout"], check=True)


if __name__ == "__main__":
    main()
