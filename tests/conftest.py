import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def small_cfg():
    cfg = tomllib.loads((ROOT / "config" / "world.toml").read_text())
    cfg["catalog"] |= {"initial": 900, "new_per_day": 60}
    cfg["users"] |= {"count": 300}
    return cfg


@pytest.fixture
def small_world(small_cfg):
    from preloved.world.sim import World

    return World(small_cfg)


@pytest.fixture
def settings():
    from preloved.lab.common import settings as s

    return s()
