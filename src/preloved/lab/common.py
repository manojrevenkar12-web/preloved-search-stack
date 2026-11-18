"""Paths, settings and the saved state shared by the scripts."""

from __future__ import annotations

import pickle
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORLDS = {"dev": "world.toml", "heldout": "heldout.toml"}


def settings() -> dict:
    return tomllib.loads((ROOT / "config" / "stack.toml").read_text())


def designs() -> tuple[dict, dict]:
    d = tomllib.loads((ROOT / "config" / "designs.toml").read_text())
    return d["designs"], d.get("variants", {})


def data_dir(world: str) -> Path:
    return ROOT / "data" / world


def models_dir(world: str) -> Path:
    return ROOT / "models" / world


def save_world(world, name: str) -> None:
    world.log = {k: [] for k in world.log}  # the log is in the warehouse (parquet), not in the state
    with open(data_dir(name) / "world.pkl", "wb") as f:
        pickle.dump(world, f)


def load_world(name: str):
    with open(data_dir(name) / "world.pkl", "rb") as f:
        return pickle.load(f)
