"""Train the re-ranker in PyTorch from logged results pages, unbiased for position.

Each logged search is one list: the items shown, their features, and what the buyer clicked and bought. The loss is a
listwise softmax cross-entropy whose targets are clicks weighted by the inverse of how likely their position was to be
looked at (estimated from randomised traffic), with purchases counted three times. `ips=False` trains on raw clicks:
the ablation that learns the old ranking's position bias back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from preloved.search.ranker import FEATURES


def position_bias(randomized: list[tuple[int, int]], positions: int) -> np.ndarray:
    """Examination by position from searches whose top 10 was shuffled: click rate per position, relative to the
    first, smoothed with a power law that also covers positions beyond the shuffled ones."""
    pos = np.array([p for p, _ in randomized])
    clk = np.array([c for _, c in randomized], dtype=float)
    k = np.arange(10)
    rate = np.array([clk[pos == i].mean() if (pos == i).any() else np.nan for i in k])
    ok = ~np.isnan(rate) & (rate > 0)
    slope = np.polyfit(np.log1p(k[ok]), np.log(rate[ok] / rate[0]), 1)[0]
    return np.clip((1.0 + np.arange(positions)) ** slope, 1e-3, 1.0)


def train(
    lists: list[dict], out: Path, cfg: dict, exam: np.ndarray, ips: bool = True, without: list[str] | None = None
) -> dict:
    """lists: [{"x": features (n, F), "pos": positions (n,), "click": (n,), "buy": (n,)}] with at least one click."""
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    allx = np.concatenate([d["x"] for d in lists])
    mean, std = allx.mean(axis=0), allx.std(axis=0) + 1e-6
    drop = [FEATURES.index(f) for f in (without or [])]
    data = []
    for d in lists:
        x = (d["x"] - mean) / std
        if drop:
            x[:, drop] = 0.0
        t = d["click"] / (exam[d["pos"]] if ips else 1.0) + 3.0 * d["buy"]
        data.append((torch.from_numpy(x.astype(np.float32)), torch.from_numpy((t / t.sum()).astype(np.float32))))
    model = nn.Sequential(nn.Linear(len(FEATURES), cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], 1))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(cfg["seed"])
    t0 = time.perf_counter()
    for _ in range(cfg["epochs"]):
        for s in torch.randperm(len(data), generator=g).tolist():
            x, t = data[s]
            loss = -(t * torch.log_softmax(model(x).squeeze(1), dim=0)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    out.mkdir(parents=True, exist_ok=True)
    l1, l2 = model[0], model[2]
    np.savez_compressed(
        out / "ranker.npz",
        w1=l1.weight.detach().numpy(),
        b1=l1.bias.detach().numpy(),
        w2=l2.weight.detach().numpy(),
        b2=l2.bias.detach().numpy(),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    card = {
        "ips": ips,
        "without": without or [],
        "lists": len(lists),
        "features": FEATURES,
        "position_bias": [round(float(v), 4) for v in exam[:10]],
        "params": {k: cfg[k] for k in sorted(cfg)},
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    return card
