"""Train the two towers in PyTorch on logged (query, clicked item) pairs; export float16 tables for NumPy serving.

In-batch sampled softmax with a temperature: every other item in the batch is a negative. Purchases count three
times. Deterministic on CPU (seeded, single thread).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from preloved.search.vectors import BUCKETS, hashed, item_features, query_features


class Tower(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.emb = nn.EmbeddingBag(BUCKETS, dim, mode="mean")
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.emb(flat, offsets), dim=1)


def _bag(rows: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.tensor(np.concatenate([[0], np.cumsum([len(r) for r in rows])[:-1]]), dtype=torch.long)
    return torch.from_numpy(np.concatenate(rows)), offsets


def train(pairs: list[tuple[str, object, dict, float]], out: Path, cfg: dict, ids: bool = False) -> dict:
    """pairs: (query text, parsed query, clicked listing, weight)."""
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    q_rows = [hashed(query_features(t, p)) for t, p, _, _ in pairs]
    i_rows = [hashed(item_features(x, ids)) for _, _, x, _ in pairs]
    w = torch.tensor([wt for *_, wt in pairs], dtype=torch.float32)
    qt, it = Tower(cfg["dim"]), Tower(cfg["dim"])
    opt = torch.optim.Adam([*qt.parameters(), *it.parameters()], lr=cfg["learning_rate"])
    g = torch.Generator().manual_seed(cfg["seed"])
    t0, losses = time.perf_counter(), []
    for _ in range(cfg["epochs"]):
        order = torch.randperm(len(pairs), generator=g).tolist()
        for s in range(0, len(order) - 1, cfg["batch_size"]):
            b = order[s : s + cfg["batch_size"]]
            qv = qt(*_bag([q_rows[k] for k in b]))
            iv = it(*_bag([i_rows[k] for k in b]))
            logits = qv @ iv.T / cfg["temperature"]
            loss = (nn.functional.cross_entropy(logits, torch.arange(len(b)), reduction="none") * w[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "towers.npz",
        query=qt.emb.weight.detach().numpy().astype(np.float16),
        item=it.emb.weight.detach().numpy().astype(np.float16),
    )
    card = {
        "ids": ids,
        "pairs": len(pairs),
        "params": {k: cfg[k] for k in sorted(cfg)},
        "final_loss": round(float(np.mean(losses[-50:])), 4),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    return card
