from __future__ import annotations

import torch


def entropy(dist: torch.Tensor) -> torch.Tensor:
    d = torch.clamp(dist, min=1e-9)
    return -(d * d.log()).sum()


def top_margin(dist: torch.Tensor) -> torch.Tensor:
    vals, _ = torch.topk(dist, k=min(2, dist.numel()))
    if vals.numel() < 2:
        return torch.tensor(0.0, dtype=dist.dtype)
    return vals[0] - vals[1]


def extract_probe_features(dist: torch.Tensor, sampled_dists: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Return [entropy, top1-top2 margin, optional variance proxy]."""
    ent = entropy(dist)
    margin = top_margin(dist)
    variance = torch.tensor(0.0, dtype=dist.dtype)
    if sampled_dists:
        stacked = torch.stack(sampled_dists)
        variance = stacked.var(dim=0).mean()
    return torch.stack([ent, margin, variance]).float()
