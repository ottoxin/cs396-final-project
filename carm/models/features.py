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


def extract_cross_modal_features(
    v_dist: torch.Tensor,
    t_dist: torch.Tensor,
    v_answer_text: str,
    t_answer_text: str,
) -> torch.Tensor:
    """Return 5-dim cross-modal agreement features.

    Features encode the agreement/disagreement signal between vision-only and
    text-only probe answers — the most informative signal for distinguishing
    C1 (consistent) from C4 (contradictory).

    Returns:
        Tensor of shape [5]:
          [0] agreement_binary: 1.0 if answers match (case-insensitive), else 0.0
          [1] dist_overlap: sum of element-wise min(v_dist, t_dist) — soft agreement
          [2] v_max_prob: max confidence in vision distribution
          [3] t_max_prob: max confidence in text distribution
          [4] both_confident_disagree: v_max * t_max * (1 - agreement) — C4 signal
    """
    v_norm = str(v_answer_text or "").strip().lower()
    t_norm = str(t_answer_text or "").strip().lower()
    agreement_binary = torch.tensor(1.0 if (v_norm == t_norm and v_norm != "") else 0.0)

    # Soft distribution overlap — measures how much the two distributions
    # assign probability mass to the same tokens.
    if v_dist.numel() > 0 and t_dist.numel() > 0 and v_dist.shape == t_dist.shape:
        v_soft = torch.softmax(v_dist.float(), dim=-1) if v_dist.dtype != torch.float32 or v_dist.sum().abs().item() > 1e-3 else v_dist.float()
        t_soft = torch.softmax(t_dist.float(), dim=-1) if t_dist.dtype != torch.float32 or t_dist.sum().abs().item() > 1e-3 else t_dist.float()
        dist_overlap = torch.min(v_soft, t_soft).sum()
        v_max_prob = v_soft.max()
        t_max_prob = t_soft.max()
    else:
        dist_overlap = torch.tensor(0.0)
        v_max_prob = torch.tensor(0.0)
        t_max_prob = torch.tensor(0.0)

    both_confident_disagree = v_max_prob * t_max_prob * (1.0 - agreement_binary)

    return torch.stack([
        agreement_binary,
        dist_overlap,
        v_max_prob,
        t_max_prob,
        both_confident_disagree,
    ]).float()


def extract_cross_modal_features_augmented(
    v_dist: torch.Tensor,
    t_dist: torch.Tensor,
    v_answer_text: str,
    t_answer_text: str,
) -> torch.Tensor:
    """Return 6-dim cross-modal agreement features (augmented for C3/C5 discrimination).

    Extends the 5-dim baseline with a 6th feature:
      [5] t_conf_advantage: max(p_t) - max(p_v)

    This feature is the targeted remedy for C3/C5 confusability: C3 has high
    text confidence and low vision confidence, so t_conf_advantage >> 0;
    C5 has low confidence in both, so t_conf_advantage ≈ 0. The sign and
    magnitude of this feature align with the C3/C5 decision boundary.
    """
    base = extract_cross_modal_features(v_dist, t_dist, v_answer_text, t_answer_text)
    # base[3] = t_max_prob, base[2] = v_max_prob
    t_conf_advantage = base[3] - base[2]
    return torch.cat([base, t_conf_advantage.unsqueeze(0)]).float()
