from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from carm.data.schema import Action
from carm.experimental.labels import ACTION_LABELS, INFO_STATE_LABELS, PAIRWISE_RELATION_LABELS


@dataclass
class ExperimentalCARMConfig:
    hidden_size: int = 128
    probe_feature_size: int = 3
    pool: str = "mean"
    trunk_hidden_size: int | None = None


@dataclass
class CascadeCARMConfig:
    """Configuration for the hierarchical cascade CARM model.

    The cascade architecture explicitly models the semantic dependency:
      informativeness → pairwise relation → action

    Each stage is conditioned on all previous stage outputs, which allows
    the action head to directly observe predicted informativeness and relation
    signals rather than re-deriving them from raw features.
    """
    hidden_size: int = 128
    probe_feature_size: int = 3
    cross_modal_feature_size: int = 5
    pool: str = "mean"
    trunk_hidden_size: int | None = None
    stage_hidden_size: int = 64


class ExperimentalCARMHeads(nn.Module):
    def __init__(self, config: ExperimentalCARMConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExperimentalCARMConfig()
        decision_dim = self.config.hidden_size + (2 * self.config.probe_feature_size)
        trunk_hidden_size = int(self.config.trunk_hidden_size or self.config.hidden_size)
        self.trunk = nn.Sequential(
            nn.Linear(decision_dim, trunk_hidden_size),
            nn.GELU(),
        )
        self.vision_info_head = nn.Linear(trunk_hidden_size, len(INFO_STATE_LABELS))
        self.text_info_head = nn.Linear(trunk_hidden_size, len(INFO_STATE_LABELS))
        self.relation_head = nn.Linear(trunk_hidden_size, len(PAIRWISE_RELATION_LABELS))
        self.action_head = nn.Linear(trunk_hidden_size, len(ACTION_LABELS))

    def pool_anchor_states(self, anchor_states: torch.Tensor) -> torch.Tensor:
        if anchor_states.dim() == 3:
            if self.config.pool == "mean":
                return anchor_states.mean(dim=1)
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")
        if anchor_states.dim() == 2:
            if self.config.pool == "mean":
                return anchor_states.mean(dim=0)
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")
        raise ValueError("anchor_states must have 2 or 3 dims")

    def forward(
        self,
        anchor_states: torch.Tensor,
        phi_v: torch.Tensor,
        phi_t: torch.Tensor,
        phi_cross: torch.Tensor | None = None,  # accepted but ignored (flat architecture)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pool_anchor_states(anchor_states)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        if phi_v.dim() == 1:
            phi_v = phi_v.unsqueeze(0)
        if phi_t.dim() == 1:
            phi_t = phi_t.unsqueeze(0)

        features = torch.cat([pooled, phi_v, phi_t], dim=-1)
        trunk_features = self.trunk(features)
        vision_info_logits = self.vision_info_head(trunk_features)
        text_info_logits = self.text_info_head(trunk_features)
        relation_logits = self.relation_head(trunk_features)
        action_logits = self.action_head(trunk_features)
        return vision_info_logits, text_info_logits, relation_logits, action_logits


class CascadeCARMHeads(nn.Module):
    """Hierarchical cascade CARM with three prediction stages.

    Stage 1: Modality informativeness (vision_info, text_info)
    Stage 2: Pairwise relation, conditioned on stage-1 softmax outputs
    Stage 3: Joint action, conditioned on stage-1 and stage-2 softmax outputs

    This matches the natural semantic dependency:
      Is each modality informative?  →  How do they relate?  →  What action to take?

    Cross-modal agreement features (phi_cross, 5-dim) provide the model with
    explicit agreement/disagreement signals that are the most discriminative
    inputs for distinguishing C1 (consistent) from C4 (contradictory).
    """

    def __init__(self, config: CascadeCARMConfig | None = None) -> None:
        super().__init__()
        self.config = config or CascadeCARMConfig()
        n_info = len(INFO_STATE_LABELS)       # 2
        n_rel = len(PAIRWISE_RELATION_LABELS)  # 4
        n_act = len(ACTION_LABELS)             # 4

        decision_dim = (
            self.config.hidden_size
            + 2 * self.config.probe_feature_size
            + self.config.cross_modal_feature_size
        )
        trunk_hidden = int(self.config.trunk_hidden_size or self.config.hidden_size)
        stage_hidden = int(self.config.stage_hidden_size)

        # Shared trunk: combines pooled anchor states with all probe features.
        self.trunk = nn.Sequential(
            nn.Linear(decision_dim, trunk_hidden),
            nn.GELU(),
        )

        # Stage 1: Modality informativeness (binary, independent heads).
        self.vision_info_head = nn.Linear(trunk_hidden, n_info)
        self.text_info_head = nn.Linear(trunk_hidden, n_info)

        # Stage 2: Pairwise relation, conditioned on stage-1 probabilities.
        # Input = trunk output + vision_info_probs + text_info_probs
        relation_input_dim = trunk_hidden + n_info + n_info
        self.relation_stage = nn.Sequential(
            nn.Linear(relation_input_dim, stage_hidden),
            nn.GELU(),
        )
        self.relation_head = nn.Linear(stage_hidden, n_rel)

        # Stage 3: Action, conditioned on stage-1 + stage-2 probabilities.
        # Input = trunk output + vision_info_probs + text_info_probs + relation_probs
        action_input_dim = trunk_hidden + n_info + n_info + n_rel
        self.action_stage = nn.Sequential(
            nn.Linear(action_input_dim, stage_hidden),
            nn.GELU(),
        )
        self.action_head = nn.Linear(stage_hidden, n_act)

    def pool_anchor_states(self, anchor_states: torch.Tensor) -> torch.Tensor:
        if anchor_states.dim() == 3:
            if self.config.pool == "mean":
                return anchor_states.mean(dim=1)
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")
        if anchor_states.dim() == 2:
            if self.config.pool == "mean":
                return anchor_states.mean(dim=0)
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")
        raise ValueError("anchor_states must have 2 or 3 dims")

    def forward(
        self,
        anchor_states: torch.Tensor,
        phi_v: torch.Tensor,
        phi_t: torch.Tensor,
        phi_cross: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pool_anchor_states(anchor_states)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        if phi_v.dim() == 1:
            phi_v = phi_v.unsqueeze(0)
        if phi_t.dim() == 1:
            phi_t = phi_t.unsqueeze(0)

        parts = [pooled, phi_v, phi_t]
        if phi_cross is not None:
            if phi_cross.dim() == 1:
                phi_cross = phi_cross.unsqueeze(0)
            parts.append(phi_cross)
        else:
            # Zero-fill cross-modal features if not provided (backward compat).
            zero = torch.zeros(pooled.shape[0], self.config.cross_modal_feature_size, device=pooled.device, dtype=pooled.dtype)
            parts.append(zero)

        features = torch.cat(parts, dim=-1)
        trunk_out = self.trunk(features)  # [B, trunk_hidden]

        # Stage 1: informativeness
        vision_info_logits = self.vision_info_head(trunk_out)
        text_info_logits = self.text_info_head(trunk_out)

        # Stage 2: relation (soft conditioning on stage 1)
        vision_info_probs = torch.softmax(vision_info_logits, dim=-1)
        text_info_probs = torch.softmax(text_info_logits, dim=-1)
        rel_in = torch.cat([trunk_out, vision_info_probs, text_info_probs], dim=-1)
        relation_features = self.relation_stage(rel_in)
        relation_logits = self.relation_head(relation_features)

        # Stage 3: action (soft conditioning on stages 1 and 2)
        relation_probs = torch.softmax(relation_logits, dim=-1)
        act_in = torch.cat([trunk_out, vision_info_probs, text_info_probs, relation_probs], dim=-1)
        action_features = self.action_stage(act_in)
        action_logits = self.action_head(action_features)

        return vision_info_logits, text_info_logits, relation_logits, action_logits


@dataclass
class FlatHiddenCARMConfig:
    """Flat single-action-head classifier on frozen hidden states.

    Ablation control for CascadeCARMHeads: uses the identical 139-dim input
    (pooled hidden states + phi_v + phi_t + phi_cross) but replaces the
    three-stage cascade with a single trunk → action head, removing both the
    auxiliary informativeness/relation supervision and the staged conditioning.

    This isolates the contribution of the cascade architecture from the
    contribution of the hidden-state input representation.
    """
    hidden_size: int = 128
    probe_feature_size: int = 3
    cross_modal_feature_size: int = 5
    pool: str = "mean"
    trunk_hidden_size: int | None = None


class FlatHiddenCARMHeads(nn.Module):
    """Single flat 4-way action head on the full 139-dim hidden-state input.

    Matches CascadeCARMHeads input exactly but omits auxiliary heads and
    cascade conditioning. Returns zero tensors for the unused info/relation
    outputs to preserve interface compatibility.
    """

    def __init__(self, config: FlatHiddenCARMConfig | None = None) -> None:
        super().__init__()
        self.config = config or FlatHiddenCARMConfig()
        n_act = len(ACTION_LABELS)  # 4

        decision_dim = (
            self.config.hidden_size
            + 2 * self.config.probe_feature_size
            + self.config.cross_modal_feature_size
        )
        trunk_hidden = int(self.config.trunk_hidden_size or self.config.hidden_size)

        self.trunk = nn.Sequential(
            nn.Linear(decision_dim, trunk_hidden),
            nn.GELU(),
        )
        self.action_head = nn.Linear(trunk_hidden, n_act)

    def pool_anchor_states(self, anchor_states: torch.Tensor) -> torch.Tensor:
        if anchor_states.dim() == 3:
            if self.config.pool == "mean":
                return anchor_states.mean(dim=1)
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")
        if anchor_states.dim() == 2:
            if self.config.pool == "mean":
                return anchor_states.mean(dim=0)
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")
        raise ValueError("anchor_states must have 2 or 3 dims")

    def forward(
        self,
        anchor_states: torch.Tensor,
        phi_v: torch.Tensor,
        phi_t: torch.Tensor,
        phi_cross: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pool_anchor_states(anchor_states)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        if phi_v.dim() == 1:
            phi_v = phi_v.unsqueeze(0)
        if phi_t.dim() == 1:
            phi_t = phi_t.unsqueeze(0)

        parts = [pooled, phi_v, phi_t]
        if phi_cross is not None:
            if phi_cross.dim() == 1:
                phi_cross = phi_cross.unsqueeze(0)
            parts.append(phi_cross)
        else:
            zero = torch.zeros(pooled.shape[0], self.config.cross_modal_feature_size, device=pooled.device, dtype=pooled.dtype)
            parts.append(zero)

        features = torch.cat(parts, dim=-1)
        trunk_out = self.trunk(features)
        action_logits = self.action_head(trunk_out)

        # Return zero tensors for unused heads (interface compatibility)
        dummy_info = torch.zeros(pooled.shape[0], len(INFO_STATE_LABELS), device=pooled.device, dtype=pooled.dtype)
        dummy_rel = torch.zeros(pooled.shape[0], len(PAIRWISE_RELATION_LABELS), device=pooled.device, dtype=pooled.dtype)
        return dummy_info, dummy_info, dummy_rel, action_logits


@dataclass
class DistributionCARMConfig:
    """Configuration for distribution-based cascade CARM.

    Uses the three answer distributions (multimodal, vision-only, text-only)
    as primary inputs instead of truncated hidden states. The answer_dist is
    already computed and cached, is 35-dim (closed vocab), and directly encodes
    what each modality predicts — making CARM a principled distribution arbitrator.

    Input: [mm_dist(vocab_size), v_dist(vocab_size), t_dist(vocab_size), phi_cross(5)]
    Total: 3*vocab_size + cross_modal_feature_size (default: 3*35+5 = 110-dim)
    """
    vocab_size: int = 35
    cross_modal_feature_size: int = 5
    trunk_hidden_size: int = 128
    stage_hidden_size: int = 64


class DistributionCARMHeads(nn.Module):
    """Cascade CARM operating on answer distributions rather than hidden states.

    The key design principle: CARM should learn to arbitrate based on *what each
    modality predicts*, not opaque internal representations. Answer distributions
    encode exactly this — probability over the closed answer vocabulary (35 tokens:
    yes/no, integers 0-20, color words).

    Architecture:
      Input: [mm_dist(35), v_dist(35), t_dist(35), phi_cross(5)] = 110-dim

    Cascade stages (same as CascadeCARMHeads):
      Stage 1: informativeness (vision_info, text_info)
      Stage 2: pairwise relation ← conditioned on stage-1
      Stage 3: joint action ← conditioned on stages 1+2

    Paper framing: "CARM observes modality-conditional answer distributions and
    routes based on their agreement structure — trusting, combining, or abstaining."
    """

    def __init__(self, config: DistributionCARMConfig | None = None) -> None:
        super().__init__()
        self.config = config or DistributionCARMConfig()
        n_info = len(INFO_STATE_LABELS)       # 2
        n_rel = len(PAIRWISE_RELATION_LABELS)  # 4
        n_act = len(ACTION_LABELS)             # 4

        decision_dim = (
            3 * self.config.vocab_size
            + self.config.cross_modal_feature_size
        )
        trunk_hidden = int(self.config.trunk_hidden_size)
        stage_hidden = int(self.config.stage_hidden_size)

        # Trunk: encodes the three distributions + agreement features jointly.
        self.trunk = nn.Sequential(
            nn.Linear(decision_dim, trunk_hidden),
            nn.GELU(),
        )

        # Stage 1: Modality informativeness.
        self.vision_info_head = nn.Linear(trunk_hidden, n_info)
        self.text_info_head = nn.Linear(trunk_hidden, n_info)

        # Stage 2: Pairwise relation, conditioned on stage-1.
        relation_input_dim = trunk_hidden + n_info + n_info
        self.relation_stage = nn.Sequential(
            nn.Linear(relation_input_dim, stage_hidden),
            nn.GELU(),
        )
        self.relation_head = nn.Linear(stage_hidden, n_rel)

        # Stage 3: Action, conditioned on stages 1 and 2.
        action_input_dim = trunk_hidden + n_info + n_info + n_rel
        self.action_stage = nn.Sequential(
            nn.Linear(action_input_dim, stage_hidden),
            nn.GELU(),
        )
        self.action_head = nn.Linear(stage_hidden, n_act)

    def _pad_or_trim(self, dist: torch.Tensor, target_size: int) -> torch.Tensor:
        """Pad (with uniform) or trim distribution to target_size."""
        if dist.dim() == 1:
            dist = dist.unsqueeze(0)
        current = dist.shape[-1]
        if current == target_size:
            return dist
        if current < target_size:
            pad = torch.full(
                (dist.shape[0], target_size - current),
                1.0 / target_size,
                dtype=dist.dtype, device=dist.device,
            )
            return torch.cat([dist, pad], dim=-1)
        return dist[:, :target_size]

    def forward(
        self,
        mm_dist: torch.Tensor,
        v_dist: torch.Tensor,
        t_dist: torch.Tensor,
        phi_cross: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        V = self.config.vocab_size
        if mm_dist.dim() == 1:
            mm_dist = mm_dist.unsqueeze(0)
        if v_dist.dim() == 1:
            v_dist = v_dist.unsqueeze(0)
        if t_dist.dim() == 1:
            t_dist = t_dist.unsqueeze(0)

        # Normalize to proper probability distributions.
        mm_dist = torch.softmax(mm_dist.float(), dim=-1)
        v_dist = torch.softmax(v_dist.float(), dim=-1)
        t_dist = torch.softmax(t_dist.float(), dim=-1)

        # Pad/trim to fixed vocab_size.
        mm_dist = self._pad_or_trim(mm_dist, V)
        v_dist = self._pad_or_trim(v_dist, V)
        t_dist = self._pad_or_trim(t_dist, V)

        if phi_cross is not None:
            if phi_cross.dim() == 1:
                phi_cross = phi_cross.unsqueeze(0)
        else:
            phi_cross = torch.zeros(
                mm_dist.shape[0], self.config.cross_modal_feature_size,
                device=mm_dist.device, dtype=mm_dist.dtype,
            )

        features = torch.cat([mm_dist, v_dist, t_dist, phi_cross], dim=-1)
        trunk_out = self.trunk(features)

        # Stage 1: informativeness.
        vision_info_logits = self.vision_info_head(trunk_out)
        text_info_logits = self.text_info_head(trunk_out)

        # Stage 2: relation.
        vision_info_probs = torch.softmax(vision_info_logits, dim=-1)
        text_info_probs = torch.softmax(text_info_logits, dim=-1)
        rel_in = torch.cat([trunk_out, vision_info_probs, text_info_probs], dim=-1)
        relation_logits = self.relation_head(self.relation_stage(rel_in))

        # Stage 3: action.
        relation_probs = torch.softmax(relation_logits, dim=-1)
        act_in = torch.cat([trunk_out, vision_info_probs, text_info_probs, relation_probs], dim=-1)
        action_logits = self.action_head(self.action_stage(act_in))

        return vision_info_logits, text_info_logits, relation_logits, action_logits


def decode_action(action_logits: torch.Tensor) -> Action:
    idx = int(torch.argmax(action_logits, dim=-1).item())
    return Action(ACTION_LABELS[idx])


def decode_info_state(info_logits: torch.Tensor) -> str:
    idx = int(torch.argmax(info_logits, dim=-1).item())
    return INFO_STATE_LABELS[idx]


def decode_pairwise_relation(relation_logits: torch.Tensor) -> str:
    idx = int(torch.argmax(relation_logits, dim=-1).item())
    return PAIRWISE_RELATION_LABELS[idx]
