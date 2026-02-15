from __future__ import annotations

from dataclasses import dataclass

from carm.data.schema import Action, CorruptedModality, EvidenceModality


MAX_SEVERITY_DEFAULT = 3


@dataclass(frozen=True)
class ReliabilityTarget:
    r_v: float
    r_t: float


# Independent oracle policy from evidence requirement and corruption state.
ORACLE_ACTION_TABLE: dict[tuple[EvidenceModality, CorruptedModality], Action] = {
    (EvidenceModality.VISION_REQUIRED, CorruptedModality.NONE): Action.TRUST_VISION,
    (EvidenceModality.TEXT_REQUIRED, CorruptedModality.NONE): Action.TRUST_TEXT,
    (EvidenceModality.BOTH, CorruptedModality.NONE): Action.REQUIRE_AGREEMENT,
    (EvidenceModality.EITHER, CorruptedModality.NONE): Action.REQUIRE_AGREEMENT,
    (EvidenceModality.VISION_REQUIRED, CorruptedModality.VISION): Action.ABSTAIN,
    (EvidenceModality.TEXT_REQUIRED, CorruptedModality.VISION): Action.TRUST_TEXT,
    (EvidenceModality.BOTH, CorruptedModality.VISION): Action.ABSTAIN,
    (EvidenceModality.EITHER, CorruptedModality.VISION): Action.TRUST_TEXT,
    (EvidenceModality.VISION_REQUIRED, CorruptedModality.TEXT): Action.TRUST_VISION,
    (EvidenceModality.TEXT_REQUIRED, CorruptedModality.TEXT): Action.ABSTAIN,
    (EvidenceModality.BOTH, CorruptedModality.TEXT): Action.ABSTAIN,
    (EvidenceModality.EITHER, CorruptedModality.TEXT): Action.TRUST_VISION,
}


def derive_oracle_action(
    evidence_modality: EvidenceModality,
    corrupted_modality: CorruptedModality,
) -> Action:
    return ORACLE_ACTION_TABLE[(evidence_modality, corrupted_modality)]


def _base_reliability(evidence_modality: EvidenceModality) -> tuple[float, float]:
    if evidence_modality == EvidenceModality.VISION_REQUIRED:
        return 1.0, 0.0
    if evidence_modality == EvidenceModality.TEXT_REQUIRED:
        return 0.0, 1.0
    if evidence_modality == EvidenceModality.BOTH:
        return 1.0, 1.0
    return 0.5, 0.5


def derive_reliability_target(
    evidence_modality: EvidenceModality,
    corrupted_modality: CorruptedModality,
    severity: int,
    max_severity: int = MAX_SEVERITY_DEFAULT,
) -> ReliabilityTarget:
    """
    Construct supervision targets for r_v and r_t from modality sufficiency + corruption metadata.
    Severity penalty is monotonic and bounded to avoid collapse to exact zeros.
    """
    r_v, r_t = _base_reliability(evidence_modality)
    severity = max(0, min(severity, max_severity))
    penalty = max(0.2, severity / max_severity) if max_severity > 0 else 0.0

    if corrupted_modality == CorruptedModality.VISION:
        r_v = max(0.05, r_v * (1.0 - penalty))
    elif corrupted_modality == CorruptedModality.TEXT:
        r_t = max(0.05, r_t * (1.0 - penalty))

    return ReliabilityTarget(r_v=round(r_v, 4), r_t=round(r_t, 4))
