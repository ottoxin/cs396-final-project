from __future__ import annotations

from dataclasses import dataclass

from carm.data.schema import Action, CorruptModality, EvidenceModality


MAX_SEVERITY_DEFAULT = 3


@dataclass(frozen=True)
class ReliabilityTarget:
    r_v: float
    r_t: float


def derive_oracle_action(
    corrupt_modality: CorruptModality,
    is_ambiguous: bool = False,
) -> Action:
    if is_ambiguous or corrupt_modality == CorruptModality.BOTH:
        return Action.ABSTAIN
    if corrupt_modality == CorruptModality.TEXT:
        return Action.TRUST_VISION
    if corrupt_modality == CorruptModality.VISION:
        return Action.TRUST_TEXT
    return Action.REQUIRE_AGREEMENT


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
    corrupt_modality: CorruptModality,
    severity: int,
    max_severity: int = MAX_SEVERITY_DEFAULT,
) -> ReliabilityTarget:
    r_v, r_t = _base_reliability(evidence_modality)
    severity = max(0, min(severity, max_severity))
    penalty = max(0.2, severity / max_severity) if max_severity > 0 else 0.0

    if corrupt_modality == CorruptModality.VISION:
        r_v = max(0.05, r_v * (1.0 - penalty))
    elif corrupt_modality == CorruptModality.TEXT:
        r_t = max(0.05, r_t * (1.0 - penalty))
    elif corrupt_modality == CorruptModality.BOTH:
        r_v = max(0.05, r_v * (1.0 - penalty))
        r_t = max(0.05, r_t * (1.0 - penalty))

    return ReliabilityTarget(r_v=round(r_v, 4), r_t=round(r_t, 4))
