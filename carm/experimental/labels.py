from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from carm.data.schema import Action, ConflictExample


INFO_STATE_INFORMATIVE = "informative"
INFO_STATE_UNINFORMATIVE = "uninformative"

PAIRWISE_RELATION_CONSISTENT = "consistent"
PAIRWISE_RELATION_CONTRADICTORY = "contradictory"
PAIRWISE_RELATION_ASYMMETRIC = "asymmetric"
PAIRWISE_RELATION_BOTH_WEAK = "both_weak"

RELIABILITY_VISION_HIGH_TEXT_LOW = "vision_high_text_low"
RELIABILITY_TEXT_HIGH_VISION_LOW = "text_high_vision_low"
RELIABILITY_BOTH_HIGH = "both_high"
RELIABILITY_BOTH_LOW = "both_low"

INFO_STATE_LABELS = (
    INFO_STATE_INFORMATIVE,
    INFO_STATE_UNINFORMATIVE,
)
INFO_STATE_TO_IDX = {label: idx for idx, label in enumerate(INFO_STATE_LABELS)}

PAIRWISE_RELATION_LABELS = (
    PAIRWISE_RELATION_CONSISTENT,
    PAIRWISE_RELATION_CONTRADICTORY,
    PAIRWISE_RELATION_ASYMMETRIC,
    PAIRWISE_RELATION_BOTH_WEAK,
)
PAIRWISE_RELATION_TO_IDX = {label: idx for idx, label in enumerate(PAIRWISE_RELATION_LABELS)}

ACTION_LABELS = tuple(action.value for action in Action)
ACTION_TO_IDX = {label: idx for idx, label in enumerate(ACTION_LABELS)}

SOURCE_FIELDS_BASE = (
    "metadata.image_state",
    "metadata.caption_state",
    "metadata.protocol_category",
    "oracle_action",
    "gold_answer",
    "vision_supported_target",
    "text_supported_target",
    "vision_info_state",
    "text_info_state",
    "pairwise_relation",
    "joint_answer",
)


def _metadata_value(example: ConflictExample, key: str) -> Any:
    if not isinstance(example.metadata, dict):
        return None
    return example.metadata.get(key)


def _flatten_source_fields(example: ConflictExample) -> dict[str, Any]:
    return {
        "metadata.image_state": _metadata_value(example, "image_state"),
        "metadata.caption_state": _metadata_value(example, "caption_state"),
        "metadata.protocol_category": _metadata_value(example, "protocol_category"),
        "oracle_action": example.oracle_action.value,
        "gold_answer": example.gold_answer,
        "vision_supported_target": example.vision_supported_target,
        "text_supported_target": example.text_supported_target,
        "vision_info_state": example.vision_info_state,
        "text_info_state": example.text_info_state,
        "pairwise_relation": example.pairwise_relation,
        "joint_answer": example.joint_answer,
    }


def _map_explicit_info_state(raw: Any) -> str | None:
    value = str(raw or "").strip().lower()
    mapping = {
        "informative": INFO_STATE_INFORMATIVE,
        "supportive": INFO_STATE_INFORMATIVE,
        "relevant": INFO_STATE_INFORMATIVE,
        "contradictory": INFO_STATE_INFORMATIVE,
        "uninformative": INFO_STATE_UNINFORMATIVE,
        "irrelevant": INFO_STATE_UNINFORMATIVE,
        "weak": INFO_STATE_UNINFORMATIVE,
    }
    return mapping.get(value)


def _map_image_state(raw: Any) -> str | None:
    value = str(raw or "").strip().lower()
    mapping = {
        "clean": INFO_STATE_INFORMATIVE,
        "relevant": INFO_STATE_INFORMATIVE,
        "supportive": INFO_STATE_INFORMATIVE,
        "irrelevant": INFO_STATE_UNINFORMATIVE,
    }
    return mapping.get(value)


def _map_text_state(raw: Any) -> str | None:
    value = str(raw or "").strip().lower()
    mapping = {
        "clean": INFO_STATE_INFORMATIVE,
        "relevant": INFO_STATE_INFORMATIVE,
        "supportive": INFO_STATE_INFORMATIVE,
        "different": INFO_STATE_INFORMATIVE,
        "contradictory": INFO_STATE_INFORMATIVE,
        "irrelevant": INFO_STATE_UNINFORMATIVE,
    }
    return mapping.get(value)


def _map_explicit_pairwise_relation(raw: Any) -> str | None:
    value = str(raw or "").strip().lower()
    mapping = {
        "consistent": PAIRWISE_RELATION_CONSISTENT,
        "contradictory": PAIRWISE_RELATION_CONTRADICTORY,
        "asymmetric": PAIRWISE_RELATION_ASYMMETRIC,
        "one_informative_one_irrelevant": PAIRWISE_RELATION_ASYMMETRIC,
        "both_weak": PAIRWISE_RELATION_BOTH_WEAK,
    }
    return mapping.get(value)


def joint_info_state_label(vision_info_state: str, text_info_state: str) -> str | None:
    if vision_info_state not in INFO_STATE_TO_IDX or text_info_state not in INFO_STATE_TO_IDX:
        return None
    return f"vision_{vision_info_state}__text_{text_info_state}"


def decode_joint_info_state(label: str | None) -> tuple[str | None, str | None]:
    if not label or "__" not in label:
        return None, None
    left, right = label.split("__", 1)
    return left.replace("vision_", "", 1), right.replace("text_", "", 1)


def derive_pairwise_relation(
    vision_info_state: str,
    text_info_state: str,
    *,
    caption_state_raw: Any = None,
    protocol_category: str | None = None,
) -> str | None:
    category = str(protocol_category or "").strip().upper()
    caption_state = str(caption_state_raw or "").strip().lower()
    if category == "C1" or caption_state in {"clean", "supportive", "relevant"}:
        if vision_info_state == INFO_STATE_INFORMATIVE and text_info_state == INFO_STATE_INFORMATIVE:
            return PAIRWISE_RELATION_CONSISTENT
    if category == "C4" or caption_state in {"different", "contradictory"}:
        if vision_info_state == INFO_STATE_INFORMATIVE and text_info_state == INFO_STATE_INFORMATIVE:
            return PAIRWISE_RELATION_CONTRADICTORY
    if (
        vision_info_state == INFO_STATE_INFORMATIVE and text_info_state == INFO_STATE_UNINFORMATIVE
    ) or (
        vision_info_state == INFO_STATE_UNINFORMATIVE and text_info_state == INFO_STATE_INFORMATIVE
    ):
        return PAIRWISE_RELATION_ASYMMETRIC
    if vision_info_state == INFO_STATE_UNINFORMATIVE and text_info_state == INFO_STATE_UNINFORMATIVE:
        return PAIRWISE_RELATION_BOTH_WEAK
    return None


def derive_reliability_proxy(vision_info_state: str, text_info_state: str) -> str:
    vision_high = vision_info_state == INFO_STATE_INFORMATIVE
    text_high = text_info_state == INFO_STATE_INFORMATIVE
    if vision_high and text_high:
        return RELIABILITY_BOTH_HIGH
    if vision_high:
        return RELIABILITY_VISION_HIGH_TEXT_LOW
    if text_high:
        return RELIABILITY_TEXT_HIGH_VISION_LOW
    return RELIABILITY_BOTH_LOW


def reliability_proxy_vector(label: str) -> tuple[float, float]:
    mapping = {
        RELIABILITY_VISION_HIGH_TEXT_LOW: (1.0, 0.0),
        RELIABILITY_TEXT_HIGH_VISION_LOW: (0.0, 1.0),
        RELIABILITY_BOTH_HIGH: (1.0, 1.0),
        RELIABILITY_BOTH_LOW: (0.0, 0.0),
    }
    out = mapping.get(label)
    if out is None:
        raise ValueError(f"Unsupported reliability label: {label}")
    return out


def action_from_structure(
    vision_info_state: str,
    text_info_state: str,
    pairwise_relation: str,
) -> tuple[str | None, str]:
    if pairwise_relation == PAIRWISE_RELATION_CONSISTENT:
        return Action.REQUIRE_AGREEMENT.value, "success"
    if pairwise_relation == PAIRWISE_RELATION_CONTRADICTORY:
        return Action.ABSTAIN.value, "success"
    if pairwise_relation == PAIRWISE_RELATION_BOTH_WEAK:
        return Action.ABSTAIN.value, "success"
    if pairwise_relation == PAIRWISE_RELATION_ASYMMETRIC:
        if vision_info_state == INFO_STATE_INFORMATIVE and text_info_state == INFO_STATE_UNINFORMATIVE:
            return Action.TRUST_VISION.value, "success"
        if vision_info_state == INFO_STATE_UNINFORMATIVE and text_info_state == INFO_STATE_INFORMATIVE:
            return Action.TRUST_TEXT.value, "success"
    return None, "failed"


@dataclass(frozen=True)
class DerivedLabels:
    example_id: str
    source_fields_used: dict[str, Any]
    vision_info_state: str | None
    text_info_state: str | None
    joint_info_state: str | None
    pairwise_relation: str | None
    reliability_proxy_target: str | None
    action_target: str | None
    derivation_status: str
    reason: str
    vision_info_target_available: bool
    text_info_target_available: bool
    relation_target_available: bool
    action_target_available: bool
    legacy_oracle_action: str
    metric_semantics_mismatch: bool

    @property
    def info_target_available(self) -> bool:
        return bool(self.vision_info_target_available and self.text_info_target_available)

    @property
    def reliability_target_available(self) -> bool:
        return self.reliability_proxy_target is not None

    def reliability_vector(self) -> tuple[float, float] | None:
        if self.reliability_proxy_target is None:
            return None
        return reliability_proxy_vector(self.reliability_proxy_target)

    def to_audit_row(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_fields_used": json.dumps(self.source_fields_used, ensure_ascii=True, sort_keys=True),
            "vision_info_state": self.vision_info_state,
            "text_info_state": self.text_info_state,
            "joint_info_state": self.joint_info_state,
            "pairwise_relation": self.pairwise_relation,
            "reliability_proxy_target": self.reliability_proxy_target,
            "action_target": self.action_target,
            "derivation_status": self.derivation_status,
            "reason": self.reason,
            "vision_info_target_available": self.vision_info_target_available,
            "text_info_target_available": self.text_info_target_available,
            "relation_target_available": self.relation_target_available,
            "info_target_available": self.info_target_available,
            "reliability_target_available": self.reliability_target_available,
            "action_target_available": self.action_target_available,
            "legacy_oracle_action": self.legacy_oracle_action,
            "metric_semantics_mismatch": self.metric_semantics_mismatch,
        }


def derive_labels(example: ConflictExample) -> DerivedLabels:
    source_fields = _flatten_source_fields(example)
    image_state_raw = source_fields["metadata.image_state"]
    caption_state_raw = source_fields["metadata.caption_state"]
    protocol_category = str(source_fields["metadata.protocol_category"] or "").strip().upper()

    vision_info_state = _map_explicit_info_state(source_fields["vision_info_state"]) or _map_image_state(image_state_raw)
    text_info_state = _map_explicit_info_state(source_fields["text_info_state"]) or _map_text_state(caption_state_raw)
    pairwise_relation = _map_explicit_pairwise_relation(source_fields["pairwise_relation"])
    if pairwise_relation is None and vision_info_state is not None and text_info_state is not None:
        pairwise_relation = derive_pairwise_relation(
            vision_info_state,
            text_info_state,
            caption_state_raw=caption_state_raw,
            protocol_category=protocol_category,
        )

    if vision_info_state is None or text_info_state is None or pairwise_relation is None:
        missing = []
        if vision_info_state is None:
            missing.append(f"unsupported image_state={image_state_raw!r}")
        if text_info_state is None:
            missing.append(f"unsupported caption_state={caption_state_raw!r}")
        if pairwise_relation is None:
            missing.append(
                "pairwise relation unavailable from explicit fields and source states"
            )
        return DerivedLabels(
            example_id=example.example_id,
            source_fields_used=source_fields,
            vision_info_state=vision_info_state,
            text_info_state=text_info_state,
            joint_info_state=(
                joint_info_state_label(vision_info_state, text_info_state)
                if vision_info_state and text_info_state
                else None
            ),
            pairwise_relation=pairwise_relation,
            reliability_proxy_target=(
                derive_reliability_proxy(vision_info_state, text_info_state)
                if vision_info_state and text_info_state
                else None
            ),
            action_target=None,
            derivation_status="failed",
            reason="; ".join(missing) or "missing modality-state metadata",
            vision_info_target_available=vision_info_state is not None,
            text_info_target_available=text_info_state is not None,
            relation_target_available=pairwise_relation is not None,
            action_target_available=False,
            legacy_oracle_action=example.oracle_action.value,
            metric_semantics_mismatch=False,
        )

    action_target, action_status = action_from_structure(
        vision_info_state,
        text_info_state,
        pairwise_relation,
    )
    derivation_status = "success" if action_status == "success" else "failed"
    reason = "all requested revised labels derived from explicit fields or source-state metadata"
    if action_status != "success":
        reason = "joint action could not be derived from modality informativeness plus pairwise relation"

    return DerivedLabels(
        example_id=example.example_id,
        source_fields_used=source_fields,
        vision_info_state=vision_info_state,
        text_info_state=text_info_state,
        joint_info_state=joint_info_state_label(vision_info_state, text_info_state),
        pairwise_relation=pairwise_relation,
        reliability_proxy_target=derive_reliability_proxy(vision_info_state, text_info_state),
        action_target=action_target,
        derivation_status=derivation_status,
        reason=reason,
        vision_info_target_available=True,
        text_info_target_available=True,
        relation_target_available=True,
        action_target_available=action_target is not None,
        legacy_oracle_action=example.oracle_action.value,
        metric_semantics_mismatch=False,
    )


def decode_reliability_prediction(r_v: float | None, r_t: float | None, threshold: float = 0.5) -> str | None:
    if r_v is None or r_t is None:
        return None
    vision_high = float(r_v) >= threshold
    text_high = float(r_t) >= threshold
    if vision_high and text_high:
        return RELIABILITY_BOTH_HIGH
    if vision_high:
        return RELIABILITY_VISION_HIGH_TEXT_LOW
    if text_high:
        return RELIABILITY_TEXT_HIGH_VISION_LOW
    return RELIABILITY_BOTH_LOW
