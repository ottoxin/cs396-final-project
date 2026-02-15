from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ConflictType(str, Enum):
    NONE = "none"
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    RELATION = "relation"
    COUNT = "count"


class CorruptedModality(str, Enum):
    NONE = "none"
    VISION = "vision"
    TEXT = "text"


class EvidenceModality(str, Enum):
    VISION_REQUIRED = "vision_required"
    TEXT_REQUIRED = "text_required"
    BOTH = "both"
    EITHER = "either"


class Action(str, Enum):
    TRUST_VISION = "trust_vision"
    TRUST_TEXT = "trust_text"
    REQUIRE_AGREEMENT = "require_agreement"
    ABSTAIN = "abstain"


@dataclass
class ConflictExample:
    example_id: str
    image_path: str
    text_input: str
    question: str
    gold_answer: str
    split: Split
    conflict_type: ConflictType
    corrupted_modality: CorruptedModality
    corruption_family: str
    severity: int
    evidence_modality: EvidenceModality
    oracle_action: Action
    source_image_id: str | None = None
    template_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ConflictExample":
        return cls(
            example_id=str(item["example_id"]),
            image_path=str(item["image_path"]),
            text_input=str(item["text_input"]),
            question=str(item["question"]),
            gold_answer=str(item["gold_answer"]),
            split=Split(item["split"]),
            conflict_type=ConflictType(item["conflict_type"]),
            corrupted_modality=CorruptedModality(item["corrupted_modality"]),
            corruption_family=str(item["corruption_family"]),
            severity=int(item["severity"]),
            evidence_modality=EvidenceModality(item["evidence_modality"]),
            oracle_action=Action(item["oracle_action"]),
            source_image_id=item.get("source_image_id"),
            template_id=item.get("template_id"),
            metadata=item.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["split"] = self.split.value
        item["conflict_type"] = self.conflict_type.value
        item["corrupted_modality"] = self.corrupted_modality.value
        item["evidence_modality"] = self.evidence_modality.value
        item["oracle_action"] = self.oracle_action.value
        return item
